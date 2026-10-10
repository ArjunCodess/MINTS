"""Demonstrate correspondence, or diagnose every saved case using cached tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.query_correspondence import certify_query_correspondence

ROOT = Path(__file__).resolve().parents[1]


def demonstration():
    # Hand-constructed tokenizations test the contract; these are not a model run.
    sequences = ['AAAACCCC', 'ATAACCCC', 'AAATCCCC']
    maps = [dict(offsets=[[0,0],[0,4],[4,8],[0,0]], ids=[0,10,11,1]),
            dict(offsets=[[0,0],[0,2],[2,4],[4,8],[0,0]], ids=[0,20,21,11,1]),
            dict(offsets=[[0,0],[0,1],[1,4],[4,8],[0,0]], ids=[0,30,31,11,1])]
    return dict(example='Hand-constructed contract example; no learned-mechanism claim',
                **certify_query_correspondence(sequences, maps, edit_positions=[1,3], mask_token_id=99))


def saved_cases():
    from transformers import AutoTokenizer
    from src.config import DEFAULT_CONFIG
    from src.control_diagnostics import token_map, common_queries, same_side_queries
    from src.controlled_edits import MotifDefinition
    from src.motif_scoring import find_jaspar_matrix_path, load_jaspar_ctcf_motif, motif_pssm
    from src.variant_protocol import VariantProtocol
    from src.utils import sha256_file

    source = ROOT/'results/mapped_variant/cases.json'
    protocol = VariantProtocol()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,
        revision=DEFAULT_CONFIG.model.revision, trust_remote_code=True, local_files_only=True)
    matrix = find_jaspar_matrix_path()
    motif = MotifDefinition('CTCF', pssm=motif_pssm(load_jaspar_ctcf_motif(matrix)), fraction=.8)
    cases = json.loads(source.read_text())
    rows = []
    for case in cases:
        sequences = [case[k] for k in ('reference_sequence','alternate_sequence','sham_sequence')]
        maps = [token_map(tokenizer, s) for s in sequences]
        if case['ids'] != [m[1] for m in maps]:
            raise ValueError('Saved IDs differ from the pinned tokenizer')
        hits = motif.hits(sequences[0])
        certificate = certify_query_correspondence(sequences,
            [dict(offsets=m[0],ids=m[1]) for m in maps],
            edit_positions=[case['variant_index'],case['sham_index']],
            motif_spans=[h[:2] for h in hits], radius_bp=protocol.query_radius_bp,
            mask_token_id=tokenizer.mask_token_id)
        historical = same_side_queries(common_queries(maps,case['variant_index'],hits,
            protocol.query_radius_bp),case['variant_index'],case['sham_index'])
        if certificate['queries'] != historical or historical != case['queries']:
            raise ValueError('Certificate differs from the saved complete query set')
        if certificate['full_boundaries_equal'] != case['full_boundary_stable'] or certificate['shifted_query_indices'] != case['query_index_shift']:
            raise ValueError('Saved boundary/index flags disagree')
        rows.append(dict(variant_id=case['variant_id'],cluster=case['cluster'],
            sequence_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in sequences], **certificate))
    generators = [ROOT/'src/query_correspondence.py',Path(__file__).resolve(),ROOT/'src/control_diagnostics.py']
    return dict(scope='Post-study saved-input diagnostic; no model predictions, new controls, endpoint changes or gate decisions',
        cases=len(rows),clusters=len({r['cluster'] for r in rows}),
        query_count=sum(r['accepted_queries'] for r in rows),
        shifted_index_cases=sum(r['shifted_query_indices'] for r in rows),
        full_boundary_stable_cases=sum(r['full_boundaries_equal'] for r in rows),
        exact_saved_query_sets=True, model_revision=DEFAULT_CONFIG.model.revision,
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        inputs={source.relative_to(ROOT).as_posix():sha256_file(source),
                'results/mapped_variant/protocol.json':sha256_file(ROOT/'results/mapped_variant/protocol.json'),
                'JASPAR MA0139.1':sha256_file(matrix)},
        generators={p.relative_to(ROOT).as_posix():sha256_file(p) for p in generators}, cases_checked=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-cases', action='store_true', help='Retokenize all saved cases locally; requires cached pinned tokenizer and JASPAR, no weights or genome')
    parser.add_argument('--output', type=Path, help='Write a new diagnostic JSON; existing files are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Choose a fresh diagnostic output path')
    result = saved_cases() if args.saved_cases else demonstration()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases_checked'},indent=2))
