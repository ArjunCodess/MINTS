"""Contract checks against a literal masked-input oracle, without model inference."""
from copy import deepcopy
import random
import pytest
from src.query_correspondence import certify_query_correspondence


def oracle(sequences, maps, edits, motifs, radius, mask):
    accepted = []
    for index, span in enumerate(maps[0]['offsets']):
        a, b = span
        if a == b or not (b <= min(edits) or a > max(edits)):
            continue
        gap = edits[0]-b if b <= edits[0] else a-edits[0]-1
        if gap > radius or any(a < d and c < b for c, d in motifs):
            continue
        token = maps[0]['ids'][index]
        indices = []
        for m in maps:
            candidates = [i for i, s in enumerate(m['offsets']) if s == span and m['ids'][i] == token]
            if len(candidates) != 1:
                break
            indices.append(candidates[0])
        if len(indices) != 3 or len({s[a:b] for s in sequences}) != 1:
            continue
        masked = []
        for m, i in zip(maps, indices):
            ids = list(m['ids'])
            ids[i] = mask
            masked.append(tuple(ids))
        if len(set(masked)) == 3:
            accepted.append(dict(span=span, indices=indices, token_id=token, width=b-a))
    return accepted


def test_shifted_indices_and_unequal_lengths_are_valid():
    from tools.check_query_correspondence import demonstration
    result = demonstration()
    assert result['queries'] == [dict(span=[4,8],indices=[2,3,3],token_id=11,width=4)]
    assert result['token_counts'] == [4,5,5]
    assert not result['full_boundaries_equal']


def test_literal_oracle_covers_partition_identity_content_and_masking():
    rng = random.Random(917)
    for _ in range(600):
        sequences = [''.join(rng.choices('ACGT', k=8)) for _ in range(3)]
        maps = []
        for sequence in sequences:
            ends = [0] + [i for i in range(1,8) if rng.random() < .6] + [8]
            spans = [[a,b] for a,b in zip(ends,ends[1:])]
            # Small repeated IDs deliberately test equal identity with unequal content.
            ids = [sum(map(ord,sequence[a:b])) % 4 + 2 for a,b in spans]
            maps.append(dict(offsets=[[0,0]]+spans+[[0,0]],ids=[0]+ids+[1]))
        edits = rng.sample(range(8),2)
        motifs = [[2,4],[3,5]] if rng.random() < .5 else []
        radius = rng.randrange(5)
        result = certify_query_correspondence(sequences,maps,edit_positions=edits,
            motif_spans=motifs,radius_bp=radius,mask_token_id=99)
        assert result['queries'] == oracle(sequences,maps,edits,motifs,radius,99)
        assert result['candidate_accounting_complete']


def fixture():
    sequences = ['AAAACCCC','ATAACCCC','AAATCCCC']
    maps = [dict(offsets=[[0,4],[4,8]],ids=[i,11]) for i in (2,3,4)]
    return sequences,maps


def test_half_open_motif_boundary_and_radius_zero():
    sequences,maps = fixture()
    args = dict(edit_positions=[3,1],radius_bp=0,mask_token_id=99)
    assert certify_query_correspondence(sequences,maps,motif_spans=[[0,4]],**args)['accepted_queries'] == 1
    assert certify_query_correspondence(sequences,maps,motif_spans=[[3,5]],**args)['accepted_queries'] == 0


def test_equal_shape_and_identity_do_not_certify_content_or_distinct_inputs():
    sequences,maps = fixture()
    sequences[1] = 'ATAACCCA'
    result = certify_query_correspondence(sequences,maps,edit_positions=[1,3],mask_token_id=99)
    assert result['rejected']['target_content'] == 1
    sequences,maps = fixture()
    maps[1] = deepcopy(maps[0])
    result = certify_query_correspondence(sequences,maps,edit_positions=[1,3],mask_token_id=99)
    assert result['rejected']['erased_contrast'] == 1


@pytest.mark.parametrize('offsets', [[[0,3],[4,8]],[[0,5],[4,8]],[[0,4],[0,4],[4,8]],[[0,4],[4,9]]])
def test_invalid_partition_fails(offsets):
    sequences,maps = fixture()
    maps[0] = dict(offsets=offsets,ids=[2]*len(offsets))
    with pytest.raises(ValueError):
        certify_query_correspondence(sequences,maps,edit_positions=[1,3],mask_token_id=99)


@pytest.mark.parametrize('edits,radius,mask', [([1,1],32,99),([True,3],32,99),([1,8],32,99),([1,3],-1,99),([1,3],32,11)])
def test_invalid_parameters_fail(edits,radius,mask):
    sequences,maps = fixture()
    with pytest.raises(ValueError):
        certify_query_correspondence(sequences,maps,edit_positions=edits,radius_bp=radius,mask_token_id=mask)
