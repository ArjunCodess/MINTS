"""Check nucleotide prediction-task correspondence without model inference.

This diagnostic does not change the frozen candidate search or scoring code.
Its certificate covers the declared tokenizations, not biological causality.
"""
from itertools import combinations


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    return value


def _span(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must have two endpoints')
    a, b = (_integer(v, name) for v in value)
    if not 0 <= a <= b <= length:
        raise ValueError(f'{name} is outside the nucleotide sequence')
    return a, b


def _tokenization(record, length, mask_token_id):
    offsets, ids = record['offsets'], record['ids']
    if len(offsets) != len(ids) or not offsets:
        raise ValueError('Token IDs and offsets must have equal nonzero lengths')
    intervals = [_span(s, length, 'Token span') for s in offsets]
    tokens = [_integer(t, 'Token ID') for t in ids]
    if any(t < 0 or t == mask_token_id for t in tokens):
        raise ValueError('Unmasked token IDs must be nonnegative and contain no mask')
    position = 0
    real = {}
    for i, (a, b) in enumerate(intervals):
        if a == b:
            continue
        if a != position:
            raise ValueError('Real token spans must be an ordered complete partition')
        real[a, b] = (i, tokens[i])
        position = b
    if position != length or not real:
        raise ValueError('Real token spans must cover the complete sequence')
    return intervals, tokens, real


def certify_query_correspondence(sequences, tokenizations, *, edit_positions,
                                motif_spans=(), radius_bp=32, mask_token_id):
    """Enumerate all accepted span/identity queries for the declared triplet.

    Real token spans must partition each canonical equal-length input. A query
    lies on the same side of both declared edit positions, outside motifs and
    within the radius of the first edit. All three inputs must assign it the
    same vocabulary ID and nucleotide content. Masking must leave three distinct
    token-ID inputs. Row indices and token counts may differ.

    The caller must establish that maps came from one pinned tokenizer/vocabulary
    and that motif spans and edit positions are the intended scientific inputs.
    No sequence-control, model-wiring, endpoint or biological gate is certified.
    """
    if len(sequences) != 3 or len(tokenizations) != 3:
        raise ValueError('A reference/alternate/sham triplet is required')
    if any(not isinstance(s, str) or not s or set(s) - set('ACGT') for s in sequences):
        raise ValueError('Canonical nonempty nucleotide inputs are required')
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError('Equal nucleotide lengths are required')
    if len(edit_positions) != 2:
        raise ValueError('Two distinct declared edit positions are required')
    edits = [_integer(i, 'Edit position') for i in edit_positions]
    if len(set(edits)) != 2 or any(not 0 <= i < length for i in edits):
        raise ValueError('Two distinct in-range edit positions are required')
    radius = _integer(radius_bp, 'Radius')
    mask = _integer(mask_token_id, 'Mask token ID')
    if radius < 0 or mask < 0:
        raise ValueError('Radius and mask token ID must be nonnegative')
    maps = [_tokenization(m, length, mask) for m in tokenizations]
    spans = sorted(_span(s, length, 'Motif span') for s in motif_spans)
    if any(a == b for a, b in spans):
        raise ValueError('Motif spans must have positive width')
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))

    # A mask removes an ID difference only when both maps use the same index.
    differences = {}
    for r, s in combinations(range(3), 2):
        left, right = maps[r][1], maps[s][1]
        differences[r, s] = None if len(left) != len(right) else sum(a != b for a, b in zip(left, right))
    queries, rejected = [], dict(motif_or_geometry=0, span_or_identity=0,
                                 target_content=0, erased_contrast=0)
    hit = 0
    for (a, b), (_, token) in maps[0][2].items():
        while hit < len(merged) and merged[hit][1] <= a:
            hit += 1
        overlap = hit < len(merged) and merged[hit][0] < b
        same_side = b <= min(edits) or a > max(edits)
        gap = edits[0] - b if b <= edits[0] else a - edits[0] - 1
        if overlap or not same_side or gap > radius:
            rejected['motif_or_geometry'] += 1
            continue
        matches = [m[2].get((a, b)) for m in maps]
        if any(m is None or m[1] != token for m in matches):
            rejected['span_or_identity'] += 1
            continue
        if len({sequence[a:b] for sequence in sequences}) != 1:
            rejected['target_content'] += 1
            continue
        indices = [m[0] for m in matches]
        distinct = True
        for (r, s), count in differences.items():
            if count is not None and indices[r] == indices[s]:
                removed = int(maps[r][1][indices[r]] != maps[s][1][indices[s]])
                distinct &= count > removed
        if not distinct:
            rejected['erased_contrast'] += 1
            continue
        queries.append(dict(span=[a, b], indices=indices, token_id=token, width=b-a))
    return dict(scope='Declared nucleotide prediction-task correspondence only',
                queries=queries, accepted_queries=len(queries),
                reference_real_tokens=len(maps[0][2]), rejected=rejected,
                token_counts=[len(m[1]) for m in maps],
                full_boundaries_equal=maps[0][0] == maps[1][0] == maps[2][0],
                shifted_query_indices=any(len(set(q['indices'])) > 1 for q in queries),
                candidate_accounting_complete=len(queries)+sum(rejected.values()) == len(maps[0][2]))
