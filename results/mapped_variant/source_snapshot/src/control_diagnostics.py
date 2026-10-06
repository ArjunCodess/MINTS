"""Sequence-only constraint intersections, separate from frozen eligibility."""
from collections import Counter

from .assay_alignment import exact_offsets
from .native_followup import target_geometry


def token_map(tokenizer, sequence):
    offsets = exact_offsets(tokenizer, sequence)
    ids = list(tokenizer(sequence, add_special_tokens=True, truncation=False)["input_ids"])
    return offsets, ids, {(a, b): (i, ids[i]) for i, (a, b) in enumerate(offsets) if b > a}


def common_queries(maps, index, hits, radius=32):
    """Map unchanged identical nucleotide spans and IDs, never tensor indices."""
    result = []
    for span, (i, token) in maps[0][2].items():
        if (target_geometry((index, index + 1), span)[0] == "overlap"
                or target_geometry((index, index + 1), span)[1] > radius
                or any(span[0] < b and span[1] > a for a, b, _ in hits)):
            continue
        if all(span in m[2] and m[2][span][1] == token for m in maps[1:]):
            result.append(dict(span=list(span), indices=[m[2][span][0] for m in maps],
                               token_id=token, width=span[1] - span[0]))
    return result


def same_side_queries(queries, index, sham_index):
    return [q for q in queries if q["span"][1] <= min(index, sham_index)
            or q["span"][0] > max(index, sham_index)]


def diagnose_window(tokenizer, window, motif, protocol):
    clean, alt, index = (window[k] for k in ("reference_sequence", "alternate_sequence", "variant_index"))
    maps = [token_map(tokenizer, s) for s in (clean, alt)]
    hits = motif.hits(clean)
    covering = sorted([h for h in hits if h[0] <= index < h[1]], key=lambda h: (-h[2], h[0]))
    queries = common_queries(maps, index, hits, protocol.query_radius_bp)
    full = maps[0][0] == maps[1][0]
    edit_width = next(b-a for a,b in maps[0][0] if a <= index < b)
    def gc(j):
        s = clean[max(0,j-16):j+17]
        return (s.count("G")+s.count("C"))/len(s)
    summary = dict(full_allele_offsets=full, reference_motif=bool(covering),
                   query_correspondence=bool(queries), reference_tokens=len(maps[0][1]),
                   alternate_tokens=len(maps[1][1]), candidate_positions=0,
                   substitution_candidates=0, trinucleotide_candidates=0)
    candidates = []
    # Every position is counted; expensive independent predicates are evaluated
    # on the explicit exact-substitution/trinucleotide risk set, even without motif.
    for j in range(1, len(clean)-1):
        if j == index or abs(j-index) > protocol.sham_radius_bp:
            continue
        summary["candidate_positions"] += 1
        if clean[j] != clean[index]:
            continue
        summary["substitution_candidates"] += 1
        if clean[j-1:j+2] != clean[index-1:index+2]:
            continue
        summary["trinucleotide_candidates"] += 1
        sham = clean[:j] + alt[index] + clean[j+1:]
        smap = token_map(tokenizer, sham)
        chosen = same_side_queries(queries, index, j)
        mapped = common_queries([*maps, smap], index, hits, protocol.query_radius_bp)
        mapped = same_side_queries(mapped, index, j)
        motif_locations = {(a,b) for a,b,_ in motif.hits(sham)} == {(a,b) for a,b,_ in hits}
        score_preserved = (not covering or abs(motif.span_score(sham,*covering[0][:2])-covering[0][2])
                           <= protocol.sham_pwm_tolerance_bits)
        predicates = dict(outside_motif=not any(a <= j < b for a,b,_ in hits),
            edit_width=next(b-a for a,b in maps[0][0] if a <= j < b) == edit_width,
            query_side=bool(chosen), geometry16=bool(chosen) and abs(j-index) <= 16,
            geometry32=bool(chosen) and abs(j-index) <= 32,
            geometry64=bool(chosen) and abs(j-index) <= 64,
            gc=abs(gc(index)-gc(j)) <= protocol.gc_tolerance,
            motif_locations=motif_locations, motif_score=score_preserved,
            full_sham_offsets=smap[0] == maps[0][0], query_identity=bool(mapped))
        base = bool(covering) and bool(queries) and all(predicates[k] for k in
            ("outside_motif", "query_side", "gc", "motif_locations", "motif_score", "query_identity"))
        predicates.update(original=base and full and predicates["edit_width"]
                          and predicates["geometry16"] and predicates["full_sham_offsets"],
            local_only=base and predicates["edit_width"] and predicates["geometry16"],
            local_no_width=base and predicates["geometry16"],
            local_geometry32=base and predicates["geometry32"],
            local_geometry64=base and predicates["geometry64"])
        candidates.append(dict(sham_index=j, distance_bp=abs(j-index),
            gc_difference=abs(gc(index)-gc(j)), common_query_count=len(mapped), **predicates))
    for key in ("original", "local_only", "local_no_width", "local_geometry32", "local_geometry64"):
        summary[key] = any(r[key] for r in candidates)
    return summary, candidates
