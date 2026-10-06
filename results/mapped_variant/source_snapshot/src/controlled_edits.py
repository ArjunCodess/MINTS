"""Composition-preserving motif edits and transition-matched sham edits."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re

import numpy as np

from .assay_alignment import validate_patch_alignment
from .counterfactuals import MutationRecord, TATA_RE
from .motif_scoring import default_support_threshold, scan_sequence_with_pssm


@dataclass(frozen=True)
class MotifDefinition:
    name: str
    pattern: str | None = None
    pssm: object | None = None
    threshold: float | None = None
    fraction: float = .8
    scan_reverse: bool = True
    prefer_center: bool = False

    def hits(self, sequence):
        if self.pssm is not None:
            scores = scan_sequence_with_pssm(sequence, self.pssm)
            threshold = self.threshold if self.threshold is not None else default_support_threshold(self.pssm, self.fraction)
            return [(int(i), int(i + self.pssm.length), float(scores[i]))
                    for i in np.flatnonzero(np.isfinite(scores) & (scores >= threshold))]
        pattern = re.compile(self.pattern or TATA_RE.pattern, re.IGNORECASE)
        reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        forward_hits = [(m.start(), m.end(), 1.) for m in pattern.finditer(sequence)]
        reverse_hits = [(len(sequence) - m.end(), len(sequence) - m.start(), 1.) for m in pattern.finditer(reverse)]
        return sorted(set(forward_hits + (reverse_hits if self.scan_reverse else [])))

    def span_score(self, sequence, start, end):
        if self.pssm is not None:
            return float(scan_sequence_with_pssm(sequence, self.pssm)[start])
        return float(any(a == start and b == end for a, b, _ in self.hits(sequence)))


def edit_signature(original, changed):
    if len(original) != len(changed):
        raise ValueError("Edit signatures require equal nucleotide lengths")
    return Counter((a, b) for a, b in zip(original, changed) if a != b)


def _record(sequence, changed, span, identifier, motif, strategy):
    start, end = span
    return MutationRecord(identifier, motif.name, sequence, changed, motif.name, start, end,
                          sequence[start:end], changed[start:end], strategy)


def controlled_edit_pairs(sequence, identifier, motif, tokenizer, max_candidates=128, max_pairs=3, seed=1729):
    """Find boundary-compatible motif/sham pairs without consulting model scores.

    Both edits preserve exact mononucleotide counts. A sham has the same base
    transition multiset, edit count, and window width as its motif edit. It is
    the nearest eligible non-motif window, with its distance logged. Exact
    matching can retain a small population; exclusions are explicitly counted.
    """
    sequence = sequence.upper()
    if not sequence or set(sequence) - set("ACGT") or max_candidates < 1 or max_pairs < 1:
        raise ValueError("Expected canonical DNA and positive candidate/pair limits")
    rng = np.random.default_rng(seed)
    hits = motif.hits(sequence)
    audit = dict(sequence_id=str(identifier), candidate_limit=max_candidates, motif_hits=len(hits),
                 motif_candidates=0, alignment_rejections=0, new_motif_rejections=0,
                 sham_rejections=0, retained_pairs=0, composition="exact A/C/G/T counts",
                 sham_matching="same base transitions, edit count and nucleotide window width")
    if not hits:
        audit["exclusion"] = "no threshold-passing motif"
        return [], audit
    # Score, then position. This policy is independent of the pretrained output.
    key = (lambda h: (abs((h[0]+h[1])/2-len(sequence)/2), -h[2], h[0])) if motif.prefer_center else (lambda h: (-h[2], h[0]))
    start, end, old_score = sorted(hits, key=key)[0]
    original = sequence[start:end]
    old_locations = {(a, b) for a, b, _ in hits}
    candidates = set()
    swaps = [(i, j) for i in range(len(original)) for j in range(i + 1, len(original)) if original[i] != original[j]]
    rng.shuffle(swaps)
    for i, j in swaps[:max_candidates // 2 + 1]:
        letters = list(original)
        letters[i], letters[j] = letters[j], letters[i]
        candidates.add("".join(letters))
    for _ in range(max_candidates):
        if len(candidates) >= max_candidates:
            break
        candidates.add("".join(rng.permutation(list(original))))
    retained = []
    for replacement in sorted(candidates, key=lambda s: (sum(a != b for a, b in zip(s, original)), s)):
        if replacement == original:
            continue
        changed = sequence[:start] + replacement + sequence[end:]
        audit["motif_candidates"] += 1
        new_hits = motif.hits(changed)
        new_locations = {(a, b) for a, b, _ in new_hits}
        if (start, end) in new_locations or not new_locations.issubset(old_locations):
            audit["new_motif_rejections"] += 1
            continue
        try:
            alignment = validate_patch_alignment(tokenizer, sequence, changed)
        except ValueError:
            audit["alignment_rejections"] += 1
            continue
        signature = edit_signature(original, replacement)
        width = end - start
        windows = sorted(range(len(sequence) - width + 1), key=lambda a: (abs(a - start), a))
        sham = None
        for control_start in windows:
            control_end = control_start + width
            if any(control_start < b and control_end > a for a, b, _ in hits):
                continue
            source = sequence[control_start:control_end]
            letters = list(source)
            used = set()
            possible = True
            for (a, b), count in sorted(signature.items()):
                available = [i for i, base in enumerate(source) if base == a and i not in used]
                if len(available) < count:
                    possible = False
                    break
                for i in available[:count]:
                    letters[i] = b
                    used.add(i)
            if not possible:
                continue
            control_replacement = "".join(letters)
            control = sequence[:control_start] + control_replacement + sequence[control_end:]
            if {(a, b) for a, b, _ in motif.hits(control)} != old_locations:
                continue
            try:
                sham_alignment = validate_patch_alignment(tokenizer, sequence, control)
            except ValueError:
                continue
            offsets = alignment['clean_offsets']
            motif_tokens = [i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]
            sham_tokens = [i for i,(a,b) in enumerate(offsets) if b>a and a<control_end and b>control_start]
            if len(motif_tokens) != len(sham_tokens):
                continue
            motif_width = offsets[motif_tokens[-1]][1] - offsets[motif_tokens[0]][0]
            sham_width = offsets[sham_tokens[-1]][1] - offsets[sham_tokens[0]][0]
            if motif_width != sham_width:
                continue
            sham = (control, (control_start, control_end), sham_alignment)
            break
        if sham is None:
            audit["sham_rejections"] += 1
            continue
        motif_record = _record(sequence, changed, (start, end), str(identifier), motif, "composition_preserving_motif_edit")
        sham_record = _record(sequence, sham[0], sham[1], str(identifier), motif, "transition_matched_nonmotif_edit")
        if Counter(changed) != Counter(sequence) or Counter(sham[0]) != Counter(sequence):
            raise AssertionError("Controlled edit changed nucleotide composition")
        retained.append(dict(motif=asdict(motif_record), sham=asdict(sham_record),
                             motif_alignment=alignment, sham_alignment=sham[2],
                             motif_score_before=old_score, motif_score_after=motif.span_score(changed, start, end),
                             edit_count=sum(signature.values()), sham_distance_bp=abs(sham[1][0] - start),
                             candidate_index=len(retained)))
        if len(retained) >= max_pairs:
            break
    audit["retained_pairs"] = len(retained)
    if not retained:
        audit["exclusion"] = "no boundary-compatible motif and matched sham pair"
    return retained, audit
