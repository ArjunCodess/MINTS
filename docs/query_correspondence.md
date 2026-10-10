# Nucleotide query correspondence

`src/query_correspondence.py` certifies a declared reference/alternate/sham prediction task without loading a model. Run `python tools/check_query_correspondence.py` for a hand-constructed example with token counts 4/5/5 and query indices 2/3/3. These maps test the contract; they are not a tokenizer or learned-model calibration.

The caller supplies three equal-length canonical nucleotide strings, token IDs and half-open offsets from one pinned tokenizer and vocabulary, two distinct edit positions, motif intervals, a radius and a mask ID. Real spans must form complete ordered partitions; zero-width special rows are excluded from candidates. The checker rejects malformed partitions and pre-existing mask IDs.

A query is accepted exactly when its positive-width reference interval has the same vocabulary ID and nucleotide content in all inputs, is outside the motif union, lies on the same side of both edits, satisfies the radius from the first edit, and leaves three distinct ID arrays after masking. Row indices and token counts can differ. Rejections are assigned to the first failed predicate group, so their counts are disjoint and cover every rejected reference token.

## Why the certificate is correct

Each positive-width interval occurs once in a valid partition, making every accepted row map unique. Sorting and merging motif intervals preserves their union; the ordered reference sweep detects exactly its overlapping candidates. Tuple lookup, content comparison and geometry checks directly enforce the remaining predicates.

The optimized masking check is equivalent to literal array comparison. Arrays with unequal lengths stay distinct. When indices differ, the mask in one array meets a non-mask in the other because unmasked inputs contain no mask ID. When indices agree, only an ID difference at that index can disappear. Counting pairwise differences once therefore gives exactly the literal masked-triplet inequality test. Every satisfying reference interval reaches acceptance, establishing completeness as well as soundness for the declared inputs.

For three inputs the expected time after tokenization is `O(T0 + T1 + T2 + L + H log H)`, with nucleotide length `L` and `H` motif intervals. Accepted reference spans are disjoint, bounding total content comparisons by `L` per input. Memory is `O(T0 + T1 + T2 + H + Q)`, including `Q` returned records. Hash lookup has the usual expected-time assumption. These are routine correctness and complexity guarantees, not a claim of algorithmic priority.

## Saved-evidence check and limits

`results/correspondence_audit/query_certificate.json` records a post-study diagnostic of all 61 saved cases in 60 clusters. The checker reproduces all 438 saved queries, including 29 cases with shifted query indices and 15 with complete boundary stability. Its input, generator and tokenizer hashes establish lineage. No candidate search, model prediction, membership, endpoint or gate changed. Retokenizing with `--saved-cases --output NEW_PATH.json` requires the cached pinned tokenizer and JASPAR matrix; the default example and stored-certificate checks do not.

The certificate does not verify tokenizer provenance, scientific adequacy of controls, model wiring, native endpoint sensitivity or biological binding. Segmentation and position elsewhere can still change model context. Full final-row transfer restores donor logits only under a loaded pointwise prediction head and compatible interfaces; a single head-channel transfer has no such guarantee. The mapped native sensitivity gate remains failed, and native head search remains stopped.

Tests compare the optimized checker with a literal masking oracle on 600 deterministic adversarial triplets, and cover shifted indices, half-open boundaries, radius zero, unchanged IDs with changed content, erased contrasts and invalid partitions. These implementation checks do not measure natural-data statistical power.
