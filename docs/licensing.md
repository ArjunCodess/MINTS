# Licensing scope

MINTS original software is MIT, under the repository's `LICENSE`. The author's original manuscript text, figures, explanatory documentation and numerical analysis outputs are CC-BY-4.0 under `LICENSE-OUTPUTS`. This permits commercial reuse with attribution; modified scientific outputs must identify changes. It does not impose a share-alike requirement. The [CC-BY-4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode) governs that grant.

Neither license covers third-party resources. In particular, a license for upstream code does not establish a license for model weights or all datasets used with that code.

| Resource | Scope and distribution boundary |
| --- | --- |
| DNABERT-2 upstream code | Apache-2.0; checkpoint terms must be checked separately. Weights are not redistributed. |
| Nucleotide Transformer v2 100M model | CC-BY-NC-SA-4.0 upstream model terms; excluded comparison does not become MIT. Weights are not redistributed. |
| Pinned downstream benchmark dataset | Explicit license was absent from the inspected pinned metadata. Source-window excerpt permissions remain unresolved; no blanket CC-BY grant is made. |
| ADASTRA 6.1 | Upstream CC-BY-4.0, with attribution retained. |
| JASPAR | Upstream CC-BY-4.0; retain source attribution and version. |
| ENCODE and UCSC genome resources | Their own source-specific terms apply. Full genomes and biological reads are not redistributed. |
| OUP authoring template | Upstream LPPL terms; private fallback template is not original MINTS code. |

The local reviewer archive contains mixed-license material and preserved source receipts. Before depositing it, resolve the dataset excerpt permissions and checkpoint-related output uncertainty documented in the local resource review. Describe the license scopes explicitly in deposit metadata; selecting CC-BY-4.0 for original outputs cannot relicense the whole archive.

## Verified resource references

The [DNABERT-2 code license](https://github.com/MAGICS-LAB/DNABERT_2/blob/main/LICENSE) is separate from the [pinned checkpoint card](https://huggingface.co/zhihan1996/DNABERT-2-117M/blob/7bce263b15377fc15361f52cfab88f8b586abda0/README.md). The inspected card does not establish checkpoint distribution terms. The [pinned downstream dataset](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/tree/851f9946252e90c665cdb3cc3eedb78f1f26197c) likewise lacked an explicit grant in inspected metadata. Public accessibility is not a license. Resolve analyzed-window and related output distribution permissions before the mixed-resource deposit.

[ADASTRA 6.1](https://zenodo.org/records/14174114) and the [JASPAR FAQ](https://jaspar.elixir.no/faq/) provide CC-BY-4.0 terms. Preserve dataset release, matrix version and source attribution. [ENCODE terms](https://www.encodeproject.org/help/rest-api/) permit downloading, analyzing and publishing results for the processed resources used here. [UCSC guidance](https://genome.ucsc.edu/FAQ/FAQlicense.html) requires checking assembly-specific notices: hg19 permits public use and hg38 retains a third-party rights caveat. These permissions do not certify biological labels or measurement quality.

No third-party weights, full genomes or raw biological reads are included in the new publication package. Original manuscript and supplementary text are covered by LICENSE-OUTPUTS; third-party reference matrices, source tables and sequence excerpts retain their own terms. Frozen local receipts record the inspected terms and unresolved questions. The own-work licenses are complete, but an unrestricted license for every upstream resource has not been established.

The [public versioned reproduction dataset](https://doi.org/10.5281/zenodo.23267982) contains original computed observations and all 61 unchanged numeric logit files under CC-BY-4.0. It excludes upstream input DNA windows, dataset distributions, checkpoint weights and tokenizer files. The separately preserved mixed-resource reviewer archive remains private. This scoped deposit does not establish new permissions for upstream resources or independently reconstruct excluded sequence inputs.

The unchanged OUP authoring bundle in paper/venues/vendor/oup-authoring-template and its identical root class/style copies retain LPPL 1.3-or-later and the individual bibliography-file redistribution notices. These are third-party layout sources, not MIT-licensed original MINTS code. Venue manuscript presentations and original abstracts retain the original scientific-output CC-BY-4.0 scope. No model weights, upstream DNA inputs or mixed-resource reviewer archive are added to those venue deliverables.
