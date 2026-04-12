# MINTS Mechanistic Interpretability Pipeline Plan

## Summary
Build a reproducible Python pipeline under `src/` with `main.py` as a single-command entrypoint. All downloaded/intermediate inputs go under `data/`; all metrics, caches, tables, plots, and manifests go under `results/`.

Use `zhihan1996/DNABERT-2-117M` as the primary model and target `transformer_lens.HookedEncoder`, not `HookedTransformer`, because the selected model is BERT-style and TransformerLens documents `HookedTransformer` mainly for autoregressive models. Add a compatibility gate: fail fast with a clear error if the current TransformerLens version cannot load DNABERT-2’s custom `BertForMaskedLM`, and point to the custom hook fallback as future work.

## Infrastructure, Model Wrapping, Data Ingestion
- Replace the current empty dependency setup with a Python 3.11/3.12 compatible environment. The existing `.venv` is Python 3.14 and has none of the required ML packages installed.
- Add dependencies for `torch`, `transformers`, `datasets`, `transformer-lens`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `requests`, `biopython`, and a genome interval reader such as `pyfaidx`.
- Implement modular modules:
  - `src/config.py`: paths, model name, task names, batch sizes, device, seeds.
  - `src/modeling.py`: tokenizer/model loading and `HookedEncoder` wrapping for `zhihan1996/DNABERT-2-117M`.
  - `src/data_ingestion.py`: Hugging Face dataset download/filter/tokenization and ENCODE downloads.
  - `src/ctcf.py`: GRCh38 sequence extraction from ENCODE peak intervals.
  - `src/cli.py`: CLI commands wired through `main.py`.
- For Hugging Face downstream tasks, load `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised` and filter the `task` column for:
  - `promoter_tata`
  - `promoter_no_tata`
  - `splice_sites_donor`
  - `splice_sites_acceptor`
- Accept the user’s plural folder names as aliases: `splice_sites_donors -> splice_sites_donor`, `splice_sites_acceptors -> splice_sites_acceptor`.
- Save tokenized task datasets under `data/hf_downstream/<task>/` and write manifests under `results/manifests/`.
- For ENCODE, read URLs from `data/ENCODE4_v1.5.1_GRCh38.txt`, download only artifact URLs ending in `.bigWig`, `.bed.gz`, or `.bigBed`, and skip the metadata URL. Store files under `data/encode/ctcf_gm12878/`.
- Add GRCh38 FASTA/twoBit acquisition as part of CTCF setup so CTCF peak/background intervals can be converted to model-ready DNA sequences. Save derived CTCF sequence tables under `data/ctcf/`.

## Circuit Extraction and Residual Probing
- Implement batched activation caching in `src/activations.py` using `model.run_with_cache()` on tokenized DNA sequences. Save compact cache-derived arrays, not full unbounded caches, under `results/activations/`.
- Cache and export at minimum:
  - residual stream vectors per selected layer and position pooling strategy,
  - attention patterns per layer/head,
  - head outputs where supported by the wrapped encoder.
- Implement `src/circuits.py` to extract `W_Q`, `W_K`, `W_V`, `W_O` for selected layers/heads and compute:
  - `QK = W_Q @ W_K.T`
  - `OV = W_V @ W_O`
- Save circuit matrices and metadata under `results/circuits/`.
- Implement `src/probing.py`:
  - freeze residual vectors,
  - train logistic regression probes for `promoter_tata`, `promoter_no_tata`, splice donor/acceptor, and CTCF binding examples,
  - use stratified train/test splits or provided dataset splits,
  - report AUROC, AUPRC, accuracy, class balance, and confidence intervals where practical.
- Implement attention enrichment in `src/enrichment.py` by mapping motif support indices from character coordinates to token positions, summing attention mass over motif-support positions, and normalizing by expected attention mass over matched non-motif positions. Save per-layer/head enrichment tables under `results/enrichment/`.

## Causal Intervention and Activation Patching
- Implement `src/counterfactuals.py` for paired clean/corrupted sequence generation:
  - TATA: mutate `TATAAA`/TATA-like hits to a GC-balanced non-motif sequence.
  - Splice donor/acceptor: mutate canonical splice dinucleotides or annotated support windows.
  - CTCF: mutate JASPAR CTCF motif hits using the existing JASPAR motif file in `data/`.
- Preserve sequence length and record exact mutation coordinates in `results/counterfactuals/`.
- Implement `src/patching.py` with TransformerLens patching utilities where compatible, targeting attention head outputs by layer/head/position.
- Use the restoration metric:
  - `(patched_logit - corrupted_logit) / (clean_logit - corrupted_logit)`
- Define the biological target logit consistently per task:
  - binary classifier head/logit if using a task-specific probe,
  - otherwise probe score on patched residuals as the primary v1 metric.
- Save layer-by-head restoration matrices to `results/patching/` and heatmaps to `results/figures/`.

## Strict Mechanistic Proof Extensions
- Implement `src/motif_scoring.py` for ground-truth motif scoring:
  - load the JASPAR CORE vertebrate CTCF PWM with matrix ID `MA0139.1` using Biopython,
  - prefer a local `data/**/MA0139.1.jaspar` file and fall back to the JASPAR 2024 API for the small matrix file,
  - scan ENCODE GM12878 CTCF peak sequences derived from ENCSR000DKV/ENCODE artifacts,
  - compute scalar motif scores per nucleotide and per token position,
  - record motif-support token intervals for later attention and patching tests.
- Implement `src/qk_alignment.py`:
  - capture layer-input hidden states for selected DNABERT-2 layers,
  - compute per-position QK attention logits from exported/direct `QK = W_Q @ W_K.T` matrices,
  - correlate QK-derived key scores with ground-truth CTCF motif scores using Pearson correlation,
  - flag candidate heads satisfying the pre-registered criterion `r >= 0.5` and `p < 0.05`,
  - save alignment tables and heatmaps under `results/qk_alignment/` and `results/figures/`.
- Extend attention enrichment:
  - isolate attention mass assigned to motif-support key positions, `a_motif`,
  - compare it with matched non-motif background mass, `a_bg`,
  - flag heads with enrichment ratio `rho_h >= 2.0`,
  - save enrichment tables and heatmaps under `results/enrichment/` and `results/figures/`.
- Implement custom DNABERT-2 PyTorch hook patching:
  - cache clean head outputs from `encoder.layer[i].attention.self`,
  - rerun corrupted counterfactual inputs while replacing only the requested layer/head slice,
  - use a probe score on patched residuals as the scalar target logit when no task-specific classifier head exists,
  - compute the normalized patching metric `(patched_logit - corrupted_logit) / (clean_logit - corrupted_logit)`,
  - save layer-by-head restoration matrices and heatmaps.
- Wire the strict proof exports into the existing one-command pipeline after residual caches, probe metrics, and QK/OV matrices exist.
- Add a CLI cap for CTCF QK-alignment sequence count so quick runs can use a subset while the default scans all prepared CTCF sequences.
- Treat a head as a strict candidate motif detector only when three conditions align:
  - QK-to-motif Pearson criterion passes,
  - attention enrichment criterion passes,
  - causal restoration criterion passes.

## Test Plan
- Unit-test URL filtering, task alias normalization, motif mutation length preservation, coordinate-to-token mapping, QK/OV matrix shape checks, and restoration metric edge cases where `clean_logit == corrupted_logit`.
- Unit-test JASPAR PWM loading/scoring, token-level motif support extraction, Pearson candidate filtering, matched-background enrichment, and head-output hook replacement.
- Add smoke tests that run on 4-8 sequences per task and produce small result files in `results/test_runs/`.
- Add integration tests for:
  - HF dataset filtering returns non-empty rows for all four canonical task names.
  - ENCODE URL reader downloads/skips the expected URL classes in dry-run mode.
  - model wrapper exposes hook names needed for residual stream, attention pattern, and head output extraction.
- Add reproducibility checks: fixed seed, saved config snapshot, model/tokenizer revision metadata, and result manifests.

## Assumptions and Defaults
- Default model: `zhihan1996/DNABERT-2-117M`; default wrapper: `transformer_lens.HookedEncoder`.
- Do not target `HookedTransformer.from_pretrained()` for v1 because the requested models are encoder/masked-LM style.
- Default CTCF path includes GRCh38 sequence extraction, not artifact download only.
- The HF dataset’s current public view exposes a `default` subset with a `task` column; the plan treats task-column filtering as the canonical path and folder names as aliases.
- Primary references checked: TransformerLens `HookedTransformer.from_pretrained` docs, Hugging Face model cards for DNABERT-2 and Nucleotide Transformer, and the Hugging Face dataset card for `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`.
