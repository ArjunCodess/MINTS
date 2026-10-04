"""Rerun test-only TATA probe patching in a separate result directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from src.config import DEFAULT_CONFIG
from src.cross_model import cross_model_paths
from src.modeling import load_hooked_encoder
from src.patching import run_batch_dnabert_activation_patching
from src.patching import summarize_sequence_patching
import argparse

from src.utils import write_json, sha256_file

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-run",help="Derive sequence-cluster intervals from saved TATA and donor effects, without model inference")
    args=parser.parse_args()
    if args.summarize_run:
        base=Path(args.summarize_run)/"results"
        for task in ["promoter_tata","splice_sites_donors"]:
            stem=f"{task}_batch_dnabert_activation_patching"
            print(summarize_sequence_patching(base/"patching"/(stem+".csv"),
                base/"patching"/(stem+"_pair_effects.npz"),
                base/"counterfactuals"/(task+"_batch_activation_patching_pairs.tsv")),flush=True)
        return
    config = replace(DEFAULT_CONFIG, paths=cross_model_paths(DEFAULT_CONFIG, "review_tata_heldout"))
    config = replace(config, paths=replace(config.paths, activations_dir=DEFAULT_CONFIG.paths.activations_dir))
    bundle = load_hooked_encoder(config.model)
    write_json(config.paths.results_dir / "review_model_manifest.json", dict(
        model=bundle.model_name, revision=config.model.revision, device=bundle.device,
        tokenizer_class=type(bundle.tokenizer).__name__, pooling="attention-mask mean including special tokens",
        scalar="trained standardized logistic probe decision function", seed=config.data.seed,
        training_cache_sha256=sha256_file(DEFAULT_CONFIG.paths.activations_dir / "promoter_tata_train_residual_mean.npz"),
        historical_checkpoint_identity="unpinned; current snapshot does not establish historical identity"))
    run_batch_dnabert_activation_patching(bundle, "promoter_tata", config=config)


if __name__=="__main__":
    main()
