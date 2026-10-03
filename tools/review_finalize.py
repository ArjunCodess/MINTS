"""Hash final outputs and assemble the anonymous source package."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
out=root/"results/review"
from huggingface_hub.constants import HF_HUB_CACHE
revision="7bce263b15377fc15361f52cfab88f8b586abda0"
snapshot=Path(HF_HUB_CACHE)/"models--zhihan1996--DNABERT-2-117M"/"snapshots"/revision
def digest(path):
    hasher=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()
if not snapshot.exists():
    raise FileNotFoundError(f"Pinned model snapshot unavailable: {snapshot}")
(out/"model_snapshot_manifest.json").write_text(json.dumps(dict(model="zhihan1996/DNABERT-2-117M",revision=revision,
    local_snapshot=str(snapshot),sha256={p.name:digest(p) for p in sorted(snapshot.iterdir()) if p.is_file()},
    scope="Pinned local assets for revision experiments; does not establish historical cache checkpoint identity"),indent=2),encoding="utf-8")
mapping={
    "Table 2":["correctness_audit.json","dataset_table.tex"],
    "Table 3":["classification_metrics.csv","table3.tex"],
    "Table 4":["classification_metrics.csv","gc_matching_diagnostics.json","table4.tex"],
    "Table 5":["ctcf_threshold_sweep.csv","threshold_table.tex"],
    "Table 6":["heldout_patching_summary.json","patching_table.tex"],
    "Figure 1":["ctcf_native_control_inference.csv","ctcf_native_control_scores.npz","ctcf_genomic_control_pairs.csv","ctcf_native_controls.png"],
    "Figure 2":["heldout_pair_plot_data.csv","heldout_patching.png"],
}
package=[root/"paper/main.tex",root/"paper/references.bib",root/"paper/neurips_2026.sty",root/"paper/main.pdf",
         *out.glob("*.tex"),out/"ctcf_native_controls.png",out/"heldout_patching.png"]
with zipfile.ZipFile(out/"mints_revision_source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
    for path in package:
        archive.write(path,path.relative_to(root).as_posix())
files=[p for p in out.iterdir() if p.is_file() and p.name not in {"final_artifact_manifest.json","commands.jsonl"} and not p.name.startswith(("page-","command_"))]
files.extend(package)
files.extend(p for p in (root/"results/cross_model/review_tata_heldout").rglob("*") if p.is_file())
files.extend(p for p in (root/"tools").glob("*.py"))
files.extend(p for p in (root/"src").glob("*.py"))
files.extend(p for p in (root/"tests").glob("*.py"))
files.extend([root/"requirements.txt",root/"main.py"])
manifest=dict(submission_commit="7112ec22b770c1651f24d33ac5f45fda3e986324",branch="xai4science-review-improvements",
              table_figure_sources=mapping,editorial_tables={"Table 1":"evidence definitions","Table 7":"docs/SOURCE_VERIFICATION.md"},
              sha256={str(p.relative_to(root).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},
              anonymous_package=[p.relative_to(root).as_posix() for p in package],
              package_scope="Manuscript sources, generated inserts, figures and PDF only; excludes author-identifying internal review docs and historical results.")
(out/"final_artifact_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(f"Hashed {len(manifest['sha256'])} files; packaged {len(package)} files")
