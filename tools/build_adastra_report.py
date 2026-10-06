"""Build the downloaded-cohort report from verified engineering evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json

from src.utils import sha256_file,write_json
from tools.audit_adastra_feasibility import audit

ROOT=Path(__file__).resolve().parents[1]


def build():
    output=ROOT/"results/adastra_exploratory"
    manifest=output/"report_manifest.json"
    backup=output/".report_manifest.building.json"
    if backup.exists():raise FileExistsError("Another report build or interrupted build needs review")
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        for key,path in {"auditor_sha256":ROOT/"tools/audit_adastra_feasibility.py",
                         "execution_sha256":output/"execution.json",
                         "report_sha256":ROOT/previous["report_path"]}.items():
            if previous[key]!=sha256_file(path):raise ValueError("Existing report evidence changed: "+key)
        # Presentation code may change; audit scientific inputs without its prior receipt.
        manifest.rename(backup)
    try:audit(output)
    finally:
        if backup.exists():backup.rename(manifest)
    summary=json.loads((output/"summary.json").read_text())
    lines="\n".join(f"| {reason} | {count} |" for reason,count in sorted(summary["reasons"].items()))
    text=f"""# Downloaded ADASTRA CTCF feasibility

The pinned [Mabel v6.1 release](https://zenodo.org/records/14174114) was downloaded
and verified against the publisher's byte count and MD5, with a SHA-256 receipt.
Its complete CTCF table contains {summary['source_records']:,} coverage-eligible
variant records. Nonsignificant records were kept in the candidate denominator.
The deposit is CC BY 4.0, attributed to Nachatoy, Abramov, Boytsov and Kulakovskiy.

A protocol frozen before table inspection selected {summary['screened']:,}
variants by SHA-256 of seed 1731 and allele-specific genomic coordinates.
Binding effects, significance and motif-concordance labels did not select cases.
Each 204-base hg38 window was checked against the reference allele, then against
the original sequence alignment, motif and substitution-matched control rules.

| First exclusion or acceptance | Records |
| --- | ---: |
{lines}

Reference alleles were verified in {summary['reference_verified']:,} selected
windows; {summary['sequence_controls']} single-SNV reference scenarios have a
feasible aligned sequence control. These are engineering candidates, not verified
individual haplotypes, independent donors or confirmation-eligible biological loci.
The hash sample describes this bounded screen, not the full-cohort retention rate.

No native model inference, head search, binding correlation or power calculation
was run. ADASTRA allele-wise effects are weighted log2 observed/expected ratios;
they cannot replace GSE81945's pooled input-normalized log odds. Aggregate dosage
correction does not establish individual donor, phase, replicate or mapping-bias
QC. Those checks, source overlap, homology and intervention variance remain missing.
The original GSE81945 zero-case pilot and its frozen decision remain unchanged.

The complete source table is now marked exploratory in its own inspection ledger.
It cannot be claimed as a fresh confirmation cohort. The earlier metadata-only
ledger remains a historical record of what had been inspected at that time.

## Reproduction

```powershell
python tools/download_adastra.py
python tools/run_adastra_feasibility.py --output results/adastra_exploratory_new
python tools/audit_adastra_feasibility.py
python tools/build_adastra_report.py
```

The downloader caches the 942,669,336-byte archive under ignored `data/adastra/`.
Scientific attempts require fresh output directories; the runner freezes its
protocol exclusively before source inspection and retains failed receipts.
The offline auditor verifies the committed full coordinate denominator, hash
membership, source/output receipts, candidate accounting and disabled inference.
It needs no archive, reference genome, tokenizer, network or model weights.
The last two commands verify and build the saved default result; pass the new
output directory to the auditor when verifying another attempt.
"""
    report=ROOT/"docs/adastra_feasibility.md";report.write_bytes(text.encode())
    write_json(output/"report_manifest.json",dict(generator_sha256=sha256_file(Path(__file__)),
        auditor_sha256=sha256_file(ROOT/"tools/audit_adastra_feasibility.py"),execution_sha256=sha256_file(output/"execution.json"),
        report_sha256=sha256_file(report),report_path="docs/adastra_feasibility.md"))
    print("verified downloaded cohort and built engineering report")


if __name__=="__main__":build()
