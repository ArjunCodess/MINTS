# Verifying this PR

Run `python tools/audit_variant_study.py` from a checkout with the baseline Git
history available. It reads committed evidence and needs no network, model
checkpoint, tokenizer cache or large genomic inputs. Both CI platforms run it
after fetching Git history. A missing local baseline tag falls back to the
recorded merged commit; a present but moved tag fails verification.

The audit checks scientific source and artifact hashes, all historical attempt
snapshots, protocol agreement, complete population accounting, diagnostic
candidate totals, biological QC, score membership, the frozen gate decision,
head-search prohibition after stopping, and confirmation readiness. It also
checks inventory counts against the retrieved CTCF metadata and binds the
inventory, inspection ledger, report generator and auditor into the report
manifest. These checks detect inconsistent claims even if an artifact's stored
hash is updated. They do not independently establish biological truth from a
publication or authenticate a repository an attacker can rewrite in full.

## Diagnostic interpretation

Seven substitutions change reference/alternate nucleotide boundaries. Six of
those cases still have equal token counts, which explains why checking tensor
shape alone would miss the alignment failure. The saved diagnostic spans show
exactly which token covers the substituted nucleotide in each allele.

The other four singleton loci yield 512 candidate positions. Their first
rejections are 500 trinucleotide mismatches, five edit-token width mismatches,
three positions inside reference motifs, three geometry mismatches and one GC
mismatch. Because checks run in a fixed order, these are sequential exclusions;
they do not establish that changing one rule would produce a valid control.
The report and CSVs derive these counts from saved sequence-only traces.

Numeric guards reject missing/nonfinite statistics, truthy nonboolean control
flags, invalid genomic coordinates, impossible divergence/correlation bounds,
inconsistent intervals, invalid head indices and malformed vocabulary vectors.
Exclusive protocol-file creation prevents concurrent runners from claiming the
same output directory. Earlier runs are retained with source snapshots whenever
implementation validation changes; frozen thresholds and biological membership
stay unchanged.

## Fresh-output reproduction

```powershell
python tools/prepare_variant_study.py --output results/variant_study_new
python tools/run_variant_pilot.py --output results/variant_pilot_new --device cuda
python tools/build_variant_artifacts.py --study results/variant_study_new --pilot results/variant_pilot_new --report docs/variant_result_new.md
python tools/audit_variant_study.py --study results/variant_study_new --pilot results/variant_pilot_new
```

Only metadata retrieval and the scientific runner need network or cached
scientific dependencies. The pilot retains its fixed published input. It does
not import another cohort or enable confirmation through command-line flags.
The report builder supports the stopped empty population recorded by this
protocol and rejects a nonempty result rather than generating misleading text.
Report outputs must be Markdown files outside both evidence directories and
cannot overwrite the frozen protocol or baseline files. Rejected targets and
unsupported nonempty results leave existing output bytes unchanged. Generated
commands use the supplied paths, quote PowerShell arguments, and link back to
the documentation from the report's location. Diagnostic CSVs use LF line
endings so rebuilding on Windows and Linux produces the same table bytes.

The remaining work requires a new scientific design or additional biological
evidence. This PR does not relax controls, score alternative endpoints, select
a head, download confirmation outcomes or estimate power from unpatched effects.
