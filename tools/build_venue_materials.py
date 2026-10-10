"""Build venue presentations from the scientific master, without submitting."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/venues"
TITLE = "MINTS: Nucleotide correspondence and control feasibility in genomic transformer interventions"
CODE = "https://github.com/ArjunCodess/MINTS/tree/recomb-submission-package"
MIRROR = "https://anonymous.4open.science/r/MINTS/"
DATA = "https://doi.org/10.5281/zenodo.23267982"
COMMUNITY = """Genomic transformers split DNA into variable-length tokens, so equal activation shapes can conceal interventions at different nucleotide positions. MINTS implements explicit nucleotide-span and vocabulary correspondence, checks controlled-edit feasibility, and separates trained-readout effects from native model predictions and biological binding evidence. Six of ten historical TATA patching pairs fail correspondence. A repaired discovery-selected CTCF head changes a fixed trained readout by 0.0157 decision-score units on 64 held-out sequences in 45 clusters, with a 95% block interval of [0.0066, 0.0251]. The TATA estimate remains inconclusive, and the selected CTCF head's measured mean base density is near background. A separately frozen mapped-query study retains 61 natural-variant scenarios in 60 genomic clusters. Variant and sham prediction divergences average 0.013053 and 0.029018 nats; their paired contrast is -0.015966, with a 95% cluster interval of [-0.024454, -0.009146]. Larger sham shifts fail the positive-sensitivity gate and stop native head search. A post-study correspondence checker reproduces all 438 saved queries without new model scoring or changed eligibility. Two limited proofs explain incompatible control constraints. Public code and sequence-free archived evidence reproduce numerical reporting. The contribution is a bounded computational validity audit. Residual segmentation effects, untested cross-model sensitivity and incomplete biological measurement provenance prevent biological binding-causality claims."""
STRUCTURED = r"""\textbf{Motivation:} Variable-length tokenization can invalidate genomic component interventions even when tensors have equal shapes. We test nucleotide correspondence and control feasibility in DNABERT-2.
\textbf{Results:} Six of ten historical TATA pairs fail correspondence. Repaired CTCF patching yields a 0.0157 trained-readout effect on 64 sequences in 45 clusters, with 95\% block interval [0.0066, 0.0251]. TATA remains inconclusive. A separately frozen mapped-query study retains 61 natural-variant cases in 60 clusters; variant-minus-sham Jensen--Shannon divergence is -0.015966 nats, with interval [-0.024454, -0.009146]. Larger sham shifts fail the positive sensitivity gate and stop head search. Explicit query correspondence increases eligibility but leaves segmentation effects unresolved. Biological binding causality is not established.
\textbf{Availability and Implementation:} Earlier code snapshot: \url{https://anonymous.4open.science/r/MINTS/}. Current code: \url{https://github.com/ArjunCodess/MINTS/tree/recomb-submission-package}. Sequence-free reproduction data: \url{https://doi.org/10.5281/zenodo.23267982}.
\textbf{Contact:} arjunv.prakash12345@gmail.com.
\textbf{Supplementary Information:} A separate supplement supplies diagnostics and reproduction."""
ALT = "Scatterplot of 60 genomic cluster means for 61 mapped-query cases. Most variant-minus-sham Jensen--Shannon contrasts lie below zero. The equal-cluster mean is -0.015966 nats, with a 95 percent cluster-bootstrap interval from -0.024454 to -0.009146. Larger sham shifts fail positive differential sensitivity on this selected population; this is not a biological binding effect."
HIGHLIGHTS = [
    "Equal activation shapes can conceal mismatched nucleotide interventions.",
    "Explicit query correspondence reproduces 438 saved queries in 61 cases.",
    "CTCF readout effects do not establish native biological binding causality.",
    "Larger sham shifts fail the frozen natural-variant sensitivity gate.",
]


def plain(text):
    text = re.sub(r"\\(?:textbf|url|path|texttt)\{([^{}]*)\}", r"\1", text)
    return text.replace(r"\%", "%")


def build(compile_pdfs=False):
    OUT.mkdir(exist_ok=True)
    master = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    body = master[master.index(r"\section{Introduction}"):master.index(r"\clearpage\bibliographystyle")]
    body = body.replace(r"\input{generated/", r"\input{../generated/")
    body = body.replace("{figures/", "{../figures/")
    body = body.replace("https://github.com/ArjunCodess/MINTS", CODE)
    abstract = master.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0].strip()
    numbers = (ROOT / "paper/generated/numbers.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", numbers))
    for name, value in macros.items():
        abstract = abstract.replace("\\" + name + "{}", value)
    header = master[:master.index(r"\begin{document}")].replace(r"\input{generated/", r"\input{../generated/")
    cbm = header + "\\begin{document}\\maketitle\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
    cbm += r"\noindent\textbf{Keywords:} genomic transformers; nucleotide correspondence; activation patching; control feasibility." + "\n" + body
    cbm += r"\clearpage\bibliographystyle{plainnat}\bibliography{references}\end{document}" + "\n"
    (OUT / "cbm.tex").write_text(cbm, encoding="utf-8", newline="\n")
    oup_body = body.replace(r"\FloatBarrier", "")
    oup_body = oup_body.replace(r"\begin{table}", r"\begin{table*}").replace(r"\end{table}", r"\end{table*}")
    oup_body = oup_body.replace("GM12878 DNase ENCSR000EMT", r"GM12878 DNase \path{ENCSR000EMT}")
    oup_body = oup_body.replace(r"Q=\{(a,b,v):\ \forall r\in\{0,1,2\},\ \exists i_r\text{ with }" + "\n" + r"s_{i_r}(x^{(r)})=[a,b),\ v_{i_r}(x^{(r)})=v\}.", r"\begin{aligned}Q=\{(a,b,v):\ &\forall r\in\{0,1,2\},\ \exists i_r\text{ with }\\" + "\n" + r"&s_{i_r}(x^{(r)})=[a,b),\ v_{i_r}(x^{(r)})=v\}.\end{aligned}")
    # Use the supplied code mirror without removing the accessible canonical code.
    oup_body = oup_body.replace("Public code, the main paper, supplement and build inputs are at", "Earlier code snapshot mirror: \\url{" + MIRROR + "}. Public code, the main paper, supplement and build inputs are at")
    oup_body = oup_body.replace(r"\end{figure}", r"\figalttext{" + ALT + "}" + "\n" + r"\end{figure}")
    oup_header = r"""\documentclass[unnumsec,webpdf,modern,large,namedate]{oup-authoring-template}
\usepackage{amsmath,amssymb,booktabs,graphicx,xurl}
\input{../generated/numbers}
\newcommand{\societylogo}{}
\begin{document}
\journaltitle{Bioinformatics Advances}\DOI{}\copyrightyear{}\pubyear{}\vol{}\issue{}\access{}\appnotes{}
"""
    oup_header += r"\title[MINTS]{" + TITLE + "}\n"
    oup_header += r"""\author[1,$\ast$]{Arjun Vijay Prakash}
\address[1]{\orgname{Independent Researcher, City Montessori School, Lucknow, India}}
\address[$\ast$]{Corresponding author. \texttt{arjunv.prakash12345@gmail.com}}
"""
    glbio = oup_header + r"\abstract{" + STRUCTURED + "}\n" + r"\keywords{genomic transformers, nucleotide correspondence, activation patching}" + "\n\\maketitle\n" + oup_body
    glbio += r"\bibliographystyle{oup-abbrvnat}\bibliography{references}\end{document}" + "\n"
    (OUT / "glbio.tex").write_text(glbio, encoding="utf-8", newline="\n")
    extended = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,url,xurl}
"""
    extended += r"\title{" + TITLE + "}\n"
    extended += r"""\author{Arjun Vijay Prakash\\Independent Researcher, City Montessori School, Lucknow, India\\\texttt{arjunv.prakash12345@gmail.com}}\date{}
\begin{document}\maketitle
""" + COMMUNITY.replace("%", r"\%") + "\n\n"
    extended += r"The correspondence certificate checks declared span maps and constraints, allowing shifted indices and unequal token counts. It does not establish tokenizer provenance or isolate segmentation elsewhere. Original memberships, endpoints, timestamps and stopped studies remain intact. Numerical reproduction uses saved clusters; fresh inference requires upstream resources under their own terms." + "\n\n"
    extended += r"Code: \url{" + CODE + "}. Reproduction data: \\url{" + DATA + "}.\n\\end{document}\n"
    (OUT / "glbio_abstract.tex").write_text(extended, encoding="utf-8", newline="\n")
    (OUT / "glbio_abstract.txt").write_text(COMMUNITY + "\n", encoding="utf-8", newline="\n")
    (OUT / "ismb_abstract.txt").write_text(COMMUNITY + "\n", encoding="utf-8", newline="\n")
    (OUT / "glbio_full_abstract.txt").write_text(plain(STRUCTURED) + "\n", encoding="utf-8", newline="\n")
    (OUT / "cbm_abstract.txt").write_text(plain(abstract) + "\n", encoding="utf-8", newline="\n")
    (OUT / "cbm_highlights.txt").write_text("\n".join("- " + h for h in HIGHLIGHTS) + "\n", encoding="utf-8", newline="\n")
    highlights_tex = r"\documentclass{article}\begin{document}\section*{Highlights}\begin{itemize}" + "\n"
    highlights_tex += "\n".join(r"\item " + h for h in HIGHLIGHTS)
    highlights_tex += "\n" + r"\end{itemize}\end{document}" + "\n"
    (OUT / "cbm_highlights.tex").write_text(highlights_tex, encoding="utf-8", newline="\n")
    (OUT / "figure_alt_text.txt").write_text(ALT.replace("--", "-") + "\n", encoding="utf-8", newline="\n")
    # Normalize the presentation copy, leaving the original worktree untouched.
    (OUT / "references.bib").write_text((ROOT / "paper/references.bib").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    counts = {"glbio_full_abstract": len(plain(STRUCTURED).split()), "glbio_community_abstract": len(COMMUNITY.split()), "cbm_abstract": len(plain(abstract).split()), "ismb_portable_abstract": len(COMMUNITY.split())}
    assert all(count <= 250 for count in counts.values())
    assert all(len(h) <= 85 for h in HIGHLIGHTS)
    if compile_pdfs:
        for name in ("glbio", "glbio_abstract", "cbm"):
            build_dir = ROOT / "submission/build/venues" / name
            build_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=" + str(build_dir), name + ".tex"], cwd=OUT, capture_output=True, text=True)
            (build_dir / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{name} compilation failed; see {build_dir}/build.log")
            log = (build_dir / (name + ".log")).read_text(errors="replace")
            overflow = [line for line in log.splitlines() if "Overfull" in line]
            # The unmodified OUP Modern Large running head emits these two
            # diagnostics even in a three-page control containing only
            # "Control.". Preserve the class and record them, rather than
            # changing journal fonts or spacing. Check rendered pages too.
            vendor_head_warnings = {
                r"Overfull \hbox (261.76535pt too wide) has occurred while \output is active",
                r"Overfull \vbox (3.0pt too high) has occurred while \output is active []",
            }
            unexpected = [line for line in overflow if name != "glbio" or line not in vendor_head_warnings]
            if unexpected or "There were undefined references" in log:
                raise RuntimeError(f"{name} has unresolved layout or references")
            shutil.copyfile(build_dir / (name + ".pdf"), OUT / (name + ".pdf"))
    inputs = [ROOT / "paper/main.tex", ROOT / "paper/references.bib", Path(__file__)]
    inputs += sorted((ROOT / "paper/generated").glob("*.tex"))
    inputs += sorted((ROOT / "paper/figures").glob("*.pdf"))
    def input_bytes(path):
        if path == ROOT / "paper/references.bib":
            return path.read_text(encoding="utf-8").encode("utf-8")
        return path.read_bytes()

    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(input_bytes(p)).hexdigest() for p in inputs}
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps({"inputs": hashes, "input_hash_normalization": {"paper/references.bib": "UTF-8 with LF newlines; original worktree file is unchanged"}, "files": files, "word_counts": counts, "science": "Unchanged saved-evidence master; presentation adaptations only", "state": "Local candidates, not submitted; route checks in docs/submission_routes.md"}, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Keep the editable journal upload ZIP local. Preserve relative TeX paths,
    # include only referenced images, and exclude scientific input datasets.
    sources = [OUT / "cbm.tex", OUT / "references.bib", OUT / "cbm_highlights.tex",
               ROOT / "paper/supplement.tex", ROOT / "paper/references.bib"]
    sources += sorted((ROOT / "paper/generated").glob("*.tex"))
    for source in list(sources):
        base = OUT if source.name == "cbm.tex" else ROOT / "paper"
        for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source.read_text(encoding="utf-8")):
            path = (base / image).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f"Missing or external journal figure: {image}")
            sources.append(path)
    sources = sorted(set(sources))
    source_manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(input_bytes(p)).hexdigest() for p in sources}
    archive = ROOT / "submission/glbio-selected/cbm-source.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sources:
            entry = zipfile.ZipInfo(source.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, input_bytes(source))
        entry = zipfile.ZipInfo("source-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(entry, json.dumps(source_manifest, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Compile three PDFs with installed latexmk")
    build(parser.parse_args().compile)
