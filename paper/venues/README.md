# Venue presentations

GLBIO 2027 full paper is the active target. The files are local candidates, not submitted manuscripts or proof of acceptance. The source and PDFs are committed for review; large evidence archives and private submission records stay local.

| File | Use |
| --- | --- |
| `glbio.pdf`, `glbio.tex`, `glbio_full_abstract.txt` | Author-bearing OUP Modern Large proceedings candidate, maximum nine pages and 250 abstract words. |
| `glbio_abstract.txt`, `glbio_abstract.pdf` | Conditional GLBIO abstract route; text is under 250 words and the extended abstract is one page. |
| `ismb_abstract.txt` | Portable ISMB/ECCB candidate; final 2027 track compliance remains conditional on the accessible call. |
| `cbm.pdf`, `cbm.tex`, `cbm_abstract.txt`, `cbm_highlights.txt`, `cbm_highlights.tex`, `cbm_cover_letter.txt` | Sequential journal fallback; current author guide and no-fee subscription option are verified; contribution and declaration forms remain author actions. |
| `figure_alt_text.txt` | Main figure description for portal accessibility fields. |
| `oup_waiver_request.txt` | Draft supporting statement for the separate OUP process; no waiver is granted or implied. |
| `manifest.json` | Presentation input/output hashes and abstract counts. |

Use the common `paper/supplement.pdf` as a separately marked supplementary attachment when needed. Code and unchanged numeric evidence remain at their existing repository and Zenodo version; do not re-upload or rebuild the published dataset for a formatting change.

Run `python tools/build_venue_materials.py --compile`. The generator uses the scientific master and existing inserts, not model inference. Render and inspect regenerated PDFs before uploading. The full route, corrected calendar, financial dependencies and conflict rules are in [submission routes](../../docs/submission_routes.md); [calendar](../../docs/submission-calendar.ics) contains reminders, not submission authorization.

The OUP bundle is copied unchanged from the previously verified local template. Its source notices retain LPPL 1.3-or-later and the individual bibliography permissions. The root class/style copies are identical to their vendor originals. Do not license these third-party files as original MINTS code.
