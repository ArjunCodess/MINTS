# Natural-variant data feasibility

The live source inventory is saved in `results/variant_study/`. Aggregate counts
identify candidate resources; none establishes a powered independent
confirmation population after motif, token, control, donor and QC exclusions.

| Source | What is established | What remains unresolved |
| --- | --- | --- |
| GSE81945 Table S3 | 15 heterozygous variants at 13 hg19 loci; 11 singleton loci after excluding unresolved adjacent groups | Pooled replicate counts, phase, mapping bias and copy-number effects; exploratory only |
| ADASTRA Mabel v6.1 | Live API reports 80,735 CTCF records and 685 experiments; published release documents hg38 and corrected allelic dosage | Eligible independent loci/donors, source overlap, controls and biological effect normalization |
| AlleleDB | Published allele-specific binding resource with personal-genome mapping | Download access failed during this run; CTCF-specific usable counts not verified |
| BaalChIP ENCODE panel | Published analysis across multiple cell lines with dosage and mapping-bias correction | CTCF-specific records, dependence and overlap with ADASTRA/old ENCODE inputs |

ADASTRA's release is available from the official
[Zenodo deposit](https://zenodo.org/records/14174114), with its
[release README](https://adastra.autosome.org/assets/readme/readme.mabel.txt)
and [live API](https://adastra.autosome.org/api/v6/). The full release is about
943 MB, so this metadata audit records release URLs and hashes without opening
variant outcomes before assigning cohorts. The retrieved README includes
coverage-eligible records rather than only significant hits. Any import must
retain that denominator and avoid selecting variants by agreement with the
model or by motif concordance labels.

The [ADASTRA study](https://doi.org/10.1038/s41467-021-23007-0) documents dosage
correction and warns that many TF-associated allelic events lack that TF's
motif. Therefore, CTCF labels alone do not establish direct motif involvement.
The [BaalChIP study](https://doi.org/10.1186/s13059-017-1165-7) and
[AlleleDB study](https://doi.org/10.1038/ncomms11101) offer additional candidates,
but their underlying experiments can overlap other collections. A database
name is not an independence guarantee.

For confirmation-level biological measurements, obtain genotypes and phased
haplotypes, per-replicate ChIP/input counts, donor/biosample identities, dosage
information, and alignment provenance. Reprocess reads with an established
allele-mapping correction such as [WASP](https://doi.org/10.1038/nmeth.3582),
checking duplicates, mapping quality and allele-specific read losses. The
current pooled table cannot supply those checks. `biological_qc` records each
missing item instead of treating the absence of metadata as a passing result.

The current decision is to build and evaluate bounded feasibility only. A
larger resource is accessible, but the study must stop before confirmation
until eligible membership and intervention-based power are established.
