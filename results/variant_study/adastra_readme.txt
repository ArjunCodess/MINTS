-|==============================================|-
   ADASTRA release Mabel v.6.1 Nov 2024: README
-|==============================================|-

This release was constructed on top of ADASTRA Bill Cipher v5, details on v5 are provided in a separate section below.

The primary change from v5 is the following:
the ASBs were reannotated with HOCOMOCO v12 taking into account motif subtypes, and the motif P-value threshold was set at 0.001.
Motif annotation and concordance evaluation were performed for all ASBs passing 10% FDR.

Release files structure:
For each cell type (CL subdirectory) and for each TF (TF subdirectory) there is a separate file listing all putative ASB events at eligible SNVs that pass the necessary coverage thresholds.

NOTE that the file format was changed in this release to use the bed-like coordinate system and thus to be more consistent with UDACHA (https://udacha.autosome.org).

Each tsv-file is a plain tab-separated text document containing one line per single-nucleotide variant (SNV) with the following columns.

  'chr': SNV chromosome, hg38 genome assembly
  'start': SNV position, hg38, 0-based
  'end': SNV position, hg38, 1-based (mimicking 'interval end' of the bed format)
  'ID': rsSNP ID of the SNV according to the dbSNP build 151
  'ref': reference allele (A,C,G, or T, according to hg38)
  'alt': alternative allele
  'repeat_type': type of the repetitive region (if any) encompassing the SNV according to the UCSC RepeatMasker track

  'mean_BAD': mean background allelic dosage (BAD) of the genomic segment encompassing the SNV across all the aggregated experiments. Higher BAD values correspond to the higher contribution of aneuploidy and local copy-number variants. BAD values are taken into account when estimating the statistical significance of individual candidate ASBs (found in different experiments). Mean BAD is computed across all SNV that were used in the statistical aggregation of the particular ASB call.
  'mean_SNP_per_segment': mean number of SNPs in a region with the constant common  BAD
  'n_aggregated': the number of datasets in aggregation
  'total_cover': total read coverage of all aggregated SNVs

  'es_mean_ref', 'es_mean_alt': allele-wise effect size (log2),  weighted-average of log-ratios of observed and expected allelic read counts (negative logarithms of individual P-values from each dataset used as weights).

  'fdrp_bh_ref', 'fdrp_bh_alt': allele-wise logit-aggregated and FDR-corrected P-values

For TF-ASBs of transcription factors with motifs available in the HOCOMOCO v.12 (https://hocomoco.autosome.org) rSNP-adapted motif collection, the P-values of the best motif hits were calculated
for the Reference and Alternative alleles using PERFECTOS-APE (https://opera.autosome.org/perfectosape).
The motif position reflects the best hit considering both the Reference and the Alternative alleles on both DNA strands:
  'motif_log_pref': -log10(motif P-value) for the best motif occurrence of the PWM (position weight matrix) for the Ref allele
  'motif_log_palt':  -log10(motif P-value) for the Alt allele
  'motif_fc': motif Fold Change (FC), log2-ratio between motif P-values for the Reference and Alternative alleles. Positive values indicate Alt-ASBs (preferred binding to the Alternative allele). Negative values indicate Ref-ASBs. The value of ‘None’ is assigned in case the PWM was not available.
  'motif_pos': position of the SNV relative to the best PWM hit (taking into account the strand orientation of the motif hit), 0-based
  'motif_orient': '+' or '-', the DNA strand of the best motif PWM hit relative to the chromosome sequence in the genome assembly
  'motif_conc': Motif Concordance indicates whether the allelic read imbalance agrees with the motif Fold Change (FC, predicted from sequence analysis). Concordance is assessed for ASBs passing 10% FDR. The following notation is used:
'None': Motif is not available or both fdrp_bh_ref and fdrp_bh_alt are > 0.25
'No hit': The best hit P-value is higher than 0.001
'Weak concordant': The absolute value of FC is less than 2 but consistent with the allelic read imbalance
'Weak discordant': The absolute value of FC is less than 2 and contrasts with the allelic read imbalance
'Concordant': The absolute value of FC is greater or equal to 2 and consistent with allelic read imbalance
'Discordant': The absolute value of FC is greater or equal to 2 and contrasts with allelic read imbalance
  'motif_index': HOCOMOCO v12 motif subtype index. When there were multiple motif subtypes with hits passing P-Value < 0.001 we selected the one yielding the largest absolute motif 'Fold Change'.

-|==========================================================================================|-
   BillCipher v5 Release Notes, July 2022, fixed rare scoring issues with high-covered SNPs
-|==========================================================================================|-

This release includes allele-specific binding sites (ASBs) for 1073 transcription factors and other DNA-binding proteins and 649 cell types obtained by meta-analysis of 15970 read alignments of GTRD ver. 20.06.

Compared to the published version (doi: 10.1038/s41467-021-23007-0), ASBs in this release show better enrichment for eQTLs and associations with phenotype traits, as well as a higher concordance with the TF recognition motifs (85% concordant ASBs at FDR < 0.05, over 90% concordant ASBs at FDR < 0.00005).

Key changes from the previous release (v4):

(1) BAD calling: to maintain ASB calling sensitivity and representation of diploid regions achieved the latest releases (Zanthar) but improve specificity (closer to release Soos), in this release we introduced the 'geometric' prior (BABACHI v1.5.8, see GitHub github.com/autosome-ru/BABACHI), which specifies prior BAD probabilities as [p, p^2, p^3, ... , p^n, ...] where n is the minimal and most likely total copy number for a given BAD (n = x + y for BAD = x/y). The value of p=0.99 resulted in the best tradeoff between sensitivity and specificity of diploid regions calls, correctly recalling 60% of SNVs with BAD=1 across the ADASTRA datasets.

(2) Underlying data: 3 datasets attributed to 22RV1 in GTRD were re-labeled as 'prostate cancer' according to the respective GEO record (GSE120738) and re-processed using individual BAD maps. Samples from cranial neural crest cells (GSE70751) of different donors were also reprocessed with individual BAD maps.

-|==============================|-
   Further comments and details
-|==============================|-

ASB significance
ASB calling is done separately for each ChIP-Seq experiment. For each candidate ASB site, the P-values for Reference and Alternative allele are calculated separately according to the fitted Negative Binomial Mixture model accounting for different assignment of the alleles to the higher or lower DNA copies in genomic regions with BAD > 1. Prior mixture weights obtained with the global fit across SNVs were updated with Bayesian estimation separately for each SNV.

For a particular SNV, the P-values from individual data sets are aggregated with the logit (Mudholkar-George) method for each TF (using ChIP-Seq data from all cell types) and cell type (using ChIP-Seq data from all TFs) and FDR-corrected with the Benjamini-Hochberg procedure for SNVs (for each TF and each cell type separately). The aggregated P-values are then corrected for multiple tested SNPs using Benjamini-Hochberg (FDR) procedure.

ASB effect size
The Effect Size of an ASB event is calculated separately for Reference and Alternative alleles and is defined as the weighted mean of log-ratios of observed and expected allelic read counts, with weights being -log10 of the respective P-values. The expected read counts are estimated from the fitted Negative Binomial Mixture model accounting for different assignments of the allies to the higher or lower DNA copies in genomic regions with BAD > 1. Prior mixture weights obtained with the global fit across SNVs were updated with Bayesian estimation separately for each SNV.

The Effect Size is not assigned (n/a) if all the raw individual P-values of an SNV on a particular genome position are equal to 1, considering Ref- and Alt-ASBs separately.
