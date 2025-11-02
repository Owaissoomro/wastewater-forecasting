# Data directory

This project consumes three human-readable, long-format CSV files placed here. All inputs are validated against configs/schema.json; extra or missing columns, wrong types, or out-of-range values will cause a hard failure during preprocessing.

Required files
- jahn_like.csv — long SNV count/coverage table (required)
- signatures.csv — lineage-by-mutation signature matrix in long format (required)
- lineages.csv — lineage annotations (optional but recommended)

Common conventions
- Encoding: UTF-8 (no BOM), Unix line endings (\n), comma separator, period as decimal point.
- Header row required; column order arbitrary; no duplicate column names.
- Missing values: only allowed in notes (lineages.csv). All other fields must be present.
- IDs should be ASCII: ^[A-Za-z0-9._-]+$ for sample_id, site_id, lineage.
- Dates must be ISO 8601 (YYYY-MM-DD), zero-padded, Gregorian calendar, local sampling date.
- Mutation identifiers must match exactly across jahn_like.csv and signatures.csv.
- Sorting: rows may be unordered. Preprocessing will sort by site_id, date, mutation for reproducibility.

Mutation key format (recommended)
- Canonical nucleotide substitution: REF POS ALT with no spaces, uppercase; regex: ^[ACGT][0-9]+[ACGT]$ (e.g., A23403G).
- If using amino-acid form, keep consistent everywhere; regex: ^[A-Za-z0-9]+:[A-Z][0-9]+[A-Z]$ (e.g., S:D614G).
- Do not mix forms across files. Preprocessing harmonizes common variants conservatively but exact matches are safest.

Replicates and batching
- sample_id must be unique per physical extraction/library. Technical or biological replicates share the same site_id and date but have distinct sample_id.
- Multiple rows per sample_id are expected (one per mutation). Missing mutations in a sample imply missingness (not zero); coverage=0 should be explicit if measured and found to be zero.
- Do not pre-aggregate across replicates; the pipeline estimates and diagnoses replicate concordance.

File: jahn_like.csv (required)
Purpose
- Observed SNV counts and per-locus coverages for each sample.

Columns and constraints
- sample_id (str): Unique identifier for the sample. Regex: ^[A-Za-z0-9._-]+$.
- site_id (str): Sampling site identifier. Regex: ^[A-Za-z0-9._-]+$.
- date (YYYY-MM-DD): Sampling date. ISO 8601, zero-padded.
- mutation (str): Mutation key. Must match signatures.csv.
- count (int >= 0): Number of reads supporting the alternate allele; must be finite and integer.
- coverage (int >= 0): Total reads covering the locus (ALT+REF after QC); count ≤ coverage.

Additional rules
- For a given sample_id+mutation, there must be exactly one row.
- If coverage = 0, then count must be 0.
- Use non-negative integers; do not include percentages or floating counts.
- Large counts are supported (32-bit safe), but ensure CSV has no thousands separators.

Tiny example (first 8 rows)
sample_id,site_id,date,mutation,count,coverage
WW_0001_P1,SITE_A,2023-01-03,A23403G,15,500
WW_0001_P1,SITE_A,2023-01-03,C14408T,9,480
WW_0001_P1,SITE_A,2023-01-03,G25563T,0,470
WW_0002_P2,SITE_A,2023-01-10,A23403G,22,520
WW_0002_P2,SITE_A,2023-01-10,C14408T,10,515
WW_0100_P1,SITE_B,2023-01-05,S:D614G,18,600
WW_0100_P1,SITE_B,2023-01-05,ORF1ab:P4715L,7,590
WW_0100_P1,SITE_B,2023-01-05,N:R203K,0,575

File: signatures.csv (required)
Purpose
- Lineage-by-mutation “signature” weights in long format for the deconvolution likelihood.

Columns and constraints
- mutation (str): Mutation key; must match jahn_like.csv exactly.
- lineage (str): Lineage name (e.g., B.1.1.7, BA.2); Regex: ^[A-Za-z0-9._-]+$.
- weight (float in [0,1]): Expected probability that a read from the lineage exhibits the mutation after curation. Values may be 0 or 1; intermediate values encode partial penetrance or QC adjustments.

Additional rules
- Composite key (mutation, lineage) must be unique.
- Each mutation may appear in multiple lineages (overlaps allowed).
- It is not required that weights across lineages sum to 1 for any mutation.
- Keep a consistent mutation naming scheme across files.

Tiny example
mutation,lineage,weight
A23403G,B.1.1.7,1.0
A23403G,BA.2,1.0
C14408T,B.1.1.7,1.0
G25563T,B.1.1.7,0.1
S:D614G,BA.2,1.0
ORF1ab:P4715L,BA.2,0.9
N:R203K,BA.2,0.0

File: lineages.csv (optional but recommended)
Purpose
- Descriptive metadata and flags for lineages used in reporting and stratified diagnostics.

Columns and constraints
- lineage (str): Must match signatures.csv; unique per row.
- label (str): Human-readable label (e.g., “Alpha (B.1.1.7)”); free text.
- is_voc (bool): Variant-of-concern flag; accepted values (case-insensitive): true/false or 1/0.
- notes (str): Optional free text; may be empty.

Tiny example
lineage,label,is_voc,notes
B.1.1.7,Alpha (B.1.1.7),true,WHO VOC early 2021
BA.2,Omicron BA.2,true,Sublineage of Omicron
B.1.177,B.1.177,false,Background lineage

Data quality checklist (strongly recommended)
- Counts and coverage
  - 0 ≤ count ≤ coverage for every row.
  - No negative or missing integers in count/coverage.
  - Coverage reflects post-QC, per-locus depth (after primer/quality filtering).
- Identifiers and dates
  - sample_id unique; no trailing/leading whitespace anywhere.
  - date adheres to YYYY-MM-DD; no time or timezone components.
- Mutation strings
  - One naming convention used consistently across all files.
  - No whitespace or ambiguous IUPAC codes; uppercase only for nucleotides and amino acids.
- Completeness
  - All mutations used in signatures.csv should appear at least once in jahn_like.csv across the dataset; otherwise they will be flagged as systematically missing.
- Replicates
  - Replicates present as distinct sample_id with the same site_id+date; do not average ahead of time.

Size and performance tips
- Keep each CSV reasonably sized for reproducible runs. As a rule of thumb, ≤ 1–2 million rows per file keeps memory usage comfortable on laptops.
- If jahn_like.csv is very large:
  - Consider pre-filtering obviously irrelevant mutations to the target signatures (while keeping constraints above intact).
  - Avoid Excel exports that introduce quoted numeric cells, localized number formats, or stray whitespace.
- Compression
  - If you compress for storage, decompress to plain CSV before running. Do not rely on OS-specific file associations.
- Stability
  - Use UTF-8 without BOM. Avoid non-breaking spaces or locale-specific characters in IDs. Ensure consistent line endings if collaborating across OSes.

Privacy and ethics
- Wastewater sequencing aggregates population signals but can still enable sensitive inferences at small sites or sparse time points.
- Recommendations:
  - Use pseudonymous site_id values (e.g., SITE_A) in public artifacts unless stakeholder consent and policies permit disclosure.
  - Do not include personally identifiable information, geographic coordinates, or patient-level metadata in any input file.
  - Apply minimum-cell-size policies when publishing derived summaries (e.g., do not highlight single-day signals at very low coverage).
  - Ensure your data sharing complies with local regulations, IRB approvals, and data use agreements.

Provenance and reproducibility
- Record how raw fastq-derived counts were computed (e.g., mapper, variant caller, filters) outside of this repository and keep that documentation with your dataset.
- Any modification or regeneration of these CSVs should be accompanied by a new data version tag and changelog.

Validation behavior (what to expect)
- Extra columns or wrong dtypes cause a validation error.
- Invalid values (e.g., count > coverage, negative coverage, malformed date) are rejected.
- Mutations present in signatures.csv but never observed will be reported as systematically missing; mutations observed but absent from signatures.csv will be carried through as “unknown” features only if explicitly configured, otherwise flagged.