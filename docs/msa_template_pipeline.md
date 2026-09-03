# MSA, Template, and RNA MSA Pipeline


OpenDDE can use precomputed feature files or generate them with CLI helpers.

## JSON fields

Protein:

```json
{
  "proteinChain": {
    "sequence": "ACDE...",
    "count": 1,
    "pairedMsaPath": "/absolute/path/to/pairing.a3m",
    "unpairedMsaPath": "/absolute/path/to/non_pairing.a3m",
    "templatesPath": "/absolute/path/to/hmmsearch.a3m"
  }
}
```

RNA:

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "unpairedMsaPath": "/absolute/path/to/rna_msa.a3m"
  }
}
```

## Commands

```bash
# Protein MSA only
opendde msa -i examples/input.json -o ./output

# Protein MSA + template search
opendde mt -i examples/input.json -o ./output

# Protein MSA + template search + RNA MSA when RNA is present
opendde prep -i examples/input.json -o ./output
```

Generated JSON files are written under `<out_dir>/.opendde_preprocessed/<input-hash>/` rather than next to the
input JSON, so read-only input directories work; the command returns and
logs the generated path.

## Protein MSA

`opendde msa` uses the public ColabFold MMseqs2 API
(`https://api.colabfold.com`) unless MSA paths already exist. The service is
shared and rate-limited; for batch runs, provide precomputed A3M files.

Set `MMSEQS_SERVICE_HOST_URL` to use a compatible self-hosted MMseqs2 endpoint.

MSA pairing uses species information from A3M headers. Supported examples:

```text
>UniRef100_<hit_name>_<species_or_taxonomy_id>/
>tr|ACCESSION|ID_SPECIES/START-END ...
>sp|ACCESSION|ID_SPECIES/START-END ...
```

`OX=...` taxonomy annotations are not read directly.

## Template search

Template search uses HMMER (`hmmbuild`, `hmmsearch`) against:

```text
$OPENDDE_ROOT_DIR/search_database/pdb_seqres_2022_09_28.fasta
```

Run with explicit tools/database if needed:

```bash
opendde mt -i examples/input.json -o ./output \
  --hmmsearch_binary_path /path/to/hmmsearch \
  --hmmbuild_binary_path /path/to/hmmbuild \
  --seqres_database_path /path/to/pdb_seqres_2022_09_28.fasta
```

Output: `hmmsearch.a3m`, referenced by `templatesPath`. During prediction,
`--use_template true` may also call `kalign` and may need template mmCIF files
from cache or remote PDBe fetch.

## RNA MSA

RNA MSA uses HMMER (`nhmmer`, `hmmalign`, `hmmbuild`) against NT-RNA, Rfam, and
RNAcentral databases.

```bash
opendde prep -i my_rna_job.json -o ./output \
  --nhmmer_binary_path /path/to/nhmmer \
  --hmmalign_binary_path /path/to/hmmalign \
  --hmmbuild_binary_path /path/to/hmmbuild
```

Output:

```text
<out_dir>/<job_name>/rna_msa/<sequence_index>/rna_msa.a3m
```

The updated JSON points RNA `unpairedMsaPath` to that file.

## Search databases

Databases are looked up under `$OPENDDE_ROOT_DIR/search_database/` and downloaded
when missing. The default download source is the AlphaFold v3.0 `.fasta.zst`
archives, which are decompressed to the `.fasta` runtime files below.

| Database | Used by | Size |
| --- | --- | --- |
| `pdb_seqres_2022_09_28.fasta` | template | ~220 MB |
| `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | ~220 MB |
| `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` | RNA MSA | ~75 GB |
| `rnacentral_active_seq_id_90_cov_80_linclust.fasta` | RNA MSA | ~13 GB |

Override downloads:

- `OPENDDE_SEARCH_DATABASE_URL`: root for all search databases.
- `OPENDDE_PDB_SEQRES_URL`, `OPENDDE_RFAM_DB_URL`, `OPENDDE_NT_RNA_DB_URL`,
  `OPENDDE_RNACENTRAL_DB_URL`: individual database URLs.

## Dependencies

- Protein MSA: public ColabFold MMseqs2 API or `MMSEQS_SERVICE_HOST_URL`.
- Template search: `hmmbuild`, `hmmsearch`.
- RNA MSA: `nhmmer`, `hmmalign`, `hmmbuild`.
- Search database decompression: `zstd` command, or the optional Python
  `zstandard` package for Python auto-downloads.
- Template inference: `kalign`.
