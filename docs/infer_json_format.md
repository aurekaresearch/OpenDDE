# Inference JSON Format


OpenDDE input is a JSON file whose top-level value is a non-empty list of jobs.
It uses AlphaFold Server-style entity keys (`proteinChain`, `dnaSequence`,
`rnaSequence`, `ligand`, `ion`), not the single-job `alphafold3` dialect.

Minimal job:

```json
[
  {
    "name": "example_job",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIKLMNPQRSTVWY",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and is omitted here; see the section below for when
to add it.

Job fields:

| Field | Required | Meaning |
| --- | :---: | --- |
| `name` | Yes | Job name used in output paths. |
| `sequences` | Yes | List of entities. Each item has exactly one entity key. |
| `modelSeeds` | No | Default seeds for the job. Overridden by `--seeds`; if neither is set, a random seed is sampled. |
| `covalent_bonds` | No | Explicit covalent links between entities. |

Every entity has `count`. Optional `id` is a list of chain IDs; its length must
match `count`.

## `proteinChain`

```json
{
  "proteinChain": {
    "sequence": "ACDEFGHIKLMNPQRSTVWY",
    "count": 1,
    "id": ["A"],
    "modifications": [
      {"ptmType": "CCD_MSE", "ptmPosition": 1}
    ],
    "pairedMsaPath": "/absolute/path/to/pairing.a3m",
    "unpairedMsaPath": "/absolute/path/to/non_pairing.a3m",
    "templatesPath": "/absolute/path/to/hmmsearch.a3m"
  }
}
```

- `sequence`: 20 standard amino-acid letters plus `X`.
- `ptmType`: CCD code prefixed with `CCD_`; `ptmPosition` is 1-based.
- `pairedMsaPath`, `unpairedMsaPath`: optional protein A3M files.
- `templatesPath`: optional template hits file (`.a3m` or `.hhr`), used only with
  `--use_template true`.

## `dnaSequence`

```json
{
  "dnaSequence": {
    "sequence": "GATTACA",
    "count": 1,
    "id": ["D"],
    "modifications": [
      {"modificationType": "CCD_6MA", "basePosition": 2}
    ]
  }
}
```

- Supported documented letters: `A`, `T`, `G`, `C`, `N`, `X`.
- DNA is single-stranded; add another `dnaSequence` for the other strand.
- `basePosition` is 1-based.

## `rnaSequence`

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "id": ["R"],
    "modifications": [
      {"modificationType": "CCD_5MC", "basePosition": 4}
    ],
    "unpairedMsaPath": "/absolute/path/to/rna_msa.a3m"
  }
}
```

- Supported documented letters: `A`, `U`, `G`, `C`, `N`, `X`.
- `unpairedMsaPath` is optional and used only with `--use_rna_msa true`.

## `ligand`

```json
{
  "ligand": {
    "ligand": "CCD_ATP",
    "count": 1,
    "id": ["L"]
  }
}
```

`ligand` can be:

- A CCD code prefixed with `CCD_`, e.g. `CCD_ATP`.
- Multiple CCD codes joined by underscores, e.g. `CCD_NAG_BMA_BGC`.
- A 3D ligand file prefixed with `FILE_` (`.pdb`, `.sdf`, `.mol`, `.mol2`).
- A SMILES string.

## `ion`

```json
{
  "ion": {
    "ion": "MG",
    "count": 2,
    "id": ["M", "N"]
  }
}
```

Ion codes are CCD component names without the `CCD_` prefix.

## `covalent_bonds`

```json
"covalent_bonds": [
  {
    "entity1": "1",
    "copy1": 1,
    "position1": "2",
    "atom1": "SG",
    "entity2": "2",
    "copy2": 1,
    "position2": "1",
    "atom2": "C1"
  }
]
```

Fields:

- `entity1`, `entity2`: 1-based indices in `sequences`.
- `copy1`, `copy2`: optional 1-based copy indices.
- `position1`, `position2`: 1-based residue/ligand-part positions.
- `atom1`, `atom2`: atom names. Integer references are also accepted for mapped
  SMILES or file ligands.

Use `entity1`/`entity2` for new inputs. The old `left_entity`/`right_entity`
style is accepted for compatibility.

## Unsupported `constraint`

The inference-only build ignores legacy `constraint` fields. Use
`covalent_bonds` for supported covalent links.

## Output layout

`opendde pred` writes:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
├── <job_name>_sample_<rank>.cif
├── <job_name>_summary_confidence_sample_<rank>.json
└── <job_name>_full_data_sample_<rank>.json   # only when --need_atom_confidence true
```

The summary JSON includes confidence metrics such as `plddt`, `gpde`, `ptm`,
`iptm`, clash flags, and `ranking_score` when available.
