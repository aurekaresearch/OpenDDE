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

### Protein template modes

`--use_template false` disables template features for every chain. With
`--use_template true`, each protein chain independently selects one mode:

| Chain fields | Behavior |
| --- | --- |
| Neither `templates` nor `templatesPath` | Eligible for the existing automatic template-search pipeline. |
| `templatesPath` | Use the existing HHR/A3M search-hit pipeline. |
| `templates: []` | Explicitly disable templates for this chain. Automatic search does not overwrite it. |
| Non-empty `templates` | Use the supplied structures and index mappings exactly; automatic search does not overwrite them. |

Do not set `templates` and `templatesPath` on the same chain. Existing inputs
that omit `templates` retain their previous automatic-search or
`templatesPath` behavior.

Each explicit template is an object with this shape:

```json
{
  "mmcifPath": "templates/target.cif",
  "queryIndices": [0, 1, 4, 5],
  "templateIndices": [0, 1, 2, 3]
}
```

- Set exactly one of `mmcifPath` or `mmcif`. `mmcif` contains the complete
  mmCIF document as a JSON string. A relative `mmcifPath` is resolved from the
  directory containing the input JSON file.
- `queryIndices` and `templateIndices` are paired, zero-based positions in the
  query sequence and parsed template polymer sequence, respectively. They are
  not PDB author residue numbers.
- Both arrays must be non-empty, have equal length, contain only non-negative
  integers, and stay within their corresponding sequences. `queryIndices` must
  be unique; array order defines the residue pairs.
- The pair at each array position defines one residue mapping. Query residues
  omitted from `queryIndices` remain untemplated, enabling sparse conditioning.
- The mmCIF must contain exactly one protein polymer chain and a valid PDBx
  revision date.
- OpenDDE uses at most four templates per protein chain, taking the first four
  in JSON order. Entries after the first four are ignored.
- Explicit templates bypass the automatic search-hit release-date cutoff and
  near-duplicate filtering. The caller is responsible for choosing suitable
  template structures and enforcing any dataset cutoff.

When every template-enabled protein chain uses explicit `templates` (or
`templates: []`), template processing does not invoke automatic search,
Kalign, a template-search database or cache, or template-network access. Normal
checkpoint and common runtime assets are still required. A mixed input still
needs search infrastructure for any chain using automatic search or
`templatesPath`.

See the runnable sparse example in
[`examples/example_explicit_template.json`](../examples/example_explicit_template.json).

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
