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

## `constraint` (TFG guidance)

Inference-only builds accept a singular `constraint` object that steers the
prediction through Training-Free Guidance (TFG). Enable it with
`--use_tfg_guidance true`. Supported keys are `contact`, `movable_chains` and
`epitope`; any other key is rejected with an error. Constraints do not create
covalent bonds; use `covalent_bonds` for hard links. Guidance modes and the
environment settings they need are described in
[TFG constraint guidance](tfg_constraint_guidance.md).

### `constraint.contact`

Soft distance restraints between atom pairs.

```json
"constraint": {
  "contact": [
    {
      "entity1": "1",
      "copy1": 1,
      "position1": "20",
      "atom1": "CA",
      "entity2": "2",
      "copy2": 1,
      "position2": "102",
      "atom2": "CA",
      "min_distance": 3.5,
      "max_distance": 8.0
    }
  ]
}
```

Fields (addressing matches `covalent_bonds`):

- `entity1`, `entity2` (or `left_entity` / `right_entity`): 1-based indices in
  `sequences`.
- `copy1`, `copy2` (or `left_copy` / `right_copy`): optional 1-based copy
  indices. Prefer explicit copies for Ab–Ag inputs.
- `position1`, `position2` (or `left_position` / `right_position`): 1-based
  residue positions.
- `atom1`, `atom2` (or `left_atom` / `right_atom`): atom names. For protein
  residues, omitted atoms default to `CA`.
- `min_distance`, `max_distance`: flat-bottom bounds in Å (defaults `3.5` /
  `8.0`).

If `constraint.contact` is present but TFG is disabled, inference logs a hint to
pass `--use_tfg_guidance true`.

### `constraint.movable_chains`

```json
"constraint": {"movable_chains": ["H", "L"], "contact": [...]}
```

A non-empty list of distinct chain IDs (`proteinChain.id`) that move together as
one rigid body during rigid-body guidance; every other chain stays where the
sampler put it. It must name some but not all chains. Without it, rigid contact
guidance supports exactly two chains and moves the one holding the second atom
of the contacts. With rigid contact guidance, every contact must join one atom
of a movable chain to one atom outside it.

### `constraint.epitope`

Asks that an antibody-like group (the `movable_chains`) binds a set of antigen
residues, without naming any antibody-side residue.

```json
"constraint": {
  "movable_chains": ["H", "L"],
  "epitope": {
    "residues": [
      {"entity": "3", "copy": 1, "position": "248", "residue": "P"},
      {"entity": "3", "copy": 1, "position": "249", "residue": "N"}
    ],
    "paratope": "all",
    "min_fraction": 0.5
  }
}
```

- `residues`: non-empty list of antigen residues. `entity`, `copy` and
  `position` are integers or strings of digits; the optional one-letter
  `residue` is checked against the sequence. A repeated residue counts once.
  Epitope residues must not lie in a movable chain.
- `paratope`: the heavy atoms of the movable chains that count as the binding
  face. `"all"` (default) uses every residue of the movable chains, `"cdr"`
  uses positions 24–40, 50–66 and 88–115 of every movable chain, and
  `{"windows": {"H": [[26, 35], [50, 66]]}}` gives explicit 1-based windows per
  movable chain.
- `min_fraction`: in `(0, 1]`, default `0.5`. The request is met when at least
  `K = max(1, ceil(min_fraction * n))` of the `n` distinct epitope residues have
  a heavy atom within 5 Å of a paratope heavy atom.

`epitope` requires `movable_chains` and cannot be combined with `contact`.
Malformed fields are errors. The epitope request is applied only by rigid-body
guidance, which runs by default when the input carries it (see
[TFG constraint guidance](tfg_constraint_guidance.md)); without
`--use_tfg_guidance true`, or with `OPENDDE_RIGID_CONTACT` set to `off` or
`control`, the run stops with an error instead of ignoring the request.

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
