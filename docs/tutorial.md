# OpenDDE Tutorial


A short walkthrough using files in [`examples/`](../examples). For install and
runtime data setup, see [inference_instructions.md](./inference_instructions.md)
or [docker_installation.md](./docker_installation.md).

## 1. Check the environment

Run commands from the repository root:

```bash
opendde doctor
export OPENDDE_ROOT_DIR=/path/to/opendde_data
```

Prediction needs:

```text
$OPENDDE_ROOT_DIR/checkpoint/opendde.pt
$OPENDDE_ROOT_DIR/common/
```

Released checkpoints keep the filenames `opendde.pt` and `opendde_abag.pt`.
Their download links and digests live in
[supported_models.md](./supported_models.md). Place them under
`$OPENDDE_ROOT_DIR/checkpoint/`, preserving those filenames. Pass
`opendde_abag.pt` directly with `--load_checkpoint_path` for ABAG runs.

```bash
bash scripts/download_opendde_data.sh \
  --root "$OPENDDE_ROOT_DIR" \
  --skip-search-database
```

The helper checks the released checkpoint's byte size and SHA-256 against the
bundled manifest before installation and prepares the required `common/`
runtime files in the same command.

Template/RNA-MSA preprocessing also needs `hmmer`; template inference may need
`kalign`.

## 2. Compatibility prediction

This disables external features and keeps the standard step/cycle counts.
Inference defaults to `fp32` and `auto` triangle kernels (PyTorch on CPU), so no
extra dtype or kernel flags are needed:

```bash
opendde pred \
  -i examples/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Outputs go to:

```text
output/<job_name>/seed_<seed>/predictions/
```

## 3. Input JSON basics

OpenDDE input is a list of jobs:

```json
[
  {
    "name": "tiny",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional here and can be left out; it is only needed to
declare explicit covalent links between entities.

Entity keys include `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, and
`ion`. Full schema: [infer_json_format.md](./infer_json_format.md).

Convert a PDB/CIF instead of writing JSON by hand:

```bash
opendde json -i examples/7pzb.pdb -o ./output --altloc first
```

## 4. Use precomputed MSA/template features

[`examples/examples_with_template/example_9fm7.json`](../examples/examples_with_template/example_9fm7.json)
already contains `pairedMsaPath`, `unpairedMsaPath`, and `templatesPath`:

```bash
opendde pred \
  -i examples/examples_with_template/example_9fm7.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa true \
  --use_template true \
  --use_rna_msa false
```

## 5. Generate MSA/template features

For an input without MSA/template paths:

```bash
opendde prep -i examples/example_without_msa.json -o ./output
```

This writes an updated JSON next to the input. Predict from that updated JSON:

```bash
opendde pred \
  -i examples/example_without_msa-final-updated.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa true \
  --use_template true \
  --use_rna_msa false
```

For protein MSA only, use `opendde msa`. For protein MSA + template only, use
`opendde mt`.

## 6. RNA MSA example

[`examples/examples_with_rna_msa/example_9gmw_2.json`](../examples/examples_with_rna_msa/example_9gmw_2.json)
contains a precomputed RNA MSA:

```bash
opendde pred \
  -i examples/examples_with_rna_msa/example_9gmw_2.json \
  -o ./output \
  -n opendde_v1 \
  --use_rna_msa true
```

To generate RNA MSA for your own RNA input, run `opendde prep` first.

## More details

- [Inference instructions](./inference_instructions.md)
- [Input JSON format](./infer_json_format.md)
- [MSA/template/RNA-MSA pipeline](./msa_template_pipeline.md)
- [Kernel options](./kernels.md)
