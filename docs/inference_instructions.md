# Inference Instructions

Concise reference for installing OpenDDE, preparing runtime data, and running
`opendde` commands.

## Install

OpenDDE supports CPython `3.11`, `3.12`, and `3.13`. We recommend
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python
installations. Choose one of the following methods.

### Install from PyPI

```bash
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu opendde
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 "opendde[gpu]"
```

### Install from source

```bash
git clone https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu -e .
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

After a PyPI or source installation, verify the environment with:

```bash
uv run --no-project --python .venv opendde doctor
```

### Use Docker

The prebuilt image targets NVIDIA GPU inference:

```bash
docker pull aurekaresearch/opendde:v1
```

See the [Docker guide](./docker_installation.md) for GPU setup, runtime-data
mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU (MPS is not supported); Intel macOS is unsupported, and
> Windows has not been validated. At runtime, `--device auto` uses CUDA when
> available and otherwise falls back to CPU.

## Runtime data

Set `OPENDDE_ROOT_DIR` to the directory that stores checkpoints and runtime data:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA search
```

The default checkpoint and managed common files come from a release-pinned
asset revision and are verified against their published size and SHA-256.
Checkpoints passed explicitly with `--load_checkpoint_path` are left untouched.

Prepare data from a source checkout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

For a protein-only prediction that disables MSA, template, and RNA-MSA features,
search databases are not needed:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

If you already have a custom checkpoint, keep a descriptive filename and pass
it directly. Use `--skip-model` when preparing only the remaining runtime data:

```bash
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /path/to/my_checkpoint.pt \
  "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt"
bash scripts/download_opendde_data.sh --skip-model
opendde pred \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt" \
  -i examples/input.json \
  -o ./output
```

The names `opendde.pt` and `opendde_abag.pt` are reserved for released assets.
Their authoritative links, sizes, and digests are in
[supported_models.md](./supported_models.md).

Use `opendde.pt` with `-n opendde_v1` as the default general-purpose
checkpoint. To use the ABAG-optimized checkpoint, keep it as
`opendde_abag.pt` and pass it with `--load_checkpoint_path`, for example
`opendde pred --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"`.

Install and verify the ABAG checkpoint from the same manifest-backed helper:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh \
  --checkpoint opendde_abag.pt \
  --skip-common \
  --skip-search-database
```

Then run general-purpose inference without an explicit checkpoint path. For ABAG
inference, add:

```bash
--load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENDDE_ROOT_DIR` | Checkpoints, common files, search databases. Defaults to `~/.cache/opendde`. |
| `OPENDDE_DEPENDENCY_URL` | Override checkpoint download root. |
| `OPENDDE_COMMON_URL` | Override common runtime file download root. Falls back to `OPENDDE_DEPENDENCY_URL` when set. |
| `OPENDDE_SEARCH_DATABASE_URL` | Override template/RNA-MSA database download root. |
| `LAYERNORM_TYPE` | LayerNorm backend; defaults to `torch`. Set to `fast_layernorm` to opt into the fused kernel. |

Template/RNA-MSA preprocessing also needs HMMER. Template inference may need
`kalign`:

```bash
apt-get update && apt-get install -y hmmer kalign
```

## Input JSON

OpenDDE input is a top-level list of jobs:

```json
[
  {
    "name": "tiny",
    "modelSeeds": [101],
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

`covalent_bonds` is optional and may be omitted from a job; include it only to
declare explicit covalent links between entities.

Full schema: [infer_json_format.md](./infer_json_format.md).

Convert a structure file to JSON:

```bash
opendde json -i examples/7pzb.pdb -o ./output --altloc first
opendde json -i examples/2lwu.cif -o ./output --altloc first --assembly_id 1
```

## Preprocess optional features

```bash
# Protein MSA
opendde msa -i examples/input.json -o ./output

# Protein MSA + template search
opendde mt -i examples/input.json -o ./output

# Protein MSA + template search + RNA MSA when RNA is present
opendde prep -i examples/input.json -o ./output
```

Notes:

- Protein MSA uses the public ColabFold MMseqs2 API unless A3M paths are already
  present in the JSON.
- Template and RNA-MSA search use local databases under
  `$OPENDDE_ROOT_DIR/search_database/`.
- Updated JSON files are written next to the input JSON.

Details: [msa_template_pipeline.md](./msa_template_pipeline.md).

## Run prediction

Standard run:

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1
```

Compatibility run with the standard step/cycle counts:

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

Inference defaults to `--device auto`, `fp32`, and `auto` triangle kernels.
Device auto-selection uses NVIDIA CUDA when available and otherwise CPU.
cuEquivariance is selected only when its Linux CUDA packages import successfully;
otherwise the model uses PyTorch triangle kernels.

## Seed batching and multi-GPU inference

Let `D=--foldcp_size_dp`, `P=--foldcp_size_cp`, and
`B=--seed_batch_size`. `D` and `P` define process placement; `B` is the maximum
number of seeds in one rank-local model batch and defaults to `1`.

| Mode | `foldcp_mode` | `D` | `P` | `B` | Required world size |
| --- | --- | ---: | ---: | ---: | ---: |
| Scalar single GPU | `single` | 1 | 1 | 1 | 1 |
| Batched single GPU | `single` | 1 | 1 | >1 | 1 |
| Sharded seed batches | `single` | >1 | 1 | >=1 | `D` |
| Fold-CP | `distributed` | 1 | >1 | 1 | `P` |
| Batched Fold-CP | - | any | >1 | >1 | Rejected |
| Hybrid | - | >1 | >1 | any | Rejected |

`WORLD_SIZE` must equal `D * P`. Hybrid and mismatched-world-size launches fail
instead of falling back to serial inference. Multi-GPU inference in this release
is single-node and requires all ranks to share the input and output filesystem.

### Rank-local seed batches

On one GPU, set `D=1, P=1` and choose `B>1`:

```bash
opendde pred \
  -i input.json -o output_b2 -n opendde_v1 \
  --seeds 101,102 \
  --seed_batch_size 2
```

Multi-GPU execution uses the same path after assigning rank `r` the ordered
slice `seeds[r::D]`. Each rank chunks that slice into groups of at most `B`, so
a smaller final batch is valid. For example, four GPUs with two seeds per GPU
use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i input.json -o output_d4_b2 -n opendde_v1 \
  --seeds 101,102,103,104,105,106,107,108 \
  --seed_batch_size 2 \
  --foldcp_mode single \
  --foldcp_size_dp 4 \
  --foldcp_size_cp 1
```

Seed batches use a leading model dimension separate from `--sample`: model
tensors are shaped conceptually as `[B_seed, N_sample, ...]`. Outputs are split
after model inference and remain one directory per seed. Seeds must be unique
when either `D>1` or `B>1`, and multi-rank runs require at least `D` seeds.
Per-seed random streams remain independent, but BF16 predictions need not be
bitwise identical across batch widths because batched kernels can use different
floating-point launch geometry.

`B>1` requires `P=1`, `num_workers=0`, `model.N_model_seed=1`, and Training-Free
Guidance disabled.
OpenDDE does not choose a width, automatically reduce it, or retry seeds
serially after an OOM; select a batch size that fits the input and GPU. `B=1`
preserves the previous tensor shapes and memory profile.

### Fold-CP `1 x P`

> [!IMPORTANT]
> Fold-CP inference does not currently support cuEquivariance (`cueq`) triangle
> kernels. Select the distributed PyTorch implementations with
> `--triatt_kernel torch --trimul_kernel torch`. On CUDA BF16, Fold-CP triangle
> attention also uses Triton 3.3.1 from the GPU install extra to fuse
> attention-bias addition; this Triton helper is separate from cuEquivariance.

Fold-CP distributes token-pair-heavy inference work over a `1 x P` mesh, where
`P` can be any available GPU count greater than one. Launch it with `torchrun`
and expose exactly the GPUs you want to use. For example, four GPUs use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i examples/protein_200.json \
  -o ./output_foldcp \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --trimul_kernel torch \
  --triatt_kernel torch \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Runtime notes:

- `--nproc_per_node P` must match `--foldcp_size_dp 1` times
  `--foldcp_size_cp P`.
- `--foldcp_size_cp P` creates a `1 x P` context-parallel mesh. The example
  uses `P=4`; other GPU counts require changing both `4` values.
- Keep the input, model, dtype, cycle, step, sample, MSA, and template settings
  identical when comparing single-GPU and Fold-CP outputs. Select
  `--triatt_kernel torch --trimul_kernel torch` for both runs; the distributed
  CUDA BF16 path additionally uses Triton for attention-bias fusion.
- Outputs are written under the requested `-o/--out_dir` just like normal
  inference.
- Optional `--foldcp_metrics_jsonl path/to/metrics.jsonl` records Fold-CP timing
  and memory metrics.

For single-GPU inference, omit the topology flags or set
`--foldcp_mode single --foldcp_size_dp 1 --foldcp_size_cp 1`. Fold-CP requires
the default `--seed_batch_size 1`.

Use prepared features:

```bash
opendde pred -i examples/examples_with_template/example_9fm7.json \
  -o ./output -n opendde_v1 \
  --use_msa true --use_template true

opendde pred -i examples/examples_with_rna_msa/example_9gmw_2.json \
  -o ./output -n opendde_v1 \
  --use_rna_msa true
```

## Optional TFG Guidance

OpenDDE includes default-off Training-Free Guidance (TFG) for protein-ligand
runs. TFG refines each sampled trajectory with geometry potentials while keeping
the requested `--sample` count unchanged.

`model.N_model_seed>1` and TFG currently require `--seed_batch_size 1`.

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1 \
  --use_tfg_guidance true
```

Outputs are written to:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

## Common flags

| Flag | Meaning |
| --- | --- |
| `-n`, `--model_name` | Model name. Currently `opendde_v1`. |
| `--load_checkpoint_path` | Explicit checkpoint path. |
| `--seeds` | Comma-separated seeds, e.g. `101,102`. Overrides the job's `modelSeeds`; if unset, `modelSeeds` are used, or a random seed when both are absent. |
| `--seed_batch_size` | Maximum seeds in one rank-local model batch. Defaults to `1`; values greater than one require `P=1`, `num_workers=0`, `model.N_model_seed=1`, and TFG disabled. |
| `--use_msa` | Use/generate protein MSA features. |
| `--use_template` | Use/generate template features. |
| `--use_rna_msa` | Use/generate RNA MSA features. |
| `--use_tfg_guidance` | Enable Training-Free Guidance. |
| `--foldcp_mode` | Use `single` for serial or seed-parallel inference and `distributed` only for `1 x P` Fold-CP. |
| `--foldcp_size_dp` | Number of seed-sharding ranks; values greater than one require `P=1`. |
| `--foldcp_size_cp` | Number of Fold-CP ranks per seed; values greater than one require `D=1`. |
| `--foldcp_devices` | Optional visible-device list recorded in Fold-CP metrics; actual GPU visibility is controlled by `CUDA_VISIBLE_DEVICES`. |
| `--foldcp_metrics_jsonl` | Optional JSONL path for Fold-CP timing and memory metrics. |
| `--dtype` | `bf16` or `fp32`. |
| `--device` | `auto`, `cpu`, or `cuda`; auto uses CUDA when available and otherwise CPU. |
| `--trimul_kernel`, `--triatt_kernel` | `auto`, `cuequivariance`, or `torch`. |

Run `opendde <command> --help` for the full option list.
