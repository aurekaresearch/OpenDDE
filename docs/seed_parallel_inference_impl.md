# Seed-Parallel Inference Implementation

Status: Implemented and qualified
Date: 2026-08-25
Source: `docs/seed_parallel_inference_spec.md`

Implement the specification as a minimal runner-level extension. Do not change
model computation or add another parallelism/configuration system.

## Current behavior and gap

- `runner/inference.py::infer_predict()` runs every resolved seed serially on
  every rank.
- `opendde/data/inference/infer_dataloader.py` uses `DistributedSampler`, which
  can give cooperating Fold-CP ranks different jobs.
- `FoldCPConfig` accepts `size_dp > 1`, but no seed ownership is assigned to
  data-parallel ranks.
- Distributed input preprocessing and randomly generated seed selection are not
  single-owner operations end to end.
- Existing model and output-rank logic is valid only for `D=1` Fold-CP.

The closest implementation is the current `torchrun` initialization,
`FoldCPConfig`, rank-0 asset broadcast, scalar seed loop, and per-seed dumper.
Extend those owners; do not add a launcher, scheduler, model batch dimension, or
parallel runner.

## Work unit 1: Validate the existing topology

Extend `opendde/distributed/foldcp/config.py` and runner startup:

- preserve serial `D=1, P=1`;
- allow seed-parallel `D>1, P=1` with `foldcp_mode=single`;
- preserve Fold-CP `D=1, P>1` with `foldcp_mode=distributed`;
- reject hybrid `D>1, P>1`;
- require `WORLD_SIZE == D * P` before checkpoint/model loading; and
- update the existing launch hint, CLI help, and distributed startup messages
  for seed-parallel `torchrun`.

Keep `FoldCPConfig.enabled` meaning Fold-CP model sharding. Do not activate the
Fold-CP model path for `D x 1` seed parallelism.

## Work unit 2: Assign seeds and inputs to ranks

Extend the current runner flow without introducing a new module:

- sort directory JSON inputs once;
- for each input JSON, run preprocessing on rank 0 and broadcast its path or
  error;
- resolve that input's seeds on rank 0 and broadcast the ordered list while
  preserving the existing first-record `modelSeeds` behavior;
- reject duplicate seeds and fewer seeds than `D` when `D>1`;
- select `seeds[DIST_WRAPPER.rank::D]` for `D x 1` execution;
- use a sequential sampler for distributed inference so each rank sees every
  input job in the same order; and
- retain the current seed initialization, prediction, cleanup, and dumping loop
  for each assigned seed.

Broadcast preprocessing and seed-validation failures to every rank and return a
nonzero command result. Do not change existing ordinary per-job prediction-error
handling.

Do not modify `OpenDDE`, diffusion, confidence, Fold-CP mesh, or dumper code.
With `P=1`, every rank is already an output owner; with `D=1, P>1`, the existing
global-rank-zero output contract remains valid.

## Work unit 3: Add focused automated tests

Prefer extending `tests/test_inference_config.py` and
`tests/test_foldcp_cpn.py`; add a new test file only if those files become less
clear.

Cover:

1. `(WORLD_SIZE,D,P)` support for `(1,1,1)`, `(2,2,1)`, `(3,3,1)`,
   `(4,4,1)`, `(2,1,2)`, `(3,1,3)`, and `(4,1,4)`.
2. Rejection of `(4,2,2)`, mode/topology mismatches, and world-size mismatch.
3. Existing CLI help describing `foldcp_size_dp` seed-lane semantics.
4. Rank seed slices for even and uneven seed counts with no loss or duplication.
5. Duplicate and insufficient seed rejection before the first model forward.
6. Rank-0 success/failure propagation for per-input preprocessing and seed
   resolution, including a nonzero distributed failure result.
7. Same ordered multi-record dataset on every distributed rank.
8. `infer_predict()` invoking and dumping only the rank's assigned seeds.
9. The unchanged serial cleanup/call path and existing Fold-CP tests.

Run focused checks first:

```bash
ruff check runner opendde/distributed/foldcp \
  opendde/data/inference tests/test_inference_config.py tests/test_foldcp_cpn.py
python -m pytest -q tests/test_inference_config.py tests/test_foldcp_cpn.py
```

Then run the repository-prescribed non-network suite:

```bash
python -m pytest tests -q -m "not network"
```

## Work unit 4: Update public documentation

Update only:

- `README.md`;
- `docs/inference_instructions.md`;
- `docs/foldcp_e2e_baseline.md`; and
- `CHANGELOG.md` under `[Unreleased]`.

Use the existing flag spellings and command style. Explain that
`foldcp_mode=single` selects the normal one-card model path even when `D>1`.
Document hybrid rejection and retain all existing Fold-CP kernel restrictions.

## Local four-GPU qualification

Create the documented Python 3.11 GPU environment if the checkout does not yet
have one. Use the checked-in `examples/foldcp_demo_placeholder.json`, released
general checkpoint, `LAYERNORM_TYPE=torch`, external features disabled,
`sample=1`, `step=2`, `cycle=1`, `dtype=fp32`, PyTorch triangle kernels, TF32
disabled, and deterministic mode.

Run this matrix from the implementation commit:

| Case | World | `D` | `P` | Seeds | Expected |
| --- | ---: | ---: | ---: | --- | --- |
| Ordinary serial reference | 1 | 1 | 1 | 101-106 | Success |
| Seed parallel | 2 | 2 | 1 | 101-106 | Success |
| Seed parallel | 3 | 3 | 1 | 101-106 | Success |
| Seed parallel | 4 | 4 | 1 | 101-104 | Success |
| Two-record seed parallel | 2 | 2 | 1 | 101-104 | Success |
| Fold-CP | 2 | 1 | 2 | 101 | Success |
| Fold-CP | 3 | 1 | 3 | 101 | Success |
| Fold-CP | 4 | 1 | 4 | 101 | Success |
| Hybrid guard | 4 | 2 | 2 | 101-104 | Fail before model load |
| World-size guard | 2 | 4 | 1 | 101-104 | Fail before model load |

Run the ordinary serial reference with `python -m runner.batch_inference pred`
and omit Fold-CP topology flags. Build the two-record smoke input in a temporary
directory from the checked-in fixture with distinct job names; do not commit it.

Before the matrix, confirm all selected GPUs are idle. Preserve output and logs
until comparisons finish. For every success, record exit status, elapsed time,
peak GPU memory, produced seed/sample paths, schema/finiteness checks, and
coordinate plus summary-confidence deltas from the matching serial result.
Report the mixed GPU models/topology and treat timing as diagnostic only.

Do not commit the environment, checkpoints, generated outputs, logs, caches, or
machine-specific paths.

## Acceptance evidence

Qualification passed on 2026-08-26. The implementation commit is the PR head
containing this document, based on public `main` at
`aa53450b5583fdd6f350ed3e1f601deda766c23b` (`v1.1.0`). Runtime code was
unchanged after the GPU matrix; only this evidence was appended.

Changed owners are limited to the existing topology config, inference runner,
batch entrypoint, inference dataloader, two existing test files, the SPEC/IMPL,
and the four required public documentation files. No model, kernel, dumper,
schema, dependency, or deployment file changed.

### Automated checks

- Focused Ruff: passed.
- Focused pytest: `94 passed`.
- Non-network suite: `311 passed, 4 deselected, 44 subtests passed`.
- `pre-commit run --all-files`: Ruff check and format passed.
- `ruff format --check` and `git diff --check`: passed.

### Environment and commands

- Python 3.11.14; PyTorch 2.7.1+cu126; CUDA runtime 12.6; NCCL 2.26.2;
  Triton 3.3.1; NVIDIA driver 590.48.01.
- Released `opendde.pt`, 2,625,249,069 bytes, SHA-256
  `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`; physical GPUs 0-2 were RTX 3090 24 GiB and
  GPU 3 was an RTX 4090 reported with 49,140 MiB. Local rank followed each
  `CUDA_VISIBLE_DEVICES` list in order.
- Three pre-run samples found all four GPUs at 1 MiB, 0% utilization, with no
  compute processes. No matrix worker remained afterward.

The local runtime-data and temporary-output paths are represented below by
environment variables so no machine-specific path is committed:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
export QUAL_DIR="$(mktemp -d /tmp/opendde-seed-parallel-XXXXXX)"
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

INPUT=examples/foldcp_demo_placeholder.json
CHECKPOINT="$OPENDDE_ROOT_DIR/checkpoint/opendde.pt"
COMMON=(
  -n opendde_v1 --load_checkpoint_path "$CHECKPOINT"
  --use_msa false --use_template false --use_rna_msa false
  --sample 1 --step 2 --cycle 1 --dtype fp32
  --trimul_kernel torch --triatt_kernel torch
  --enable_tf32 false --deterministic true
)

# Ordinary serial reference; no topology flags.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m runner.batch_inference pred \
  -i "$INPUT" -o "$QUAL_DIR/serial" --seeds 101,102,103,104,105,106 \
  "${COMMON[@]}"

# D x 1. Set CASE/WORLD/GPUS/SEEDS to the successful DP rows below.
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone \
  --nproc_per_node "$WORLD" -m runner.batch_inference pred \
  -i "$INPUT" -o "$QUAL_DIR/$CASE" --seeds "$SEEDS" "${COMMON[@]}" \
  --foldcp_mode single --foldcp_size_dp "$WORLD" --foldcp_size_cp 1

# 1 x P. Set CASE/WORLD/GPUS to the successful CP rows below.
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone \
  --nproc_per_node "$WORLD" -m runner.batch_inference pred \
  -i "$INPUT" -o "$QUAL_DIR/$CASE" --seeds 101 "${COMMON[@]}" \
  --foldcp_mode distributed --foldcp_size_dp 1 --foldcp_size_cp "$WORLD"

# Matched two-record reference and D2 run.
TWO_INPUT="$QUAL_DIR/two_records.json"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m runner.batch_inference pred \
  -i "$TWO_INPUT" -o "$QUAL_DIR/two_serial" --seeds 101,102,103,104 \
  "${COMMON[@]}"
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone \
  --nproc_per_node 2 -m runner.batch_inference pred \
  -i "$TWO_INPUT" -o "$QUAL_DIR/two_dp2" --seeds 101,102,103,104 \
  "${COMMON[@]}" \
  --foldcp_mode single --foldcp_size_dp 2 --foldcp_size_cp 1

# Rejected hybrid and mismatched-world-size launches.
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone \
  --nproc_per_node 4 -m runner.batch_inference pred \
  -i "$INPUT" -o "$QUAL_DIR/hybrid_guard" --seeds 101,102,103,104 \
  "${COMMON[@]}" \
  --foldcp_mode distributed --foldcp_size_dp 2 --foldcp_size_cp 2
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone \
  --nproc_per_node 2 -m runner.batch_inference pred \
  -i "$INPUT" -o "$QUAL_DIR/world_guard" --seeds 101,102,103,104 \
  "${COMMON[@]}" \
  --foldcp_mode single --foldcp_size_dp 4 --foldcp_size_cp 1
```

The two-record input was a temporary copy of the checked-in 40-residue record
with names `foldcp_demo_first` and `foldcp_demo_second`. It was run serially and
with `D=2` using seeds 101-104. The guard commands used the distributed template
with `(WORLD,D,P)=(4,2,2)` and the seed-parallel template with `(2,4,1)`.
Every command had a 900-second timeout and a 200 ms `nvidia-smi` memory trace.
For each `WORLD=N` row, `GPUS` was `0,...,N-1`; `CASE` and `SEEDS` match the
matrix entries.

### Four-GPU matrix

Peak memory lists selected physical GPUs in local-rank order. Elapsed time is
end to end and is diagnostic only on this heterogeneous host.

| Case | World | `D` | `P` | Seeds | Result | Elapsed (s) | Peak MiB |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| Serial reference | 1 | 1 | 1 | 101-106 | Pass (0) | 21.147 | 5722 |
| Seed parallel D2 | 2 | 2 | 1 | 101-106 | Pass (0) | 20.461 | 5888 / 5888 |
| Seed parallel D3 | 3 | 3 | 1 | 101-106 | Pass (0) | 20.595 | 5876 / 5876 / 5876 |
| Seed parallel D4 | 4 | 4 | 1 | 101-104 | Pass (0) | 19.414 | 5876 / 5876 / 5876 / 6046 |
| Two-record serial | 1 | 1 | 1 | 101-104 | Pass (0) | 21.718 | 5722 |
| Two-record D2 | 2 | 2 | 1 | 101-104 | Pass (0) | 20.091 | 5888 / 5888 |
| Fold-CP P2 | 2 | 1 | 2 | 101 | Pass (0) | 23.085 | 5888 / 5888 |
| Fold-CP P3 | 3 | 1 | 3 | 101 | Pass (0) | 23.569 | 5876 / 5876 / 5876 |
| Fold-CP P4 | 4 | 1 | 4 | 101 | Pass (0) | 24.184 | 5876 / 5876 / 5876 / 6046 |
| Hybrid guard | 4 | 2 | 2 | 101-104 | Expected fail (1) | 4.895 | 1 / 1 / 1 / 1 |
| World-size guard | 2 | 4 | 1 | 101-104 | Expected fail (1) | 4.947 | 1 / 1 |

Both guards failed without `Loading from`, wrote no prediction CIF, and exited
without a timeout or lingering worker. Successful cases produced exactly one
CIF, summary-confidence JSON, and full-data JSON per expected `(job, seed)`.

### Numerical alignment

Every CIF had the same atom-site schema and exact atom identity/order as its
matched serial result (335 atoms), and all CIF/full-data/summary numeric values
were finite. Coordinates and every numeric summary-confidence leaf were checked
with `numpy.testing.assert_allclose(atol=5e-4, rtol=5e-4)`:

| Candidate | Compared `(job, seed)` pairs | Max coordinate abs delta | Max B-factor abs delta | Max summary abs delta |
| --- | ---: | ---: | ---: | ---: |
| D2 | 6 | 0 | 0 | 0 |
| D3 | 6 | 0 | 0 | 0 |
| D4 | 4 | 0.001 | 0.02 | 0.0000991821 |
| Fold-CP P2 | 1 | 0.001 | 0.03 | 0.0003738403 |
| Fold-CP P3 | 1 | 0.001 | 0.03 | 0.0003738403 |
| Fold-CP P4 | 1 | 0.001 | 0.03 | 0.0012588501 |
| Two-record D2 | 8 | 0 | 0 | 0 |

All comparisons passed the stated absolute/relative tolerance. D4's nonzero
deltas were confined to seed 104 on rank 3 (the RTX 4090); seeds 101-103 on the
three RTX 3090 ranks had zero recorded deltas. Fold-CP's small deltas were also
present on homogeneous RTX 3090 P2/P3 runs and are attributable to its changed
parallel reduction order plus CIF decimal serialization. No seed, sample, atom,
or schema divergence occurred.

## Complexity inventory

- New CLI flags or config keys: none.
- New model, kernel, output, or input paths: none.
- New dependencies, services, scripts, or abstractions: none.
- Required code owners changed: existing config, runner, and inference sampler
  only.

## Non-goals

The SPEC non-goals are binding, especially same-GPU seed batching, hybrid
`D x P`, model changes, automatic sizing/retry, and Predictify integration.
