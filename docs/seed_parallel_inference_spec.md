# Seed-Parallel Inference Specification

Status: Implementation and qualification in progress
Date: 2026-08-26

This specification adds tensor-batched seed inference on one GPU and composes
it with the existing multi-GPU seed sharding. Single- and multi-GPU execution
use the same rank-local seed-batch path; additional GPUs only change which
seeds each rank owns.

## Goal

Run independent inference seeds concurrently within one model forward while
preserving each seed's features, random stream, samples, and output files.

Reuse the existing Fold-CP topology flags for process placement and add one
orthogonal batch-width control:

- `D` is `--foldcp_size_dp`, the number of seed-sharding ranks;
- `P` is `--foldcp_size_cp`, the number of Fold-CP ranks cooperating on one
  model execution; and
- `B` is `--seed_batch_size`, the maximum number of seeds in one rank-local
  model batch.

`B` defaults to `1`, so existing commands retain their prior memory profile.
It does not change `WORLD_SIZE` or `--sample` (`N_sample`).

## Public interface

The new CLI/config option is:

```text
--seed_batch_size INTEGER  Maximum seeds per rank-local model batch. [default: 1]
```

For example, two seeds can share one GPU:

```bash
opendde pred \
  -i input.json -o output_b2 -n opendde_v1 \
  --seeds 101,102 \
  --seed_batch_size 2
```

The same option composes with multi-GPU seed sharding. This command assigns
two seeds to each of four ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i input.json -o output_d4_b2 -n opendde_v1 \
  --seeds 101,102,103,104,105,106,107,108 \
  --seed_batch_size 2 \
  --foldcp_mode single --foldcp_size_dp 4 --foldcp_size_cp 1
```

## Supported execution modes

`WORLD_SIZE` must equal `D * P`.

| Mode | `D` | `P` | `B` | Behavior |
| --- | ---: | ---: | ---: | --- |
| Scalar single GPU | 1 | 1 | 1 | One seed per model call |
| Batched single GPU | 1 | 1 | >1 | Up to `B` seeds per model call |
| Sharded seed batches | >1 | 1 | >=1 | Rank-local batches after seed sharding |
| Fold-CP | 1 | >1 | 1 | Existing context-parallel inference |
| Batched Fold-CP | any | >1 | >1 | Rejected |
| Hybrid `D x P` | >1 | >1 | any | Rejected |

`--foldcp_mode single` selects the normal model path for both one and multiple
seed-sharding ranks. Same-GPU batching requires `P=1`; this version does not add
a seed batch dimension to Fold-CP.

## Scheduling semantics

Seed precedence remains unchanged: command-line `--seeds`, then the input
job's `modelSeeds`, then one generated seed.

For each preprocessed input:

1. Resolve the ordered seed list once and share it with all ranks.
2. Rank `r` receives `seeds[r::D]`.
3. Each rank partitions that ordered slice into consecutive chunks of at most
   `B` seeds.
4. Every chunk, including a smaller final chunk, follows the same rank-local
   feature, model, postprocessing, and output path.
5. Every requested seed is assigned exactly once. Multiple ranks change only
   ownership, not the batching implementation.

For example, with seeds `101-107`, `D=2`, and `B=2`:

```text
rank 0: [101, 103], [105, 107]
rank 1: [102, 104], [106]
```

Every rank enumerates and consumes input jobs in the same deterministic order.
In a multi-rank run, rank 0 preprocesses each input once and broadcasts the
resulting shared path or failure; single-process preprocessing remains local.
Each rank writes only its assigned seed directories.

## Model and random-stream contract

Seed-dependent features are constructed independently and stacked on a leading
seed dimension. The normal model path carries tensors shaped conceptually as:

```text
[B_seed, N_sample, ...]
```

`B_seed` is the current rank-local chunk size and may be smaller than the
configured `B`. `N_sample` remains the number of diffusion samples within each
seed; seed batching does not reinterpret or replace it.

MSA sampling, reference-position augmentation, and diffusion noise must retain
independent per-seed random streams. Training-Free Guidance with `B>1` is
rejected. Scalar postprocessing and dumping split the leading seed dimension and
continue to receive one seed at a time, preserving existing output schemas and
ranking behavior.

Per-seed random draws must not depend on the other seeds in a batch. End-to-end
predictions are not required to be bitwise identical across batch widths:
batch-shaped kernels can use different floating-point launch geometry, and
small BF16 differences can amplify through iterative diffusion. Qualification
must report those numerical deltas rather than treating them as RNG-stream
mixing.

## Output and compatibility contract

The output layout remains:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

Changing `D` or `B` must not change:

- seed precedence or values;
- `--sample` cardinality within a seed;
- output filenames, schemas, ranking, or atom ordering; or
- the independent random stream associated with a `(job, seed, sample)` tuple.

Completion and log ordering may differ. Logs must identify rank ownership and
the seeds in each batch sufficiently to audit execution.

## Validation and errors

Configuration must fail before model inference when:

- `B < 1`;
- `WORLD_SIZE != D * P`;
- `D > 1` and `P > 1`;
- `B > 1` and `P > 1`;
- `B > 1` and inference data loading uses `num_workers != 0`;
- `B > 1` and `model.N_model_seed != 1`;
- `B > 1` and Training-Free Guidance is enabled;
- `--foldcp_mode distributed` does not have `D=1, P>1`; or
- `--foldcp_mode single` does not have `P=1`.

Seeds must be unique when either `D > 1` or `B > 1`. A multi-rank request must
contain at least `D` seeds. Errors must identify the conflicting values and
must never silently drop, duplicate, or overwrite a seed.

The runner must not catch a CUDA OOM and silently reduce `B`, retry seeds one at
a time, or select a batch size automatically. Users choose a width that fits
their input and device.

## Documentation requirements

The implementation PR updates the existing public owners:

- `README.md` with concise single- and multi-GPU examples;
- `docs/inference_instructions.md` with `D`, `P`, and `B` semantics;
- `docs/foldcp_e2e_baseline.md` with reproducible comparison and guard commands;
  and
- `CHANGELOG.md` under `[Unreleased]`.

Do not add a second run guide or rename the Fold-CP guide.

## Test requirements

Automated tests must cover:

- `B` validation and CLI/config propagation;
- exact rank-to-seed assignment followed by batching, including uneven tails;
- supported and rejected `D`, `P`, `B`, and world-size combinations;
- independent, persistent per-seed random streams;
- batched feature and model tensor shapes with `N_sample > 1`;
- scalar-versus-batched random-stream and small-model-stage equivalence for
  individual seeds;
- per-seed postprocessing and exactly-once output ownership;
- unchanged `B=1` single- and multi-rank behavior; and
- unchanged existing `1 x P`, `B=1` Fold-CP behavior.

Qualification must exercise world sizes 1, 2, 3, and 4, scalar and batched
single-GPU execution, batched multi-GPU seed sharding, a partial final batch,
Fold-CP with `B=1`, and every new guard. Successful comparisons must record the
exact commit, command, GPU mapping, dtype, timing, peak GPU memory, output
cardinality, and numerical deltas from matching scalar runs.

## Non-goals

- Seed batching combined with Fold-CP (`P > 1`).
- Hybrid `D x P` execution with both dimensions greater than one.
- Automatic batch sizing, OOM retries, or performance policy.
- Changes to checkpoint or input/output schemas.
- Predictify, Cloud runner, container, or deployment changes.
- Multi-node execution or filesystems not shared by all ranks.

## Acceptance criteria

- Single- and multi-GPU seed execution share one rank-local batch path.
- Every requested `(job, seed)` runs and is written exactly once.
- Per-seed random streams remain aligned when placed in a batch; BF16
  end-to-end deltas are recorded explicitly.
- `N_sample` remains independent of seed batch width.
- Existing `B=1` and Fold-CP behavior remain compatible.
- Focused tests and the non-network suite pass.
- The local one-to-four-GPU matrix exits without hangs and records timing,
  memory, output, and numerical-alignment evidence.
- Documentation and CLI help describe the implemented behavior consistently.
