# Seed-Parallel Inference Specification

Status: Implemented
Date: 2026-08-25

This specification adds seed-level data parallelism to the existing inference
runner without changing model computation, seed meaning, output files, or the
current `1 x P` Fold-CP path.

## Goal

Allow explicit inference seeds to run concurrently on separate GPUs under
`torchrun`. Reuse the existing `--foldcp_size_dp` and `--foldcp_size_cp`
topology flags rather than adding another parallelism interface.

This version supports single-node `torchrun --standalone`; all ranks share the
same checkout, runtime assets, input paths, and output filesystem.

The default `D=1, P=1` execution must remain the existing serial path, where:

- `D` is `--foldcp_size_dp`, the number of independent seed lanes; and
- `P` is `--foldcp_size_cp`, the number of Fold-CP ranks used by one seed.

## Public interface

No new CLI flag, config key, input field, or output field is added.

Seed-parallel inference uses the existing single-card model path with one
process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i input.json -o output_dp4 -n opendde_v1 \
  --seeds 101,102,103,104 \
  --use_msa false --use_template false --use_rna_msa false \
  --foldcp_mode single --foldcp_size_dp 4 --foldcp_size_cp 1
```

`--foldcp_mode single` means that each seed lane uses the normal single-card
model path. It does not require `WORLD_SIZE=1` when `D > 1`.

## Supported topologies

`WORLD_SIZE` must equal `D * P`.

| Mode | `D` | `P` | Behavior |
| --- | ---: | ---: | --- |
| Serial | 1 | 1 | Existing single-process inference |
| Seed parallel | >1 | 1 | One independent seed lane per GPU |
| Fold-CP | 1 | >1 | Existing `1 x P` context-parallel inference |
| Hybrid | >1 | >1 | Rejected in this version |

Hybrid seed parallelism plus Fold-CP is out of scope because current model and
output ownership are global-rank-oriented. Supporting it would require changes
inside the Fold-CP model/process-mesh path and would make this PR materially
larger.

## Execution semantics

Seed precedence remains unchanged: command-line `--seeds`, then the input
job's `modelSeeds`, then one generated seed.

For seed-parallel `D x 1` execution:

1. For each preprocessed input JSON, rank 0 resolves and validates the ordered
   seed list once and shares it with all ranks. Existing precedence and the
   first top-level record's `modelSeeds` behavior remain unchanged.
2. Seeds must be unique and their count must be at least `D`; invalid requests
   fail before model inference.
3. Rank `r` receives `seeds[r::D]`. Every requested seed is assigned exactly
   once and order is stable within each lane.
4. Every rank consumes every input job in the same deterministic order. Input
   records are not sharded by `DistributedSampler`.
5. Shared input preprocessing runs once on rank 0 and shares the resulting path
   or failure with all ranks. A preprocessing or seed-validation failure
   terminates every rank and returns a nonzero command result; existing
   per-job prediction-error handling remains unchanged.
6. Each rank retains the existing per-seed sequence: seed global RNG state,
   featurize, run the model, rank samples, and dump results.
7. Each rank writes only its assigned seed directories.

For existing `1 x P` Fold-CP execution, all ranks continue to consume the same
jobs and seeds, and global rank 0 remains the only output writer. No model,
kernel, checkpoint, or numerical path changes are required for seed parallelism.

Directory inputs must be sorted before distributed iteration so every rank
processes files in the same order.

## Output and determinism contract

The output layout remains:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

Seed-parallel width is an execution setting only. It must not change:

- seed precedence or seed values;
- `--sample` cardinality within each seed;
- output filenames, schemas, ranking, or atom ordering; or
- the result associated with a `(job, seed, sample)` tuple.

Completion and log ordering may differ because lanes finish independently.
The implementation must log the global rank and assigned seeds sufficiently to
audit ownership. It must not claim a performance gain without a controlled
benchmark.

## Validation and errors

Configuration must fail before model loading when:

- `WORLD_SIZE != D * P`;
- `D > 1` and `P > 1`;
- `--foldcp_mode distributed` does not have `D=1, P>1`;
- `--foldcp_mode single` does not have `P=1`.

After each input's seeds are resolved, but before its first model forward, the
command must fail when seed-parallel execution receives duplicate seeds or
fewer seeds than lanes.

Errors must identify the conflicting values and the supported topology. A
request must never silently fall back to serial execution or silently drop,
duplicate, or overwrite a seed.

## Documentation requirements

The implementation PR must update only the existing public owners:

- `README.md` with one concise seed-parallel example;
- `docs/inference_instructions.md` with topology and flag semantics;
- `docs/foldcp_e2e_baseline.md` with reproducible `D x 1`, `1 x P`, and invalid
  hybrid validation commands; and
- `CHANGELOG.md` under `[Unreleased]`.

Do not add a second run guide or rename the Fold-CP guide.

## Test requirements

Automated tests must cover:

- supported and rejected topology validation;
- exact world-size validation before model loading;
- rank-to-seed assignment, including uneven tails;
- rank-0 seed resolution and preprocessing propagation;
- multi-record inputs appearing in the same order on every rank;
- exactly-once `(job, seed)` execution and output ownership;
- unchanged serial `D=1, P=1` behavior; and
- unchanged existing `1 x P` Fold-CP behavior.

The implementation must also be qualified locally on the available four-GPU
host using world sizes 1, 2, 3, and 4. The matrix must include serial,
seed-parallel, Fold-CP, mismatched-world-size, and rejected-hybrid cases. Use the
same small checked-in input, released checkpoint, seeds, dtype, kernels, cycle,
step, and sample settings for matched comparisons.

For successful runs, verify seed directories, sample counts, CIF schema, atom
ordering, finite values, coordinates, and summary-confidence outputs against
the matching serial result with an explicit dtype-appropriate tolerance. Record
the exact commit, commands, environment, GPU mapping, raw timing, peak memory,
and numerical deltas. The heterogeneous four-GPU host provides correctness
evidence only, not a portable speed claim.

## Non-goals

- Same-GPU tensor batching of multiple seeds.
- Hybrid `D x P` execution with both dimensions greater than one.
- Automatic batch sizing, OOM retries, scheduling, or performance policy.
- Changes to `N_sample`, checkpoints, kernels, model code, output schemas, or
  input JSON.
- Predictify, Cloud runner, container, or deployment changes.
- Multi-node execution or filesystems that are not shared by all local ranks.

## Acceptance criteria

- The supported topology table is enforced exactly.
- Every requested `(job, seed)` runs and is written exactly once.
- Serial and existing Fold-CP behavior remain compatible.
- Focused tests and the non-network test suite pass.
- The complete local world-size 1-4 GPU matrix exits without hangs and records
  numerical-alignment evidence.
- Documentation and CLI help describe the implemented behavior consistently.
- The implementation diff contains no unrelated refactor or new abstraction.
