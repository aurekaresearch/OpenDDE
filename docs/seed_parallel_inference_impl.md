# Seed-Parallel Inference Implementation

Status: Implementation and qualification in progress
Date: 2026-08-26
Source: `docs/seed_parallel_inference_spec.md`

Implement seed batching as a minimal extension of the normal model path. One
rank-local scheduler owns seed assignment and batching for both single- and
multi-GPU execution; multi-GPU execution only gives that scheduler a strided
subset of the seeds.

## Current behavior and gap

The prior seed-parallel implementation assigned `seeds[rank::D]` but still ran
one full model call for every seed. It therefore provided multi-GPU seed
sharding, not same-GPU seed batching.

Using `N_sample` as a substitute is incorrect: samples share the seed's
featurization and trunk state, while independent seeds affect MSA sampling,
reference features, diffusion noise, and other random inputs. True seed
batching requires a separate leading model dimension and independent random
streams.

## Work unit 1: Expose one batch-width control

Add `seed_batch_size: 1` to the inference defaults/schema and expose it as
`--seed_batch_size` on `opendde pred`.

- Treat the value as the maximum rank-local batch width, not a GPU count.
- Preserve `B=1` as the compatibility default.
- Pass the value through existing runner construction; do not add a launcher or
  a second inference entrypoint.
- Validate `B >= 1` before model loading.

## Work unit 2: Unify rank sharding and local batching

Keep seed resolution and distributed input handling in the existing inference
runner:

1. Sort directory inputs on every rank; for multi-rank runs, preprocess each
   input once on rank 0 and broadcast the shared path or failure.
2. Resolve and broadcast the ordered seed list.
3. Assign rank `r` the stable slice `seeds[r::D]`.
4. Chunk that slice in order into batches of at most `B`.
5. Send every chunk, including an uneven tail, through the same prediction
   method.

There must be no separate single-GPU implementation. `D=1` naturally assigns
all seeds to rank 0; `D>1` changes only the input slice to the same chunking
loop.

Preserve one featurization iterator and random state per seed across multiple
input records. Seed-dependent records in a chunk must describe the same job and
topology before they are stacked. Keep `num_workers=0` mandatory for `B>1`, so
the runner can preserve those independent random streams without a second
worker scheduling protocol.

Reject duplicates when `D>1` or `B>1`, fewer seeds than ranks when `D>1`, and
all existing topology mismatches. Do not retry an OOM with a smaller batch.

## Work unit 3: Add the seed dimension to the normal model path

Stack seed-dependent tensors on a leading `B_seed` dimension while verifying
that shared topology metadata agrees. Carry the normal path as:

```text
[B_seed, N_sample, ...]
```

Update only operators that currently assume an unbatched leading shape:

- per-seed MSA row sampling and gather;
- MSA input projection broadcasting;
- template feature broadcasting where templates are shared;
- diffusion random draws;
- confidence-stage handoff before scalar postprocessing.

Keep independent generators for each seed. Preserve featurization and MSA
generator state across input records; diffusion derives an independent rollout
generator from each seed per record. Do not draw one MSA permutation for the
entire batch.

After batched confidence computation, split `B_seed` before the existing
non-batch-safe ranking, structure conversion, and dumper path. Each split item
must retain its original seed, features, atom array, entity metadata, and
`N_sample` predictions.

Do not add the seed dimension to Fold-CP or Training-Free Guidance. Reject
`B>1, P>1`, `B>1` with `model.N_model_seed>1`, and `B>1` with Training-Free
Guidance; retain their existing `B=1` implementations.

## Work unit 4: Add focused automated tests

Extend the closest existing config, runner, MSA, diffusion, and model tests.
Add a focused seed-batch test module only where no existing owner is clear.

Cover:

1. CLI/default/schema propagation for `B=1` and explicit widths.
2. Rank-strided assignment followed by ordered chunks for `D=1-4`, including a
   partial tail.
3. Duplicate, insufficient-seed, `B<1`, `B>1/P>1`, nonzero-worker, and TFG
   guards.
4. Per-lane MSA and diffusion random streams matching equivalent scalar calls.
5. Batched model shapes with `B_seed>1` and `N_sample>1`.
6. Per-seed postprocessing and exactly-once dumping.
7. Unified prediction-call behavior for `D=1` and `D>1`.
8. Existing scalar and Fold-CP regression tests.

Run focused checks first, followed by the repository-prescribed non-network
suite and formatting checks. GPU-only equivalence belongs in the local
qualification matrix, not a network-dependent unit test.

## Work unit 5: Update public documentation

Update only:

- `README.md`;
- `docs/inference_instructions.md`;
- `docs/foldcp_e2e_baseline.md`; and
- `CHANGELOG.md` under `[Unreleased]`.

Use the existing flag spelling and command style. Document the default,
rank-first assignment, partial batches, Fold-CP restriction, worker restriction,
per-seed outputs, and explicit no-retry behavior.

## Local four-GPU qualification

Run from the final implementation commit with all selected GPUs idle. Record
raw end-to-end time and sampled peak memory per GPU, and preserve logs and
outputs until numerical comparisons finish.

The correctness matrix must include:

| Case | World | `D` | `P` | `B` | Expected |
| --- | ---: | ---: | ---: | ---: | --- |
| Scalar reference | 1 | 1 | 1 | 1 | Success |
| Same-GPU batch | 1 | 1 | 1 | 2+ | Success |
| Partial tail | 1 | 1 | 1 | 2 | Success with three seeds |
| Sharded batches | 2 | 2 | 1 | 2 | Success |
| Sharded batches | 3 | 3 | 1 | 2 | Success |
| Sharded batches | 4 | 4 | 1 | 2 | Success |
| Fold-CP | 2-4 | 1 | 2-4 | 1 | Success |
| Batched Fold-CP guard | 2 | 1 | 2 | 2 | Fail before inference |
| Hybrid guard | 4 | 2 | 2 | 1 | Fail before model load |
| World-size guard | 2 | 4 | 1 | 1 | Fail before model load |

For real-input capacity and performance evidence:

- run the five checked-in MSA examples for 7R6R, 5SAK, 7ST3, 7PZB, and 9FM7
  in BF16 on both an RTX 3090 and RTX 4090;
- start at `B=1`, verify `B=2`, and increase `B` until the largest explicit
  width that fits each `(input, GPU)` is established without automatic retry;
- compare each produced seed against its scalar result and report throughput,
  speedup, and peak memory; and
- exercise 7EOW with the ABAG checkpoint across world sizes 1-4, using the same
  rank-local batching path and reporting per-rank memory.

For every success, verify the exact set of seed/sample outputs, CIF schema,
atom identity/order, and finite values. Compare coordinates and
summary-confidence values against the matched scalar result and report the
observed deltas; BF16 batch-shaped kernels are not expected to be bitwise
identical to scalar launches. Record the commit, checkpoint identity, complete
commands, environment, GPU mapping, timing boundary, raw time, and peak memory.
Treat heterogeneous-host timings as local evidence, not portable performance
claims.

Do not commit runtime environments, checkpoints, generated inputs, outputs,
logs, caches, or machine-specific paths.

## Acceptance evidence

Append final commands and results here only after the implementation commit has
passed the automated checks and local matrix. Prior `D x 1` results measured
rank sharding with scalar per-seed model calls and are not evidence for
same-GPU batching.

## Complexity inventory

- New public control: one inference config/CLI integer, `seed_batch_size`.
- New distributed topology: none.
- New output or input schema: none.
- New dependency, service, launcher, or deployment path: none.
- Model changes: only the normal-path leading seed dimension and per-seed RNG
  support required by true batching.

## Non-goals

The SPEC non-goals are binding, especially batched Fold-CP, hybrid `D x P`,
automatic sizing/retry, and Predictify integration.
