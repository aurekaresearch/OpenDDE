# Seed-Parallel Inference Implementation

Status: Implemented and locally qualified
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

Implementation commit: `b0a5ddb`.

Automated qualification passed with `335 passed, 4 deselected, 44 subtests
passed` for the non-network suite. `ruff check .`, `ruff format --check .`, and
`git diff --check` also passed. The focused model-stage test executes a real
small `DiffusionModule` and `ConfidenceHead` with `B=2, N_sample=2` and compares
both seed lanes with scalar calls.

Local GPU qualification used BF16, deterministic algorithms, TF32 disabled,
MSA enabled, one recycle, two diffusion steps, and one sample unless stated
otherwise. End-to-end time includes process startup, model load, featurization,
model execution, and output writing. Peak memory was sampled every 200 ms with
`nvidia-smi`. The released checkpoint was 2,625,249,069 bytes with SHA-256
`7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`.

The scalar time in this table runs the same number of seeds sequentially with
`B=1`. `Max B` succeeded and the next explicit width OOMed; no run retried at a
smaller width.

| GPU | MSA case (`N_token/N_atom/N_msa`) | Max B | Next | B=1 time (s) | Batched time (s) | Speedup | B=1 / batched peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3090 24 GiB | 7R6R (`245/2529/703`) | 10 | 11 OOM | 48.382 | 42.725 | 1.132x | 6,138 / 23,208 |
| RTX 3090 24 GiB | 5SAK (`437/3074/9667`) | 3 | 4 OOM | 51.854 | 49.732 | 1.043x | 12,414 / 21,408 |
| RTX 3090 24 GiB | 7ST3 (`545/4281/12707`) | 2 | 3 OOM | 59.024 | 55.422 | 1.065x | 17,530 / 23,630 |
| RTX 3090 24 GiB | 7PZB single copy (`300/2611/12424`) | 7 | 8 OOM | 55.550 | 54.588 | 1.018x | 7,470 / 23,432 |
| RTX 3090 24 GiB | 9FM7 single copy (`322/2186/8856`) | 6 | 7 OOM | 48.137 | 53.557 | 0.899x | 7,722 / 23,748 |
| RTX 4090 48 GiB | 7R6R (`245/2529/703`) | 21 | 22 OOM | 66.965 | 57.812 | 1.158x | 6,290 / 47,884 |
| RTX 4090 48 GiB | 5SAK (`437/3074/9667`) | 7 | 8 OOM | 63.147 | 60.595 | 1.042x | 12,566 / 46,866 |
| RTX 4090 48 GiB | 7ST3 (`545/4281/12707`) | 4 | 5 OOM | 59.662 | 71.813 | 0.831x | 17,682 / 46,320 |
| RTX 4090 48 GiB | 7PZB single copy (`300/2611/12424`) | 13 | 14 OOM | 66.031 | 58.972 | 1.120x | 7,622 / 45,992 |
| RTX 4090 48 GiB | 9FM7 single copy (`322/2186/8856`) | 14 | 15 OOM | 73.820 | 66.703 | 1.107x | 7,874 / 46,950 |

The checked-in 7PZB and 9FM7 examples describe doubled assemblies. Their real
single-copy protein/nucleic-acid or ligand assemblies were used so the required
`B=2` control fits on a 24 GiB GPU; both retain their checked-in MSA features.
The full doubled assemblies explicitly OOMed at `B=2` on the RTX 3090.

A longer 7R6R control used ten recycles and 200 diffusion steps for the same two
seeds:

| GPU | B=1 / B=2 time (s) | Speedup | B=1 / B=2 peak MiB | B=1 / B=2 seeds/min |
| --- | ---: | ---: | ---: | ---: |
| RTX 3090 | 69.839 / 64.607 | 1.081x | 6,138 / 9,116 | 1.718 / 1.857 |
| RTX 4090 | 58.003 / 50.534 | 1.148x | 6,290 / 9,268 | 2.069 / 2.375 |

The unified sharding-plus-batching path produced exactly eight seed outputs for
`D=4, P=1, B=2` and the expected rank-strided assignments. The same 7R6R test
completed in 24.675 s for `D=2` (four seeds), 25.215 s for `D=3` (six seeds),
and 25.750 s for `D=4` (eight seeds), at approximately 9.2 GiB per rank. A
single-GPU `B=2` run with three seeds also completed its `[2, 1]` partial tail.

The 7EOW antibody-antigen case used `N_sample=2` and the ABAG checkpoint,
2,625,271,509 bytes with SHA-256
`5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d`.
Four fixed seeds with `B=1` completed in 45.887, 33.958, 34.351, and 27.091 s
for `D=1,2,3,4`, respectively; `D=4` peaked at 10,106/10,106/10,106/10,274
MiB. On one RTX 4090, two seeds in one true batch completed in 30.032 s at
44,934 MiB versus 29.691 s at 24,982 MiB for sequential `B=1`; `B=2` OOMed on
the RTX 3090. This large case is memory-bound and does not benefit from local
batching.

Existing Fold-CP remained scalar in the seed dimension: `P=2,3,4`, `B=1`
completed the small regression in 23.695, 24.283, and 24.804 s with one output
and approximately 5.8-6.0 GiB per rank. The batched Fold-CP, hybrid topology,
world-size, worker, nested-model-seed, and TFG guards all failed before model
selection or loading as specified.

Every successful comparison had the exact requested seed/sample cardinality,
matching CIF schema and atom identity/order, and finite CIF and JSON values.
Per-lane MSA permutations and diffusion random draws matched scalar streams in
focused tests. End-to-end BF16 predictions were not bitwise equivalent: across
the five maximum-width cases, maximum absolute coordinate deltas versus scalar
runs ranged from 3.59 to 37.98 Angstrom on the RTX 4090 and 3.63 to 32.86
Angstrom on the RTX 3090; maximum summary-confidence deltas ranged from 0.061 to
0.663 and 0.074 to 0.671, respectively. These observed deltas are recorded, not
accepted as a numerical tolerance; maximum-capacity widths are likewise not a
throughput recommendation.

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
