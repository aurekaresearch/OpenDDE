# Fold-CP 1 x P Reproduction Guide

This document describes how to run and validate seed-batched inference,
multi-GPU seed sharding, and arbitrary-`P` Fold-CP. It intentionally does not
publish performance or capacity numbers: those claims require the exact source
commit, input provenance, runtime environment, and raw measurements from the
same run.

## Execution requirements

Let `D=--foldcp_size_dp`, `P=--foldcp_size_cp`, and
`B=--seed_batch_size`:

| Mode | `foldcp_mode` | `D` | `P` | `B` | Required world size |
| --- | --- | ---: | ---: | ---: | ---: |
| Scalar single GPU | `single` | 1 | 1 | 1 | 1 |
| Batched single GPU | `single` | 1 | 1 | >1 | 1 |
| Sharded seed batches | `single` | >1 | 1 | >=1 | `D` |
| Fold-CP | `distributed` | 1 | >1 | 1 | `P` |
| Batched Fold-CP | - | any | >1 | >1 | Rejected |
| Hybrid | - | >1 | >1 | any | Rejected |

- `D=1, P=1, B=1` uses scalar single-process inference.
- `B>1` batches independent seeds on the leading model dimension and requires
  `P=1` and `num_workers=0`.
- Training-Free Guidance requires `B=1`; combining it with `B>1` is rejected.
- Rank `r` receives `seeds[r::D]`, then partitions its ordered slice into
  chunks of at most `B`; a smaller final chunk is valid. This is the same path
  for `D=1` and `D>1`.
- Seeds must be unique when `D>1` or `B>1`. Multi-rank runs require at least
  `D` seeds.
- Fold-CP uses a `1 x P` topology with
  `--foldcp_size_dp 1 --foldcp_size_cp P --seed_batch_size 1`, where `P > 1`.
- `--nproc_per_node` must equal
  `--foldcp_size_dp * --foldcp_size_cp`.
- Hybrid `D>1, P>1` and mismatched-world-size launches are rejected before
  model loading.
- OpenDDE does not automatically choose or reduce `B`, or retry serially after
  an OOM.
- Multi-GPU inference is single-node and requires a shared input/output
  filesystem.
- Multi-GPU Fold-CP uses
  `--trimul_kernel torch --triatt_kernel torch`; cuEquivariance triangle
  kernels are not supported in this mode.
- On CUDA BF16, distributed triangle attention uses Triton 3.3.1 from the GPU
  install extra to fuse attention-bias addition. This is part of the Fold-CP
  PyTorch execution path, not cuEquivariance.
- Deterministic comparisons should disable TF32, enable deterministic mode,
  and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Fold-CP does not impose a CUDA allocator fraction by default.

## Launch commands

Run the single-process reference from the same checkout as the distributed
comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m runner.batch_inference pred \
  -i input.json -o output_p1 -n opendde_v1 \
  --seeds 101,102,103,104 \
  --seed_batch_size 1 \
  --sample 1 --step 2 --cycle 1 --dtype bf16 \
  --use_msa true --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch --enable_tf32 false \
  --deterministic true
```

Run the same four seeds two at a time on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m runner.batch_inference pred \
  -i input.json -o output_b2 -n opendde_v1 \
  --seeds 101,102,103,104 \
  --seed_batch_size 2 \
  --sample 1 --step 2 --cycle 1 --dtype bf16 \
  --use_msa true --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch --enable_tf32 false \
  --deterministic true
```

To run two seeds per GPU on four GPUs, use `D=4, P=1, B=2`:

```bash
D=4
GPU_LIST="$(seq -s, 0 $((D - 1)))"

CUDA_VISIBLE_DEVICES="$GPU_LIST" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node "$D" \
  -m runner.batch_inference pred \
  -i input.json -o "output_d${D}_b2" -n opendde_v1 \
  --seeds 101,102,103,104,105,106,107,108 \
  --seed_batch_size 2 \
  --sample 1 --step 2 --cycle 1 --dtype bf16 \
  --use_msa true --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch --enable_tf32 false \
  --deterministic true \
  --foldcp_mode single --foldcp_size_dp "$D" --foldcp_size_cp 1
```

For `P>1`, launch one process per GPU. This example uses four GPUs; replace
both occurrences of `4` to test another supported process count:

```bash
P=4
GPU_LIST="$(seq -s, 0 $((P - 1)))"

CUDA_VISIBLE_DEVICES="$GPU_LIST" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node "$P" \
  -m runner.batch_inference pred \
  -i input.json -o "output_p${P}" -n opendde_v1 \
  --seeds 101 \
  --seed_batch_size 1 \
  --sample 1 --step 2 --cycle 1 --dtype bf16 \
  --use_msa true --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch --enable_tf32 false \
  --deterministic true \
  --foldcp_mode distributed --foldcp_size_dp 1 --foldcp_size_cp "$P"
```

The commands are a controlled comparison template, not a performance
benchmark. Adjust the model settings for production, but keep every setting
except `D`, `P`, and `B` identical between scalar, batched, and distributed
comparisons.

Validate the execution guards separately. These commands must fail before model
loading:

```bash
# Hybrid D=2, P=2.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i input.json -o output_invalid_hybrid -n opendde_v1 \
  --seeds 101,102,103,104 \
  --foldcp_mode distributed --foldcp_size_dp 2 --foldcp_size_cp 2

# WORLD_SIZE=2 does not match D=4, P=1.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
  -m runner.batch_inference pred \
  -i input.json -o output_invalid_world -n opendde_v1 \
  --seeds 101,102,103,104 \
  --foldcp_mode single --foldcp_size_dp 4 --foldcp_size_cp 1

# Seed batching is not supported with Fold-CP P=2.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
  -m runner.batch_inference pred \
  -i input.json -o output_invalid_batched_foldcp -n opendde_v1 \
  --seeds 101,102 --seed_batch_size 2 \
  --foldcp_mode distributed --foldcp_size_dp 1 --foldcp_size_cp 2

# Training-Free Guidance requires B=1.
CUDA_VISIBLE_DEVICES=0 python -m runner.batch_inference pred \
  -i input.json -o output_invalid_batched_tfg -n opendde_v1 \
  --seeds 101,102 --seed_batch_size 2 --use_tfg_guidance true
```

## Validating numerical alignment

- Use the same Git commit, checkpoint, input JSON, MSA/template features, seed,
  dtype, cycle count, diffusion steps, and kernel settings for every run.
- For every `D x 1, B` run, verify that each requested `(job, seed)` directory
  exists exactly once and no other seed directory was produced.
- Verify output schemas, tensor shapes, atom ordering, and finite values before
  comparing numerical values.
- Compare prediction coordinates and summary-confidence outputs with tolerances
  appropriate for the selected dtype and deterministic settings.
- Compare each seed in a batch against the same seed from the scalar run;
  `N_sample` remains independent of `B`.
- Record observed BF16 deltas: batch-shaped kernels can use different
  floating-point launch geometry, so bitwise identity is not expected even
  though per-seed random streams remain independent.
- Do not use file hashes as the only numerical-alignment check; metadata and
  serialization details can change without changing the prediction.
- Save the outputs for every compared rank count so the result can be audited.

## Publishing performance or capacity results

Record and publish all of the following with any benchmark claim:

- the exact Git commit and OpenDDE version;
- checkpoint identity and input/MSA/template provenance;
- the complete command line and relevant environment variables;
- Python, PyTorch, Triton, CUDA, driver, and container versions;
- GPU model, GPU count, host topology, and allocator settings;
- `D`, `P`, `B`, raw per-rank timing logs, and GPU memory samples;
- whether timing covers model forward only or end-to-end preprocessing and
  inference;
- deterministic settings and the numerical-alignment result.

Measure scalar `B=1`, each batched width, and every `P>1` run from the same
checkout and environment. Report OOM widths as failures, not as automatic
fallbacks. Do not present a single successful run as a supported production
capacity limit, and do not carry performance or capacity conclusions across
code, checkpoint, input, kernel, or hardware changes without remeasurement.
