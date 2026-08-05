# Kernel Configuration


OpenDDE has a safe PyTorch path and optional GPU kernels.

## LayerNorm

Default:

```bash
export LAYERNORM_TYPE=torch
```

Optional CUDA LayerNorm:

```bash
export LAYERNORM_TYPE=fast_layernorm
```

Use `torch` for CPU, debugging, or environments where CUDA extension compilation
is unavailable.

## Triangle kernels

Both triangle attention and triangle multiplication support:

| Value | Meaning |
| --- | --- |
| `auto` | Use cuEquivariance when available, otherwise PyTorch. |
| `cuequivariance` | Force cuEquivariance GPU kernels. |
| `torch` | Select OpenDDE's PyTorch triangle implementation. Distributed CUDA BF16 triangle attention also uses Triton for attention-bias fusion. |

CLI flags:

```bash
opendde pred \
  --triatt_kernel auto \
  --trimul_kernel auto
```

### Multi-GPU Fold-CP limitation

The current official cuEquivariance release does not support OpenDDE's
distributed Fold-CP path. Multi-GPU inference must therefore select the
PyTorch triangle implementations:

```bash
--triatt_kernel torch --trimul_kernel torch
```

On CUDA BF16, distributed triangle attention uses Triton 3.3.1 from the GPU
install extra to fuse the two attention-bias additions. Triton is part of this
Fold-CP PyTorch path; it does not enable cuEquivariance. This limitation applies
to multi-GPU Fold-CP only. Single-GPU inference may continue to use `auto` or
`cuequivariance`.

## Compatibility run

```bash
LAYERNORM_TYPE=torch opendde pred \
  -i examples/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --triatt_kernel torch \
  --trimul_kernel torch \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --dtype fp32
```
