# Changelog

User-facing changes to OpenDDE are documented here.

## [Unreleased]

### Added

- Apple silicon (Metal/MPS) inference. `--device mps` runs the model on the
  Metal backend, and `--device auto` now picks CUDA first, then MPS, then CPU.
  MPS uses PyTorch triangle kernels and defaults to FP32; `--dtype bf16` opts
  into the same dynamic mixed-precision policy CUDA uses. The trunk uses BF16;
  by default, diffusion and the confidence head stay FP32 through 2560 tokens,
  while confidence uses BF16 above 2560 and diffusion above 3840 to reduce
  memory. Regions the model pins to FP32 now clear the autocast state of every
  accelerator instead of CUDA's alone, since MPS keeps its own. Per-seed memory
  cleanup releases the Metal allocator cache, and `opendde doctor` reports the
  MPS build/availability flags. Distributed Fold-CP stays CUDA-only. Attention
  explicitly broadcasts mismatched Q/K/V batch dimensions and folds
  higher-rank inputs before calling the layout-sensitive Metal scaled-dot-product
  kernel, and chain pLDDT accumulates in FP32 there because Metal has no FP64.

### Changed

- Lowered the memory peak of lazily materialized relative position encodings,
  used by Fold-CP to build its pair tiles. The one-hot sections are now filled
  into a single pre-allocated buffer instead of being concatenated, which
  avoids holding the intermediate tensors, the joined result, and its float
  copy at the same time. Predictions are unchanged: the filled buffer is
  bit-for-bit identical to the concatenated one.

### Fixed

- Best-effort CUDA cleanup after a failed batch no longer raises a second error,
  and a failed synchronize no longer prevents the allocator from releasing its
  cached blocks. Normal seed-boundary synchronization still surfaces
  asynchronous CUDA failures instead of reporting success.

## [1.1.0] - 2026-08-16

### Added

- Accelerated multi-GPU Fold-CP `1 x P` diffusion by caching query-owned
  attention biases and atom-window state across denoising steps, avoiding
  repeated pair-bias projection and communication.

  Controlled cache-on/cache-off benchmarks on NVIDIA A100(80GB) GPUs with
  200 diffusion steps measured the following speedups across validated 2-, 4-,
  and 8-GPU Fold-CP configurations:

  | Timed region | Speedup |
  | --- | ---: |
  | Full forward | **1.64×–1.85×** |
  | Diffusion sampling | **2.27×–2.68×** |

  These timings cover model execution, excluding external MSA generation, input
  preprocessing, and result serialization. Exact gains vary with input size and
  GPU topology; all same-topology cache-on/cache-off output comparisons were
  bitwise identical.

- Added a configurable resident-memory budget for the Fold-CP diffusion cache.
  The budget defaults to 16 GiB per Fold-CP rank (one rank per GPU) and is not
  preallocated. The actual cache size depends on the input and GPU count. If the
  estimated resident storage exceeds the budget, OpenDDE warns and automatically
  falls back to per-step projection. Set
  `OPENDDE_FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES` to a byte value to override
  the default.

### Fixed

- Released inference batches before synchronized CUDA cleanup at seed
  boundaries, preventing device tensors from remaining live across multi-seed
  inference.
- Kept the Fold-CP triangle-multiplication operand bound after host offload,
  preventing an `UnboundLocalError` in the offloaded `1 x P` path.
- Ensured fully padded `1 x P` ranks still participate in diffusion collectives
  and made fused attention-bias projection handle zero-width shards safely.

### Compatibility

- The new query-owned cache is enabled automatically for multi-GPU Fold-CP under
  the existing shared-variable cache setting, which is on by default for
  inference. Normal single-GPU, non-Fold-CP inference never constructs this
  cache and continues to use its existing diffusion path.
- Model checkpoints and input/output formats remain compatible with OpenDDE
  1.0.x.

## [1.0.3] - 2026-08-05

### Added

- Extended multi-GPU Fold-CP inference to a `1 x P` context-parallel topology
  for arbitrary launched world sizes `P > 1`, including non-square GPU counts.

### Fixed

- Preserved ion entities in MSA and template feature metadata, preventing
  missing or shifted asymmetric-chain mappings for ion-containing inputs.
  Thanks to [@MoritzErtelt](https://github.com/MoritzErtelt) for contributing
  this fix in [#18](https://github.com/aurekaresearch/OpenDDE/pull/18).
- Replaced square-mesh assumptions in Fold-CP pair, atom, confidence, distogram,
  MSA, and trunk paths with the current `1 x P` layouts.
- Removed the legacy `distributed_outer_product_mean` helper, whose local pair
  layout was incompatible with `1 x P`; the runtime uses the integrated
  Pairformer OPM path.
- Aligned Fold-CP documentation with the CUDA BF16 execution path and replaced
  unverifiable benchmark and capacity claims with a reproducible validation
  procedure.

### Compatibility

- Multi-GPU Fold-CP requires
  `--triatt_kernel torch --trimul_kernel torch`; cuEquivariance triangle
  kernels remain unsupported in this mode. CUDA BF16 Fold-CP triangle attention
  additionally uses Triton 3.3.1 for attention-bias fusion.

## [1.0.2] - 2026-07-17

### Fixed

- Updated the Linux x86_64 GPU dependencies to Triton 3.3.1 and
  cuEquivariance 0.10.0, resolving the `PY_SSIZE_T_CLEAN` failure seen during
  accelerated inference on larger protein systems.

## [1.0.1] - 2026-07-16

### Added

- Explicit CPU/CUDA device selection and the `opendde doctor` environment
  report.
- Verified, release-pinned downloads for checkpoints and runtime assets.
- Simplified CPU and CUDA 12.6 installation, with optional Linux x86_64
  cuEquivariance acceleration through `opendde[gpu]`.
- Validated four-GPU Fold-CP inference using native PyTorch triangle kernels.

### Fixed

- Improved CPU/PyTorch fallbacks, CUDA compatibility checks, runtime downloads,
  and Kalign discovery.
- Preserved explicit checkpoint paths and safely repaired incomplete managed
  assets.

### Compatibility

- CPython 3.11, 3.12, and 3.13 are supported. Linux x86_64 is the primary GPU
  platform; Apple Silicon is CPU-only, and Windows is not currently validated.

For installation and upgrade commands, see the
[OpenDDE 1.0.1 release notes](https://github.com/aurekaresearch/OpenDDE/blob/v1.0.1/docs/releases/1.0.1.md).

## [1.0.0] - 2026-07-15

- Initial PyPI bootstrap release of the `opendde` package name.

[Unreleased]: https://github.com/aurekaresearch/OpenDDE/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.1.0
[1.0.3]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.3
[1.0.2]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.2
[1.0.1]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.1
[1.0.0]: https://pypi.org/project/opendde/1.0.0/
