# Changelog

User-facing changes to OpenDDE are documented here.

## [Unreleased]

No changes yet.

## [1.2.0] - 2026-10-09

### Added

- Added constraint-guided inference. With `--use_tfg_guidance true`, the input
  JSON `constraint` field steers the sampler towards partial knowledge of an
  antibody–antigen interface, without changing the weights or the trunk:
    - `constraint.contact`: residue-pair distance windows (default 3.5–8 Å);
    - `constraint.movable_chains`: the antibody chains that move as one rigid body;
    - `constraint.epitope`: antigen residues that the movable chains must bind
      (`paratope`, default `"all"`; `min_fraction`, default 0.5).

    The movable chains are moved as one rigid body until the request is met. A
    sample that already meets it is never moved, and a move that creates a severe
    atomic overlap is refused. The evaluated schedule (an early pass at steps
    100–187 and a late pass at steps 190–199) is the default, and
    `OPENDDE_RIGID_CONTACT=off` keeps the atom-level restraint for contacts. See
    `docs/tfg_constraint_guidance.md` and `docs/infer_json_format.md`.

- Added fused Triton kernels for the guidance (`OPENDDE_RIGID_CORE`,
  `OPENDDE_VINA_FAST`). They are used when a CUDA device and Triton are
  available and TF32 is off (the default); otherwise the dense torch
  implementation runs instead, without a warning.
- Added an optional on-disk cache of the Pairformer trunk output
  (`OPENDDE_TRUNK_CACHE`), keyed by every non-guidance input feature, so inputs
  that differ only in their `constraint` reuse one trunk.
  `OPENDDE_TRUNK_CACHE_MODE` selects `r`, `w` or `rw` (default). The cache is
  ignored under Fold-CP and with several model seeds, a file that cannot be read
  or written is skipped with a warning, and its files are Python pickles: use a
  directory you trust.
- Added `examples/tfg`: four antibody–antigen complexes (Fab, Fv and VHH), each with
  unconstrained, contact and pocket inputs and the MSAs they need, and a script
  that checks which predicted samples satisfy a constraint.
- Added `benchmarks/OpenBench2026` with the target and interface lists of the
  antibody–antigen panel.

### Changed

- Inference now defaults to `--dtype bf16` and `--enable_tf32 false`; earlier
  versions defaulted to `fp32` with TF32 on. Pass `--dtype fp32 --enable_tf32 true`
  for the previous behaviour. Predictions can differ numerically at the new
  precision. On Apple MPS, BF16 needs macOS 14 or newer and is downgraded to FP32
  below that.
- The input JSON `constraint` field is no longer ignored. Earlier versions
  logged a warning and ignored the whole field; it now accepts only `contact`,
  `movable_chains` and `epitope`, and any other key (including a misspelled one),
  a `constraint` that is not an object, a `contact` that is not a list, or a
  non-numeric or negative contact distance raises an error.
- Moved the co-folding examples into `examples/cofold` (`input.json`, `protein_200.json`,
  `example.json`, `example_without_msa.json`, `template/`, `rna_msa/`, `msa/`,
  `structures/`) and removed unused example files. Update paths such as
  `examples/input.json` to `examples/cofold/input.json`.

### Compatibility

- Model checkpoints and output formats are unchanged; the default precision is
  the one change that can alter predictions. Inputs without a
  `constraint` field are not affected. Inputs that carry a `constraint` field
  with other keys, which earlier versions ignored, must remove those keys.
- The guidance has been validated on a single GPU with the default
  200 diffusion steps.

## [1.1.1] - 2026-09-02

### Added

- Added Apple Silicon MPS inference through `--device mps`; `--device auto`
  selects CUDA, then MPS, then CPU. MPS uses the PyTorch triangle kernels and
  defaults to FP32, while Fold-CP remains CUDA-only.
- `InferenceRunner` now supports context-manager use and keeps runner/model
  configuration updates synchronized.

### Changed

- Reduced inference memory use through bounded dynamic chunking, lower-peak
  relative-position materialization, and CPU retention of multi-sample and
  multi-seed outputs. Prediction values and output formats are unchanged.
- Hardened multi-input and multi-seed inference with per-job seed schedules,
  earlier input validation, reliable failure reporting, atomic prediction
  directories, and preprocessing outputs that work with read-only inputs.
- Improved MSA/template file routing and Fold-CP failure coordination. Fold-CP
  now uses the supported `1 x P` topology and safely handles inputs smaller
  than the launched GPU count.
- Restored process-wide determinism and TF32 settings after inference so an
  embedded OpenDDE runner does not alter its host application's PyTorch state.

### Fixed

- Rebuilds invalid OXT coordinates at free protein C termini before CIF output,
  while preserving externally bonded termini and skipping repair when the
  local C/CA/O anchor geometry is invalid.
- Fixed CUDA cleanup, Fold-CP confidence output placement, output-directory
  permissions, repeated CCD cache eviction, and caller-data mutation during
  inference and serialization.

### Compatibility

- Model checkpoints and input/output formats remain compatible with OpenDDE
  1.1.0.
- Fold-CP supports `1 x P`; `OPENDDE_FOLDCP_CUDA_MEMORY_FRACTION` is no longer
  used.

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

[Unreleased]: https://github.com/aurekaresearch/OpenDDE/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.2.0
[1.1.1]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.1.1
[1.1.0]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.1.0
[1.0.3]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.3
[1.0.2]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.2
[1.0.1]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.1
[1.0.0]: https://pypi.org/project/opendde/1.0.0/
