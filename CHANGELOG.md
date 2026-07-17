# Changelog

User-facing changes to OpenDDE are documented here.

## [Unreleased]

No changes yet.

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

[Unreleased]: https://github.com/aurekaresearch/OpenDDE/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.2
[1.0.1]: https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.0.1
[1.0.0]: https://pypi.org/project/opendde/1.0.0/
