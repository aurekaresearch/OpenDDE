# AGENTS.md — OpenDDE Agent Guide


This file applies to the whole repository unless a deeper `AGENTS.md` overrides it.

## Core rules

- Start by checking `git status --short`; do not overwrite user changes.
- Make the smallest task-focused change. Do not refactor, rename, or format unrelated files.
- Prefer targeted tests. If GPU, network, checkpoints, databases, or heavy dependencies are unavailable, say exactly what was not validated.
- Keep public interfaces compatible. CLI flags, config keys, JSON fields, model names, output layouts, docs, and tests must change together.
- Never commit secrets, private paths, checkpoints, databases, caches, generated outputs, or large artifacts.

## Repository map

- `opendde/`: main Python package.
  - `model/`: model, diffusion, pairformer, confidence, layer norm, kernel dispatch.
  - `data/`: parsers, CCD/RDKit/Biotite handling, tokenizer, features, MSA/template/RNA-MSA utilities.
  - `config/`: config parser, defaults, registries, dependency URLs.
  - `utils/`, `metrics/`: shared utilities and metrics.
- `runner/`: Click CLI and inference runner (`opendde = runner.cli:opendde_cli`).
- `tests/`: pytest/unittest suite; `tests/smoke/` is heavier and often GPU/environment dependent.
- `docs/`: user docs. Prefer these for behavior details: inference, Docker, kernels, JSON input, supported models, MSA/template pipeline.
- `examples/`: small example inputs and structures.

## Environment and commands

- Python requirement: `>=3.11,<3.14`; CI tests Python `3.11`, `3.12`, and `3.13` on Ubuntu.
- CPU/dev setup:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cpu -e .
uv pip install --group dev
```

- Common checks:

```bash
ruff check .
pre-commit run --all-files
python -m pytest tests -q -m "not network"
```

- Run the smallest relevant test first, then broaden. Network tests are excluded in CI with `-m "not network"`.
- GPU/kernel changes may also need `tests/test_triton_compatibility.py` and `tests/smoke/test_cueq_runtime.py` on a CUDA machine.
- Pre-commit uses Ruff `--fix` and `ruff-format`; only format files relevant to the task unless asked otherwise.

## CLI, config, and inference

- Public CLI commands are registered across `runner/cli.py` (`doctor`) and `runner/batch_inference.py` (`pred`, `json`, `msa`, `mt`, `prep`).
- Config parsing lives in `opendde/config/config.py`; dotted CLI keys look like `--model.N_cycle 4` and `--sample_diffusion.N_step 20`.
- Put model defaults in `opendde/config/model_base.py`, model names/overrides in `model_registry.py`, data/cache roots in `data.py`, and inference defaults in `inference_defaults.py`.
- `ListValue` CLI args are comma-separated, e.g. `--seeds 101,102`; docs/examples should use lowercase `true`/`false` for bools.
- Currently supported model: `opendde_v1`.
- Inference assets belong under `$OPENDDE_ROOT_DIR`; checkpoints are expected under `$OPENDDE_ROOT_DIR/checkpoint/`.
- CPU-safe smoke inference should use `LAYERNORM_TYPE=torch`, disable external features, and force torch triangle kernels.
- Do not re-enable extra `_wounresol.cif` output by default; structure saving should emit the normal prediction CIF unless explicitly requested.

## Code conventions

- Use 4-space indentation, `snake_case` for functions/variables/modules, `PascalCase` for classes, and nearby tensor naming such as `N_atom`, `N_token`, `c_z`.
- Preserve dtype/device/autocast semantics, CPU fallbacks, and import-time soft failures for optional CUDA/GPU code.
- For tensor changes, check shape, mask, broadcast, NaN/Inf, and deterministic behavior with small tests when possible.
- Use existing logging patterns such as `opendde.utils.logger.get_logger`; do not leave debug `print`s outside script-style code.
- Avoid new heavy dependencies unless required and declared in `pyproject.toml`.

## Data and kernel caveats

- Input JSON details live in `docs/infer_json_format.md`; top level is a job list with entities such as `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, `ion`, plus `covalent_bonds`.
- MSA/template/RNA-MSA preprocessing may need network access, large databases, HMMER/Kalign binaries, and Docker/GPU-specific setup. Do not assume they exist locally.
- CCD/ligand/SMILES/SDF/MOL/PDB paths depend on RDKit, Biotite, Gemmi, and PDBe CCD data; cover residues, modifications, ions, ligands, and covalent-bond edge cases when changing them.
- Triangle kernels are `auto`, `cuequivariance`, or `torch`; always preserve PyTorch fallback.
- `LAYERNORM_TYPE=torch` is the safe default. `fast_layernorm` is optional CUDA/JIT behavior and must fail softly.

## Handoff checklist

- `git diff` contains only task-related changes.
- Relevant tests/docs/examples/config/CLI help were updated together.
- Validation commands and results are listed in the final response; skipped validation has a concrete reason.
- No secrets, private paths, checkpoints, databases, generated outputs, logs, or large artifacts were introduced.
