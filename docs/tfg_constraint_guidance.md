# TFG Constraint Guidance

`constraint` fields in the input JSON (see
[`infer_json_format.md`](infer_json_format.md)) steer inference through
Training-Free Guidance (TFG). They work with any checkpoint, need
`--use_tfg_guidance true`, and change only the sampled coordinates; the trunk
and the weights are untouched.

## Two ways to apply `constraint.contact`

| Mode | When it runs | What it does |
| --- | --- | --- |
| Rigid-body guidance (default) | `--use_tfg_guidance true`, and `movable_chains` is given or the complex has exactly two chains | The `movable_chains` move as one rigid body until the contacts are met; no atom is displaced on its own. A move that adds a severe clash or does not lower the energy is refused. The atom restraint is switched off. |
| Atom restraint | `OPENDDE_RIGID_CONTACT=off`, or a contact request without `movable_chains` on a complex with more than two chains | `UserDistanceRestraintPotential` pulls the listed atom pairs into their `[min_distance, max_distance]` window during sampling. Each atom moves independently, and a pair ends at the edge of its window. |

`OPENDDE_RIGID_CONTACT` is `auto` by default: rigid-body guidance runs when TFG is
enabled and the input carries an epitope request or contact pairs it can apply,
and an input without a constraint is not touched. `on` forces it (the run stops
with an error if there is nothing to apply or TFG is off), `off` disables it, and
`control` keeps the TFG machinery but applies neither the atom restraint nor
rigid guidance; it is the matched control for an experiment.

`constraint.epitope` is applied only by rigid-body guidance: with
`OPENDDE_RIGID_CONTACT=off` or `control`, or without `--use_tfg_guidance true`,
the run stops with an error.

## How rigid-body guidance runs

Both modes minimise one energy, `0.5 * ||violations||^2 + 10 * clash`, over rigid
motions of the moving chains. A sample whose request is already met is never
moved.

- **Refinement** is gradient descent on that energy for at most 40 iterations
  per call (at most 120 at the last diffusion step). Each iteration reduces the
  atomic gradient to one translation and one rotation about the centroid, tries
  ten step sizes (1, 1/2, ..., 1/512) at once, and accepts the first that lowers
  the energy without creating a severe overlap. The call ends early when no
  sample accepts a step.
- **Coarse search** scores a fixed set of candidate poses and keeps the lowest
  energy one without a severe overlap if it improves on the current pose:
  1,500 poses for a contact request, 4,440 for an epitope on one chain (2,880 on
  several chains).
- **Early pass** (`OPENDDE_RIGID_X0_START`): at the scheduled steps the
  denoiser's clean estimate is refined, searched for the samples that still miss
  the request, and refined again; the sampler continues from the corrected
  estimate.
- **Late pass** (`OPENDDE_RIGID_START`, `OPENDDE_RIGID_EVERY`): on the sampled
  state, a coarse search at step `OPENDDE_RIGID_START` and at the last step, and
  a refinement at every step from `OPENDDE_RIGID_START` on that is a multiple of
  `OPENDDE_RIGID_EVERY`, and at the last step.

## Default settings

The defaults are the settings of the evaluated runs: single GPU (no Fold-CP),
`--use_tfg_guidance true`, `--dtype bf16`, 10 recycles, the default 200
diffusion steps and 5 samples per seed, with TF32 off. The step numbers below
assume 200 diffusion steps, and no environment variable has to be set:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENDDE_RIGID_CONTACT` | `auto` | see above |
| `OPENDDE_RIGID_START`, `OPENDDE_RIGID_EVERY` | `190`, `2` | late pass: coarse search at step 190 and the last step, refinement at every step from 190 that is a multiple of 2 and at the last step (with fewer than 191 steps the start is the last step; an explicit value must lie inside the run) |
| `OPENDDE_RIGID_X0_START`, `OPENDDE_RIGID_X0_EVERY`, `OPENDDE_RIGID_X0_LAST` | `100`, `3`, `189` | early pass on the denoiser's clean estimate: steps 100, 103, ..., 187 (steps beyond the run are skipped, so a run of 100 steps or fewer has no early pass) |
| `OPENDDE_RIGID_CORE`, `OPENDDE_VINA_FAST` | `auto` | accelerated kernels, see below |

`OPENDDE_RIGID_X0_START=off` turns the early pass off and `OPENDDE_RIGID_LATE=off`
turns the late pass off. Worked examples with commands, inputs and a result
checker are in [`examples/tfg/`](../examples/tfg/README.md).

## Kernel switches

The fast kernels are Triton kernels and need a CUDA GPU. Every switch below
keeps a dense implementation that stays available.

| Variable | Values (default first) | Meaning |
| --- | --- | --- |
| `OPENDDE_RIGID_CORE` | `auto`, `on`, `off`, `check` | The accelerated core shared by both modes. `auto` uses it when a CUDA device and Triton are available and the dense implementation otherwise, without a warning; `on` does the same but warns when the core cannot be used and falls back to the dense implementation when the core raises; `check` runs both, returns the dense result and logs the parity (`EPITOPE_GUIDANCE core parity`, `RIGID_CONTACT core parity`). |
| `OPENDDE_VINA_FAST` | `auto`, `on`, `off`, `check` | Fused kernel for `VinaStericPotential`. `auto` uses it on a CUDA device with Triton; `on` tries it on any CUDA device; both fall back to the dense implementation, with a warning if the kernel fails, and use it silently on CPU or MPS. `off` always uses the dense implementation, and `check` compares with the dense result and returns the dense one. |

The kernels reproduce the summation order of `torch.cdist`; when the installed
PyTorch orders it differently, guidance uses the dense implementation and logs a
warning. The accelerated core also needs TF32 matmul off, which is the default
(`--enable_tf32 false`); with TF32 on it is skipped and the dense implementation
runs. The first skip of a run is logged once.

## Other switches

- `OPENDDE_TRUNK_CACHE=<dir>` with `OPENDDE_TRUNK_CACHE_MODE=r|w|rw` (default
  `rw`) stores the Pairformer trunk output on disk. The key is a hash of every
  input feature that is not a guidance feature, plus the number of recycles, so
  inputs that differ only in their `constraint` reuse one trunk. It is bypassed
  under Fold-CP and when several model seeds run (`N_model_seed > 1`). A cache
  file that cannot be read or written is skipped with a warning. Cache files
  are Python pickles: use a directory you trust.

## Not covered

Fold-CP with more than one rank per input and step counts other than 200 have
not been validated for these modes. Without Triton or CUDA, guidance runs the
dense implementation: it was run end to end on CPU for a small complex (see
[`examples/tfg`](../examples/tfg/README.md#without-triton-on-cpu-or-mps)) and is
slow on large ones. Apple MPS has not been tested.
