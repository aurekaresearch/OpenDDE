# Constraint-guided inference (TFG): worked examples

OpenDDE can use partial knowledge of an antibody–antigen interface at inference
time, without retraining. You describe the knowledge in a `constraint` field of
the input JSON; Training-Free Guidance (TFG) then moves the antibody as a rigid
body, inside the sampler, until the constraint is met (see
[`docs/tfg_constraint_guidance.md`](../../docs/tfg_constraint_guidance.md) for
how it works and [`docs/infer_json_format.md`](../../docs/infer_json_format.md)
for every field).

Two kinds of knowledge are supported:

| Kind | You know | JSON field |
| --- | --- | --- |
| **Contact** | residue pairs, antibody residue – antigen residue, that touch | `constraint.contact` |
| **Pocket** | antigen residues that the antibody binds, with no antibody side | `constraint.epitope` |

Both use `constraint.movable_chains`: the antibody chains that move together as
one rigid body (heavy and light chain of a Fab, or the single chain of a VHH).
Every other chain stays where the sampler puts it. A pocket constraint always
needs it; a contact constraint needs it unless the complex has exactly two
chains.

## The cases

The cases are complexes for which the plain prediction is wrong, because that is
where a constraint is worth giving. Each is a folder named after its PDB entry
with the inputs `<pdb>_unconstrained.json` (the baseline), `<pdb>_contact.json`
(four residue pairs) and `<pdb>_pocket.json` (four antigen residues). They differ
only in the `constraint` field. The `msa/` folder of each case holds its MSAs.

| PDB | Antibody | Antigen |
| --- | --- | --- |
| [`1a14`](1a14) | Fv: `H`, `L` | `N` |
| [`9lh2`](9lh2) | VHH: `C`, `D` | `A`, `B` |
| [`9sat`](9sat) | Fab: `A`, `B` | `C` |
| [`9xqn`](9xqn) | Fab: `B`, `C` | `A` |

The constraints are read off the deposited structure, so they are correct; with
your own data they come from your experiments, see
[Constraint quality](#constraint-quality). The four contact pairs are the closest
antibody CDR–antigen Cα–Cα pairs (3.5 to 7 Å, every residue used once) with the
window 3.5–8 Å. The four pocket residues are the antigen residues that touch the
most antibody heavy atoms (within 4.5 Å); the whole antibody is the paratope and
half of the residues must be reached (`min_fraction: 0.5`). In `9lh2` the VHH `C`
is the movable chain and its second copy `D` stays where the sampler puts it.

[`check_constraints.py`](check_constraints.py) reports which predicted samples
satisfy the constraint of an input.

## Requirements

- The OpenDDE package and its model checkpoint (see the main README). The
  antibody–antigen checkpoint used for the evaluation is `opendde_abag.pt`; pass
  it with `--load_checkpoint_path` if you have it.
- A CUDA GPU with Triton for practical speed. Without them guidance still runs,
  but much more slowly; see [Without Triton, on CPU or MPS](#without-triton-on-cpu-or-mps).
- Run every command from the repository root. The inputs point to their MSAs
  (`<pdb>/msa/`) through relative `pairedMsaPath` / `unpairedMsaPath` entries, so
  `--use_msa true` starts no MSA search. OpenDDE searches only for a path that
  does not exist, which is what happens when the working directory is wrong.
- Keep the MSA. Constraints need a good enough structure to be met: in a run of
  `1a14` without an MSA the antigen was predicted wrongly and none of five samples
  satisfied the contact request.

## Configuration

### Environment variables

The environment variables of the guidance already default to the settings of the
evaluated runs, so none has to be set. The step numbers assume the default 200
diffusion steps.

| Variable | Default | Values | What it does |
| --- | --- | --- | --- |
| `OPENDDE_RIGID_CONTACT` | `auto` | `auto`, `on`, `off`, `control` | Rigid-body guidance. `auto` runs it when TFG is on and the input carries a constraint it can apply (an epitope, or contact pairs with `movable_chains` or on a two-chain complex). `on` forces it and stops with an error if there is nothing to apply. `off` keeps the atom-level restraint for contacts and refuses an epitope. `control` runs TFG without applying any constraint. |
| `OPENDDE_RIGID_X0_START` | `100` | step number, `off` | First step of the early pass on the denoiser's clean estimate; `off` turns the early pass off. |
| `OPENDDE_RIGID_X0_EVERY` | `3` | integer ≥ 1 | The early pass runs every this many steps. |
| `OPENDDE_RIGID_X0_LAST` | `189` | step number | Last step of the early pass (steps 100, 103, ..., 187 with the defaults). |
| `OPENDDE_RIGID_LATE` | `on` | `on`, `off` | `off` turns the late pass off. |
| `OPENDDE_RIGID_START` | `190` | step number | First step of the late pass on the sampled state; the coarse search runs here and at the last step. Scaled down for runs with fewer than 191 steps; an explicit value must lie inside the run. |
| `OPENDDE_RIGID_EVERY` | `2` | integer ≥ 1 | The late pass refines every this many steps from `OPENDDE_RIGID_START`, and at the last step. |
| `OPENDDE_RIGID_CORE` | `auto` | `auto`, `on`, `off`, `check` | Fast Triton kernels of the guidance. `auto` uses them when a CUDA device and Triton are available and the dense implementation otherwise, without a warning; `on` warns when it cannot; `off` always uses the dense implementation; `check` runs both and logs the difference. |
| `OPENDDE_VINA_FAST` | `auto` | `auto`, `on`, `off`, `check` | Fast kernel for the steric term of the TFG engine; same values as above. |

The defaults written out, which is the same as setting nothing:

```bash
export OPENDDE_RIGID_CONTACT=auto
export OPENDDE_RIGID_X0_START=100
export OPENDDE_RIGID_X0_EVERY=3
export OPENDDE_RIGID_X0_LAST=189
export OPENDDE_RIGID_LATE=on
export OPENDDE_RIGID_START=190
export OPENDDE_RIGID_EVERY=2
export OPENDDE_RIGID_CORE=auto
export OPENDDE_VINA_FAST=auto
```

What the defaults do: rigid-body guidance for contact and pocket constraints, an
early pass at steps 100, 103, ..., 187 and a late pass at steps 190, 192, ..., 198
and the last step. To change one part, set only its variable, for example
`OPENDDE_RIGID_X0_START=off` for the late pass alone, or
`OPENDDE_RIGID_CONTACT=off` for the atom-level restraint on contacts.
[`docs/tfg_constraint_guidance.md`](../../docs/tfg_constraint_guidance.md) explains
each pass.

### Command-line flags

| Flag | Why |
| --- | --- |
| `--use_tfg_guidance true` | the constraint is applied only in the TFG sampler |
| `--enable_tf32 false` | the contact kernels need exact fp32 matrix products; with TF32 on, they are skipped and the slow dense implementation runs |
| `--dtype bf16` | the evaluated setting (`fp32` also works) |
| `--sample 5 --seeds 101,102,103` | several candidates per input; keep the one that satisfies the constraint best |

The guided run is the same inference with a few extra steps per denoising
iteration: on one A800, `1a14` (612 tokens, 5 samples) took 148 s unguided and
172 s with contact guidance.

## Run the examples

```bash
OUT=./tfg_demo
FLAGS="--seeds 101 --sample 5 --dtype bf16 --use_msa true --use_template false"

# 1. baseline: no constraint, standard sampler
opendde pred -i examples/tfg/1a14/1a14_unconstrained.json -o $OUT/unconstrained $FLAGS \
  --use_tfg_guidance false

# 2. contact constraint
opendde pred -i examples/tfg/1a14/1a14_contact.json -o $OUT/contact $FLAGS \
  --use_tfg_guidance true --enable_tf32 false

# 3. pocket constraint
opendde pred -i examples/tfg/1a14/1a14_pocket.json -o $OUT/pocket $FLAGS \
  --use_tfg_guidance true --enable_tf32 false
```

Structures are written to `<out>/<job name>/seed_<seed>/predictions/`. The other
cases run the same way; for example the pocket constraint of the Fab `9sat`:

```bash
opendde pred -i examples/tfg/9sat/9sat_pocket.json -o $OUT/9sat $FLAGS \
  --use_tfg_guidance true --enable_tf32 false
```

### Check the result and choose a candidate

```bash
python examples/tfg/check_constraints.py examples/tfg/1a14/1a14_contact.json \
  $OUT/contact/1a14_contact/seed_101/predictions
python examples/tfg/check_constraints.py examples/tfg/9sat/9sat_pocket.json \
  $OUT/9sat/9sat_pocket/seed_101/predictions
```

The script prints, per sample, the distance of every contact pair, or the
distance of every epitope residue to the antibody and how many are reached, and
how many samples satisfy the constraint. Run it on the baseline as well to see
whether the unguided prediction already agrees with your knowledge. With a pocket
constraint, pick the candidate that reaches the most epitope residues (ties by
`ranking_score`).

## Reading the constraint

**Contact** (`1a14/1a14_contact.json`):

```json
"constraint": {
  "contact": [
    {"entity1": "3", "copy1": 1, "position1": "248", "atom1": "CA",
     "entity2": "2", "copy2": 1, "position2": "93",  "atom2": "CA",
     "min_distance": 3.5, "max_distance": 8.0}
  ],
  "movable_chains": ["H", "L"]
}
```

`entity` is the 1-based index of an item of `sequences` (here 1 = `H`, 2 = `L`,
3 = `N`), `position` the 1-based residue position in that sequence, and the
distance window is in Å. Every pair must join one atom of a movable chain to one
atom outside the movable chains.

**Pocket** (`1a14/1a14_pocket.json`):

```json
"constraint": {
  "movable_chains": ["H", "L"],
  "epitope": {
    "residues": [{"entity": "3", "copy": 1, "position": "248", "residue": "P"}],
    "paratope": "all",
    "min_fraction": 0.5
  }
}
```

An epitope residue counts as reached when one of its heavy atoms is within 5 Å
of a heavy atom of a movable chain. The request is met when at least
`ceil(min_fraction * n)` of the `n` residues are reached, so `min_fraction`
decides how many of your residues may stay untouched. `paratope: "all"` lets
every heavy atom of the movable chains count as binding face; `"cdr"` restricts
it to numbered CDR windows. `residue` is optional and checked against the
sequence, which catches position typos.

## Constraint quality

- **Contact constraints are strict.** Every pair has the same weight, so a
  wrong pair can pull the antibody away from the true interface. Give only pairs
  you trust, and check the result with `check_constraints.py`.
- **Pocket constraints tolerate some error** when `min_fraction` is below 1:
  only the closest `K` residues are enforced, so a few wrong residues need not be
  touched. With `min_fraction: 1.0` every residue must be touched and one wrong
  residue spoils the result.
- A contact and an epitope request cannot be combined in one job; use two jobs.
- Guidance never moves a sample that already satisfies the request, and it
  refuses a move that creates a severe atomic overlap. When no overlap-free pose
  satisfies the request (for instance because the constraint is wrong), a sample
  can stay unsatisfied; `check_constraints.py` reports it.
- The atom-level alternative for `constraint.contact` (`OPENDDE_RIGID_CONTACT=off`)
  pulls each pair to the edge of its window and stops there, so a distance can
  end at 8.00 Å; count such samples with `check_constraints.py --tolerance 0.1`.

## Running many variants of one input

The constraint enters only the diffusion sampler, not the trunk network. When
you run an input several times with different constraints, let the trunk be
computed once and reused:

```bash
export OPENDDE_TRUNK_CACHE=$OUT/trunk_cache   # first run writes it, later runs read it
```

Inputs that differ only in their `constraint` (and the unconstrained run) share
one cache entry. The cache holds Python pickles: use a directory you trust. It is
ignored under Fold-CP.

## Without Triton, on CPU or MPS

The Triton kernels are an optional speed-up. On a machine without Triton or
without a CUDA GPU, leave `OPENDDE_RIGID_CORE` and `OPENDDE_VINA_FAST` at their
default (`auto`); guidance then runs the dense torch implementation that the
kernels are checked against (largest coordinate difference in the check runs:
4e-4 Å), without a warning. Importing OpenDDE needs no Triton, and the unit tests
run on CPU. Select the device with `--device cpu` (or `--device auto`).

- **CPU, no Triton, no CUDA** (tested): a 135-residue nanobody–peptide complex
  (a sequence-only input, not one of these cases) with rigid-body guidance, one
  sample, `--dtype fp32`, `--use_msa false`, finished in 310 s on a 128-core host
  and satisfied its four contact pairs. A pocket request on the same complex
  finished in 241 s; there the request was already met by the unguided structure,
  so the dense pocket moves ran only in the unit tests, not in this run.
- **Speed is the limit.** The dense coarse search is the slow part: on an A800
  one call for `1a14` (612 tokens) took about 79 s, against 0.03 s with the
  kernel. A complex of that size is not practical on a CPU; a nanobody with a
  peptide or a small antigen is.
- **Atom restraint** for `constraint.contact` (`OPENDDE_RIGID_CONTACT=off`) is
  plain torch and has no kernel to leave out.
- **Apple MPS** (not tested): the pocket guidance computes some small rotations
  in float64, which the MPS backend does not provide, so expect `constraint.epitope`
  to fail there; contact guidance may need `PYTORCH_ENABLE_MPS_FALLBACK=1` for
  linear-algebra operators MPS lacks. `OPENDDE_RIGID_CORE` and `OPENDDE_VINA_FAST`
  must not be forced `on`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `constraint.epitope is applied only by rigid-body guidance` | `OPENDDE_RIGID_CONTACT` is set to `off` or `control`; unset it |
| `constraint.epitope is applied only through TFG guidance` | add `--use_tfg_guidance true` |
| `Unsupported constraint field(s): ...` | only `contact`, `movable_chains` and `epitope` are accepted |
| `RIGID_CONTACT core not used (...)` or `dense fallback` in the log | the fast kernels were skipped: no CUDA device, no Triton, or TF32 enabled; the run is correct but slow |
| the log has no `x0 hook` lines | `OPENDDE_RIGID_X0_START=off` is set, so only the late pass runs, or the run has fewer than 100 diffusion steps |

