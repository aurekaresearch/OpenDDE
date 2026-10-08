"""Report whether predicted structures satisfy the `constraint` of an OpenDDE input JSON.

    python examples/tfg/check_constraints.py INPUT.json PREDICTIONS_DIR [--tolerance ANGSTROM]

PREDICTIONS_DIR is the folder with the `*_sample_<n>.cif` files of one seed, for example
`out/contact/1a14_contact/seed_101/predictions`.

constraint.contact: the distance of every pair is printed; a pair is satisfied when it lies in
    [min_distance, max_distance], widened by --tolerance (default 0) on both sides.
constraint.epitope: an epitope residue is reached when one of its heavy atoms is within 5 A of a heavy atom
    of a movable chain; the request is met when at least K = max(1, ceil(min_fraction * n)) residues are reached.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

GATE = 5.0  # same distance the guidance uses to call an epitope residue reached


def chain_of(job, entity, copy=1):
    """Chain id of the `copy`-th copy of the 1-based `entity` of the job."""
    item = next(iter(job["sequences"][int(entity) - 1].values()))
    ids = item["id"] if isinstance(item["id"], list) else [item["id"]]
    return ids[int(copy) - 1]


def atom_xyz(structure, chain, position, atom_name):
    mask = (
        (structure.chain_id == chain)
        & (structure.res_id == int(position))
        & (structure.atom_name == atom_name)
    )
    if not mask.any():
        raise ValueError(f"no atom {atom_name} at {chain}{position}")
    return structure.coord[mask][0]


def check_contacts(job, structure, tolerance=0.0):
    rows, ok = [], True
    for pair in job["constraint"]["contact"]:
        a = chain_of(job, pair["entity1"], pair.get("copy1", 1))
        b = chain_of(job, pair["entity2"], pair.get("copy2", 1))
        pa, pb = pair["position1"], pair["position2"]
        xa = atom_xyz(structure, a, pa, pair.get("atom1", "CA"))
        xb = atom_xyz(structure, b, pb, pair.get("atom2", "CA"))
        d = float(np.linalg.norm(xa - xb))
        lo, hi = pair.get("min_distance", 0.0), pair.get("max_distance", 8.0)
        met = lo - tolerance <= d <= hi + tolerance
        ok &= met
        rows.append(
            f"{a}{pa}-{b}{pb} {d:5.2f} A [{lo}, {hi}] {'ok' if met else 'VIOLATED'}"
        )
    return ok, rows


def check_epitope(job, structure, tolerance=0.0):
    request = job["constraint"]["epitope"]
    movable = np.isin(structure.chain_id, job["constraint"]["movable_chains"])
    residues = request["residues"]
    k = max(1, math.ceil(request.get("min_fraction", 0.3) * len(residues)))
    rows, reached = [], 0
    for r in residues:
        chain = chain_of(job, r["entity"], r.get("copy", 1))
        mask = (structure.chain_id == chain) & (structure.res_id == int(r["position"]))
        d = np.linalg.norm(
            structure.coord[mask][:, None] - structure.coord[movable][None], axis=-1
        ).min()
        hit = d <= GATE
        reached += hit
        rows.append(f"{chain}{r['position']} {d:5.2f} A {'reached' if hit else '-'}")
    rows.append(f"reached {reached} of {len(residues)}, required {k}")
    return reached >= k, rows


def main(job_path, prediction_dir, tolerance=0.0):
    job = json.loads(Path(job_path).read_text())[0]
    constraint = job.get("constraint") or {}
    kind = (
        "contact"
        if "contact" in constraint
        else "epitope"
        if "epitope" in constraint
        else None
    )
    if kind is None:
        sys.exit("The input has no constraint.contact or constraint.epitope.")
    check = check_contacts if kind == "contact" else check_epitope
    files = sorted(
        Path(prediction_dir).glob("*_sample_*.cif"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not files:
        sys.exit(f"No *_sample_<n>.cif in {prediction_dir}.")
    met = 0
    for path in files:
        structure = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
        ok, rows = check(job, structure, tolerance)
        met += ok
        print(f"{path.name}: {'SATISFIED' if ok else 'not satisfied'}")
        for row in rows:
            print("   ", row)
    print(f"{met} of {len(files)} samples satisfy the {kind} constraint")


if __name__ == "__main__":
    args = sys.argv[1:]
    tolerance = 0.0
    if "--tolerance" in args:
        i = args.index("--tolerance")
        tolerance = float(args[i + 1])
        del args[i : i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    main(args[0], args[1], tolerance)
