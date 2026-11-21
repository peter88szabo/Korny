"""Measure native radical saddle curvature along the first two opposing IRC steps.

The reference and model use exactly the same force secant in mass-weighted
coordinates. This measures one local direction at an archived geometry.
An independently optimized saddle and a full Hessian would test other modes.
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.native_irc_pes import NativeRadicalPotential
from experiments.full_public.trial_irc_transfer import IRC
from experiments.reactivity.frequency_validation import TO_CM1
from experiments.reactivity.stationary_fit import MASSES

STAGES = (61, 63, 64)
# Use the most abundant isotope, consistent with the light atoms in MASSES.
MASS_U = dict(MASSES.values()) | {16: 31.972071}


def aligned(r, reference, mass):
    """Remove rigid motion using a mass-weighted proper rotation onto the saddle."""
    centered = r - np.average(r, axis=0, weights=mass)
    u, _, vt = np.linalg.svd((mass[:, None] * centered).T @ reference)
    u[:, -1] *= np.linalg.det(u @ vt)
    rotation = u @ vt
    return centered @ rotation, rotation


def path_chord(rx):
    """Align both first steps and their forces with the same saddle and atom order."""
    coords, forces = [], []
    for side in ("forw", "rev"):
        with np.load(IRC / f"{rx}-{side}.npz") as f:
            z = f["z"]
            ts, r = f["r_A"][:2].astype(float)
            force = f["force_eV_A"][1].astype(float)
        if side == "forw":
            atoms = z
            mass = np.array([MASS_U[int(n)] for n in z])
            root_mass = np.sqrt(mass)[:, None]
            reference = ts - np.average(ts, axis=0, weights=mass)
        else:
            np.testing.assert_array_equal(z, atoms)
            other_ts, _ = aligned(ts, reference, mass)
            if np.sqrt(np.mean(np.sum((other_ts - reference) ** 2, axis=1))) > 0.002:
                raise ValueError(f"Different saddle geometries in {rx}")
        pos, rotation = aligned(r, reference, mass)
        coords.append(pos)
        forces.append(force @ rotation)
    steps = np.array([(r - reference) * root_mass for r in coords]).reshape(2, -1)
    lengths = np.linalg.norm(steps, axis=1)
    if min(lengths) < 1e-6:
        return {"id": rx, "excluded": "degenerate displacement"}, None
    cosine = float((steps[0] @ steps[1]) / np.prod(lengths))
    if cosine > -0.9:
        return {"id": rx, "excluded": "branches do not oppose", "cosine": cosine}, None
    chord = steps[1] - steps[0]
    length = np.linalg.norm(chord)
    # dot(F, displacement) is the mass-weighted force projected on the chord.
    displacement = (chord / length).reshape(-1, 3) / root_mass
    metadata = {"id": rx, "cosine": cosine, "step_lengths_A_sqrt_u": lengths.tolist()}
    return metadata, (z, coords, forces, displacement, length)


def signed_wavenumber(curvature):
    return np.sign(curvature) * np.sqrt(np.abs(curvature)) * TO_CM1


def curvature_case(rx):
    metadata, data = path_chord(rx)
    if data is None:
        return metadata
    z, coords, forces, displacement, length = data

    def measure(f0, f1):
        return float(signed_wavenumber(-np.sum((f1 - f0) * displacement) / length))

    values = {"reference": measure(*forces)}
    for stage, potential in models.items():
        values[str(stage)] = measure(*(potential.predict(z, r)[1] for r in coords))
    return metadata | {"signed_cm1": values}


def run(stages=STAGES):
    global models
    models = {}
    for s in stages:
        root = IRC.parent / f"multiscale-path{s}"
        ids = [r["rxn"] for r in json.loads((root / "snapshot.json").read_text()) if r["heldout"]]
        if models and ids != held:
            raise ValueError("Native model validation IDs differ")
        held = ids
        models[s] = NativeRadicalPotential.load(root / "native-radical-pes.npz")
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(curvature_case, held, chunksize=4, buffersize=6))
    data = [r["signed_cm1"] for r in rows if "signed_cm1" in r]
    reference = np.array([r["reference"] for r in data])
    summary = {}
    for s in stages:
        pred = np.array([r[str(s)] for r in data])
        summary[str(s)] = {
            "n": len(pred),
            "signed_mae_cm1": float(np.mean(abs(pred - reference))),
            "reference_downhill": int(np.sum(reference < 0)),
            "correct_downhill": int(np.sum((reference < 0) & (pred < 0))),
        }
    kt.write_json(root / "held-path-curvature.json", {"summary": summary, "rows": rows})
    print("Held radical saddle force-secant validation", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(*STAGES, 66, 67))
    stage = parser.parse_args().stage
    run((stage,) if stage else STAGES)
