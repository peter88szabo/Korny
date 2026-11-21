"""Merge two finite-feature approximations into one conservative Korny readout.

Use a fixed 50/50 average of energy predictions. The two models share the same
frozen potential and atom-centered kernels. Concatenating their bond kernels
preserves the average analytically, including Cartesian derivatives, and avoids
computing the shared backbone or the coordination map twice.
"""

import json
import pickle
import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_fit import private_score
from src.bond_environment import BondBasis
from src.forcefield import ForceField

COMPONENTS = ("p8-public", "p10-public")
OUTPUT = "p11-public"


def model_and_scores(target, geoms):
    paths = [kt.WORK / f"{s}-{target}.npz" for s in COMPONENTS]
    # The linear combination is valid only if all parent and atomic basis data match.
    changes = {"rx_rb_weights", "rx_rb_bond_phases", "rx_rb_bond_frequencies"}
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        assert a.files and set(a.files) == set(b.files)
        for key in set(a.files) - changes:
            np.testing.assert_array_equal(a[key], b[key])
    components = [ForceField.load(p, dispersion="d3") for p in paths]
    prediction = ForceField.load(paths[0], dispersion="d3")
    merged = prediction.reactive_basis
    parts = [c.reactive_basis for c in components]
    new = BondBasis(
        np.concatenate([t.bond.frequencies for t in parts], axis=1),
        np.concatenate([t.bond.phases for t in parts], axis=1),
    )
    common = merged.size - merged.bond.size
    # Each Fourier bank is normalized by sqrt(2 / width). Undo this change in
    # normalization when using both banks in a single larger design matrix.
    bond_weights = (
        np.column_stack(
            [
                np.sqrt(new.width / t.bond.width)
                * t.weights[common:].reshape(t.bond.phases.shape)
                for t in parts
            ]
        )
        * 0.5
    )
    merged.weights = np.r_[
        0.5 * (parts[0].weights[:common] + parts[1].weights[:common]),
        bond_weights.ravel(),
    ]
    merged.size = len(merged.weights)
    merged.bond = new
    out = kt.WORK / f"{OUTPUT}-{target}.npz"
    prediction.save_with_base(kt.original.BASE, out)
    model = ForceField.load(out, dispersion="d3")
    for g in geoms:
        val, force = model.predict(g.z, g.r, *g.state)
        original = [m.predict(g.z, g.r, *g.state) for m in components]
        np.testing.assert_allclose(
            val, np.mean([o[0] for o in original]), atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            force, np.mean([o[1] for o in original], axis=0), atol=1e-9, rtol=0
        )
    data = []
    for s in COMPONENTS:
        key = json.loads((kt.WORK / f"{s}-settings.json").read_text())["selected"]
        with np.load(kt.WORK / f"{key}-{target}-equations.npz") as f:
            data.append({k: f[k] for k in f.files})
    a, b = data
    for k in ("source", "kind", "case"):
        np.testing.assert_array_equal(a[k], b[k])
    err = (a["error_kcal"] + b["error_kcal"]) * 0.5
    np.savez_compressed(
        kt.WORK / f"{OUTPUT}-{target}-equations.npz", **dict(a, error_kcal=err)
    )
    results = kt.stats(
        (None, None, a["source"], None, a["kind"], a["case"]),
        err / kt.KCAL,
        np.ones(len(err), bool),
    )
    result = dict(validation=results)
    if target == "high" and (kt.WORK / "leuven-data.pkl").exists():
        with (kt.WORK / "leuven-data.pkl").open("rb") as f:
            xyz, _, paths = pickle.load(f)
        values = np.zeros(len(xyz))
        for index in {i for _, _, ids, _ in paths for i in ids}:
            z, r, q, m = xyz[index]
            values[index] = model.energy(z, r, q, m)
        result["leuven"] = private_score(values, np.zeros_like(values), paths)
    return result


def run():
    gs, _, _ = kt.inputs()
    rng = np.random.default_rng(418)
    geoms = [gs[k] for k in rng.choice(list(gs), 8, replace=False)]
    result = {t: model_and_scores(t, geoms) for t in kt.TARGETS}
    kt.write_json(kt.WORK / f"{OUTPUT}.json", result)
    kt.write_json(
        kt.WORK / f"{OUTPUT}-settings.json",
        dict(selected=OUTPUT, components=COMPONENTS, weights=[0.5, 0.5]),
    )
    print(
        "Verified merged energy and forces on eight structures for both targets",
        flush=True,
    )


if __name__ == "__main__":
    run()
