"""Identify published educt/product IRC endpoint complexes and validate the deployed PES.

An assignment requires an atom-by-atom match of both endpoints to archived XYZ
structures after a rigid rotation and translation. Evaluate the two directed
electronic barriers to those complexes. Each Gaussian path is in the gas
phase and its zero-point/thermal corrections are excluded.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public.rdb19_irc import OUT
from src.forcefield import ForceField


def rms(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    u[:, -1] *= np.linalg.det(u @ vt)
    return np.sqrt(np.mean(np.sum((a @ u @ vt - b) ** 2, axis=1)))


def load(model_names):
    global models
    models = [
        ForceField.load(kt.WORK / f"{n}-public-wb97xd3.npz", dispersion="d3")
        for n in model_names
    ]


def score(row):
    rx = row["rxn"]
    ends = []
    branches = []
    for k in ("IRC_educt", "IRC_product"):
        path = OUT / "source-endpoints" / f"{rx}-{k}.npz"
        if not path.is_file():
            return {"rxn": rx, "status": "missing endpoint XYZ"}
        with np.load(path) as f:
            ends.append((f["z"], f["r_A"]))
    for k in ("forw", "rev"):
        with np.load(OUT / f"{rx}-{k}.npz") as f:
            z = f["z"]
            q, m = int(f["charge"]), int(f["multiplicity"])
            xyz = f["r_A"][[0, -1]].astype(float)
            branches.append((xyz[1], f["energy_eV"][-1]))
            ts = xyz[0], f["energy_eV"][0]
    if any(not np.array_equal(z, a) for a, _ in ends):
        return {"rxn": rx, "status": "endpoint atom order differs"}
    dist = np.array([[rms(r, b[0]) for b in branches] for _, r in ends])
    forward = np.argmin(dist, axis=1)
    if len(set(forward)) != 2 or dist[np.arange(2), forward].max() > 0.002:
        return {"rxn": rx, "status": "ambiguous assignment", "rms_A": dist.tolist()}
    xyz = [branches[forward[0]][0], ts[0], branches[forward[1]][0]]
    ref = np.array([branches[forward[0]][1], ts[1], branches[forward[1]][1]])

    def observables(e):
        return (
            np.array([e[2] - e[0], e[1] - (e[2] + e[0]) / 2, e[1] - e[0], e[1] - e[2]])
            * kt.KCAL
        )

    vals = [
        observables(np.array([model.energy(z, r, q, m) for r in xyz]) - ref).tolist()
        for model in models
    ]
    return {
        "rxn": rx,
        "family": row["family"],
        "status": "assigned",
        "rms_A": dist.tolist(),
        "errors_kcal": vals,
    }


def run(stage, panel=None):
    if panel is None:
        panel = stage
    root = OUT.parent / f"path-kernel{panel}"
    held = [r for r in json.loads((root / "manifest.json").read_text()) if r["heldout"]]
    models = ["p43", f"p{stage}"]
    with (
        threadpool_limits(1),
        ProcessPoolExecutor(
            max_workers=2, initializer=load, initargs=(models,)
        ) as pool,
    ):
        res = list(pool.map(score, held))
    arr = np.array([r["errors_kcal"] for r in res if r["status"] == "assigned"])
    metrics = {}
    if len(arr):
        for m, err in zip(models, np.abs(arr).mean(0), strict=True):
            metrics[m] = dict(
                zip(
                    ("reaction", "midpoint", "forward", "reverse"),
                    err.tolist(),
                    strict=True,
                )
            )
            metrics[m]["pooled"] = (err[2] + err[3]) / 2
    kt.write_json(
        root
        / (
            "deployed-endpoint-validation.json"
            if panel == stage
            else f"p{stage}-deployed-endpoint-validation.json"
        ),
        {
            "heldout": len(held),
            "assigned": len(arr),
            "models": models,
            "mae_kcal": metrics,
            "per_reaction": res,
        },
    )
    print(
        "Held-out directed electronic IRC endpoint validation",
        len(arr),
        "/",
        len(held),
        metrics,
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=int)
    p.add_argument("--panel", type=int)
    opts = p.parse_args()
    run(opts.stage, opts.panel)
