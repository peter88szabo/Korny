"""Compute p42 predictions and its existing polynomial basis for the new reference data.

The same scalar potential used for validation supplies both endpoint and TS
energies. Cache shared geometries once, even when two electronic reference
levels are available. Features and parent energies are checked against the
deployment API before they can enter the training loss.
"""

import multiprocessing as mp
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_periodic as kernel
from experiments.full_public.paper_data import OUT
from experiments.full_public.trial_measured_curvature import load_basis
from src.forcefield import ForceField

PARENT = "p42-public"
ROOT = OUT / "features42"


def jobs():
    """CC and explicit radical DFT use their matching legacy potential as a prior."""
    query = 'select id,method,z,r,charge,mult,signature from geometries where duplicate_of is null and split in ("train","validation") order by id'
    slots = {}
    items = []
    index = {}
    with sqlite3.connect(f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True) as db:
        for name, method, zb, rb, q, m, sig in db.execute(query):
            target = (
                "high"
                if "CCSD" in method or m == 1 and method.startswith("BASChem")
                else "wb97xd3"
            )
            key = (sig, q, m, target)
            if key not in slots:
                slots[key] = len(items)
                z = np.frombuffer(zb, np.uint8)
                r = np.frombuffer(rb, "<f8").reshape(-1, 3)
                items.append((z, r, q, m, target))
            index[name] = slots[key]
    return items, index


def chunk(lo):
    dest = ROOT / f"block-{lo:06d}.npz"
    if dest.exists():
        return
    fragment = items[lo : lo + 32]
    raw = kernel.batch([(z, r) for z, r, *_ in fragment])[0][:, cols]
    baseline = np.array([models[t].energy(z, r, q, m) for z, r, q, m, t in fragment])
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez(f, x=raw, base=baseline)
    dest.with_suffix(".pending").replace(dest)


def prepare():
    global items, cols, models
    ROOT.mkdir(exist_ok=True)
    items, index = jobs()
    kt.write_json(ROOT / "index.json", index)
    kernel.setup()
    kernel.basis, cols = load_basis()
    kernel.OPTIONS = ((0.85, 3),)
    models = {
        t: ForceField.load(kt.WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        for t in kt.TARGETS
    }
    for i in np.random.default_rng(142).choice(len(items), 4, replace=False):
        z, r, q, m, t = items[i]
        a = kernel.batch([(z, r)])[0][0, cols]
        b, _df = kernel.basis.design(z, r, True)
        np.testing.assert_allclose(a, b[cols], atol=1e-7)
        model = models[t]
        np.testing.assert_allclose(
            model.energy(z, r, q, m), model.predict(z, r, q, m)[0], rtol=0, atol=2e-7
        )
    print(
        "Verified training-to-deployment energies and basis; encoding",
        len(items),
        "unique state/geometry/parent rows",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for j, _ in enumerate(pool.map(chunk, range(0, len(items), 32), buffersize=9)):
            if (j + 1) % 64 == 0:
                print(
                    "Prepared paper feature rows",
                    min((j + 1) * 32, len(items)),
                    len(items),
                    flush=True,
                )
    print(
        "Completed energy-feature preparation for published reactive data", flush=True
    )


if __name__ == "__main__":
    with threadpool_limits(1):
        prepare()
