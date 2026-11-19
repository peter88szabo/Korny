"""Use Grambow Cartesian Hessians without equating different electronic levels.

At each available structure constrain two measured curvatures along its softest
internal mode (including the unstable direction of a saddle) and its highest
stretch. The scalar energy's centered difference supplies the corresponding
kernel row. Validation directions and eigenvalues are never fed to the solver.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import pickle
import tarfile
from time import monotonic

import numpy as np
from rdkit import Chem

from experiments.full_public.specified import OUT
from experiments.full_public.references import fold
from experiments.reactivity.radical_curvatures import ARCHIVE, final_hessian
from experiments.reactivity.stationary_fit import internal_transform
from src.dispersion.d3 import d3_energy_forces
from src.forcefield import ForceField


def rows_for(item):
    name, z, r, st, h, stationary = item
    split = fold(tuple(sorted(map(int, z))))
    surface = model
    v, df = surface.design(z, r, *st, True)
    base, f = surface.prior(z, r, *st, True)
    de, disp_force = d3_energy_forces(z, r)
    e0 = base + de
    observations = []
    if stationary and split == "train":
        # Measured equilibria and saddles both have zero first derivative.
        weight = 2.0 / np.sqrt(r.size)
        observations.append(weight * np.column_stack((df, -(f + disp_force).ravel())))
    mass = np.array(
        [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s)) for s in z]
    )
    internal = internal_transform(mass, r)
    vals, axes = np.linalg.eigh(internal.T @ h @ internal)
    samples = []
    for index in (0, -1):
        vec = (internal @ axes[:, index]).reshape(r.shape)
        vec /= np.linalg.norm(vec)
        step = 0.004

        def calc(x):
            return surface.design(z, x, *st)[0], surface.prior(z, x, *st)[
                0
            ] + d3_energy_forces(z, x)[0]

        vp, ep = calc(r + step * vec)
        vm, em = calc(r - step * vec)
        row = (vp + vm - 2 * v) / step**2
        prior = (ep + em - 2 * e0) / step**2
        kref = float(vec.ravel() @ h @ vec.ravel())
        inertia = np.sum(mass[:, None] * vec**2)
        if split == "train":
            observations.append((0.16 * np.r_[row, kref - prior])[None, :])
        else:
            samples.append((name, index, row, prior, kref, inertia))
    a = np.vstack(observations) if observations else None
    if a is not None:
        cols = np.flatnonzero(np.any(a != 0, axis=0))
        a = cols, a[:, cols]
    return split, name, a, samples


def run():
    global model
    model = ForceField.load(OUT / "wb97xd3-reaction.npz")
    data = []
    excluded = Counter()
    with tarfile.open(ARCHIVE, "r|gz") as arc:
        for member in arc:
            if not member.isfile() or not member.name.endswith(".log"):
                continue
            s = arc.extractfile(member).read().decode(errors="replace")
            try:
                g = final_hessian(s)
            except (ValueError, IndexError, StopIteration):
                g = None
            if g is None:
                excluded["missing_or_incomplete_hessian"] += 1
                continue
            z, r, st, h = g
            data.append((member.name, z, r, st, h, "OPTIMIZATION CONVERGED" in s))
    print("verified Hessians", len(data), "unavailable", excluded, flush=True)
    dim = len(model.weights)
    gram = np.zeros((dim + 1, dim + 1))
    held = []
    count = Counter()
    t = monotonic()
    nrows = 0
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for split, name, a, validation in pool.map(
            rows_for, data, chunksize=4, buffersize=10
        ):
            count[split] += 1
            held.extend(validation)
            if a is not None:
                ids, x = a
                gram[np.ix_(ids, ids)] += x.T @ x
                nrows += len(x)
            if sum(count.values()) % 1000 == 0:
                print(
                    "Hessians",
                    dict(count),
                    round(monotonic() - t),
                    "seconds",
                    flush=True,
                )
    np.save(OUT / "radical-derivative-gram.npy", gram)
    with (OUT / "radical-projections.pkl").open("wb") as f:
        pickle.dump(held, f, protocol=5)
    (OUT / "radical-derivatives.json").write_text(
        json.dumps(dict(counts=count, excluded=excluded, training_rows=nrows), indent=2)
        + "\n"
    )
    print(dict(count), nrows, "training rows", flush=True)


if __name__ == "__main__":
    run()
