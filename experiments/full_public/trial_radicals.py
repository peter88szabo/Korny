"""Partition archived radical curvature observations by inner TS formula.

Save the measured softest and highest-frequency directions at each structure for
evaluating new features. Subtract the withheld fold's exact legacy Gram when
generating out-of-fold radical predictions. No quantum calculations are run.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
import pickle
import tarfile

import numpy as np
from rdkit import Chem
from experiments.full_public import radical_hessians as rh
from experiments.full_public.kernel_trials import OUT, WORK
from experiments.full_public.references import fold
from experiments.reactivity.stationary_fit import internal_transform
from src.forcefield import ForceField


def inner(z):
    comp = tuple(sorted(map(int, z)))
    return (
        int.from_bytes(
            hashlib.sha256(("kt1/" + str(comp)).encode()).digest()[:4], "big"
        )
        % 8
    )


def extract(item):
    name, z, r, st, h, stationary = item
    mass = np.array(
        [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(a)) for a in z]
    )
    t = internal_transform(mass, r)
    _, axes = np.linalg.eigh(t.T @ h @ t)
    modes = []
    for i in (0, -1):
        v = (t @ axes[:, i]).reshape(r.shape)
        v /= np.linalg.norm(v)
        modes.append(
            (v, float(v.ravel() @ h @ v.ravel()), float(np.sum(mass[:, None] * v**2)))
        )
    split = fold(tuple(sorted(map(int, z))))
    f = inner(z) if split == "train" else -1
    gram = rh.rows_for(item)[2] if f in (0, 1) else None
    return f, (name, z, r, st, stationary, modes), gram


def prepare():
    if (WORK / "radical-modes.pkl").exists():
        return
    rh.model = ForceField.load(OUT / "wb97xd3-reaction.npz")

    def items():
        with tarfile.open(rh.ARCHIVE, "r|gz") as arc:
            for m in arc:
                if not m.isfile() or not m.name.endswith(".log"):
                    continue
                txt = arc.extractfile(m).read().decode(errors="replace")
                try:
                    h = rh.final_hessian(txt)
                except (ValueError, IndexError, StopIteration):
                    h = None
                if h is not None:
                    yield (m.name, *h, "OPTIMIZATION CONVERGED" in txt)

    dim = len(rh.model.weights) + 1
    grams = [np.zeros((dim, dim)) for _ in range(2)]
    records = []
    counts = Counter()
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("fork")) as pool:
        for f, item, sparse in pool.map(extract, items(), chunksize=4, buffersize=16):
            records.append((f, item))
            counts[f] += 1
            if sparse is not None:
                ids, v = sparse
                grams[f][np.ix_(ids, ids)] += v.T @ v
    assert counts[-1] == 844 and len(records) == 4078
    for f, a in enumerate(grams):
        np.save(WORK / f"radical-fold-{f}.npy", a)
    with (WORK / "radical-modes.tmp").open("wb") as stream:
        pickle.dump(records, stream)
    (WORK / "radical-modes.tmp").replace(WORK / "radical-modes.pkl")
    print("Derivative split", counts, flush=True)


if __name__ == "__main__":
    prepare()
