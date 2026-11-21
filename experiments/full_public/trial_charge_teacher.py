"""Learn electronic environment labels from the training portion of existing archives.

Hirshfeld and formal charges are two independent bond-transfer fits. The two
remaining observables are local pi valence and the number of radical electrons.
Development geometries only enter the diagnostic pass after the fits finish.
"""

import pickle
from collections import Counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk
from experiments.full_public import kernel_trials as kt, trial_periodic as periodic
from src.atomic_states import StateFeatures, neighbors, laplacian
from src.periodic_kernel import PeriodicKernel

WORK = kt.WORK


def load_teacher():
    with np.load(WORK / "atom-property-teacher.npz") as data:
        geom = PeriodicKernel.load(data)
        return StateFeatures(geom, {g: data[f"coeff_{g}"] for g in geom.centers})


def projected(teacher, z, r):
    """Use the native density evaluator and the same projection as the deployed model."""
    p = teacher.periodic
    arr = periodic.raw(z, r)
    types = p.groups[z]
    result = {}
    for g in np.unique(types):
        if g not in p.projections:
            continue
        data = arr[types == g]
        active = np.flatnonzero(np.any(data, axis=0))
        result[g] = data[:, active] @ p.projections[g][active]
    return result


class NormalEquations:
    """Accumulate dense least squares in small blocks with bounded memory."""

    def __init__(self, width):
        self.gram = np.zeros((width + 1, width + 1), order="F")
        self.pending = []
        self.count = 0

    def add(self, x, y, weights=1.0):
        self.pending.append(np.column_stack((x, y)) * weights)
        self.count += len(x)
        if self.count >= 2048:
            self.flush()

    def flush(self):
        if self.count:
            block = np.asfortranarray(np.vstack(self.pending).T)
            self.gram = dsyrk(1.0, block, beta=1.0, c=self.gram, lower=1, overwrite_c=1)
            self.count = 0
            self.pending.clear()

    def solve(self):
        self.flush()
        gram = self.gram[:-1, :-1].copy()
        gram.flat[:: len(gram) + 1] += 0.15
        return cho_solve(cho_factor(gram, lower=True), self.gram[-1, :-1])


def train():
    periodic.setup()
    with np.load(WORK / "periodic-basis.npz") as data:
        p = PeriodicKernel.load(data)
    for g, centers in p.centers.items():
        p.centers[g] = centers[
            np.linspace(0, len(centers) - 1, min(len(centers), 112), dtype=int)
        ]
    p.size = sum(map(len, p.centers.values()))
    p.weights = np.zeros(p.size)
    slices = {}
    width = 0
    for g, c in p.centers.items():
        slices[g] = slice(width, width + 1 + 2 * len(c))
        width = slices[g].stop
    teacher = StateFeatures(p, {})
    fits = [NormalEquations(width) for _ in range(5)]
    archives = [("qm7x-charge-labels", "train"), ("atom-state-labels", "fit")]
    for source, (fname, train_split) in enumerate(archives):
        with (WORK / (fname + ".pkl")).open("rb") as f:
            rows = pickle.load(f)
        for _, split, z, r, q, m, y in rows:
            if split != train_split:
                continue
            u = projected(teacher, z, r)
            atom = np.zeros((len(z), width))
            for g, vals in u.items():
                atom[p.groups[z] == g, slices[g]] = teacher.interpolation(g, vals)[0]
            a, b, _, w, _ = neighbors(z, r, p.radii)
            transfer = laplacian(len(z), a, b, w[:, 0]) @ atom
            shares = p.radii[z] ** 2
            fits[source].add(
                transfer, (y if source == 0 else y[:, 0]) - q * shares / shares.sum()
            )
            if source == 1:
                fits[2].add(atom, y[:, 2])
                fits[3 if m == 1 else 4].add(
                    atom, y[:, 1], np.sqrt(1 + 3 * (y[:, 1:2] > 0))
                )
        print("Electronic label training pass completed", fname, flush=True)
    coeff = np.column_stack([fit.solve() for fit in fits])
    teacher.coefficients = {g: coeff[s] for g, s in slices.items()}
    values = dict(p.fields())
    values.update({f"coeff_{g}": c for g, c in teacher.coefficients.items()})
    np.savez_compressed(WORK / "atom-property-teacher.npz", **values)
    errors = Counter()
    counts = Counter()
    for source, (fname, _) in enumerate(archives):
        with (WORK / (fname + ".pkl")).open("rb") as f:
            rows = pickle.load(f)
        for _, split, z, r, q, m, y in rows:
            if split != "dev":
                continue
            fields = teacher.context(z, r, q, m, projected(teacher, z, r))[2]
            if source == 0:
                pairs = [("Hirshfeld charge", fields[:, 0] - y)]
            else:
                pairs = [
                    ("formal charge", fields[:, 1] - y[:, 0]),
                    ("pi valence", fields[:, 2] - y[:, 2]),
                    (
                        "unpaired electrons (singlet)"
                        if m == 1
                        else "unpaired electrons (open-shell)",
                        fields[:, 3] - y[:, 1],
                    ),
                ]
                ids = np.flatnonzero(y[:, 1] > 0)
                if m > 1 and len(ids) == 1:
                    pairs.append(
                        (
                            "single radical top-1 accuracy",
                            np.array([int(np.argmax(fields[:, 3]) == ids[0])]),
                        )
                    )
            for label, v in pairs:
                errors[label] += np.abs(v).sum()
                counts[label] += len(v)
    metrics = {
        k: {"n": counts[k], "mean": float(v / counts[k])} for k, v in errors.items()
    }
    kt.write_json(WORK / "atom-property-diagnostics.json", metrics)
    print("Development electronic-property metrics", metrics, flush=True)


if __name__ == "__main__":
    train()
