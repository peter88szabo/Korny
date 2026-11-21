"""Estimate vibrational force constants using the archived QM7-X PBE0 force changes.

Each training conformer has a reference geometry and 100 normal-mode distortions.
Align coordinates and vectors in the reference frame and remove rigid motions.
Fit a symmetric force-constant matrix. Withheld distortions of that same training
conformer and split-half spectra assess the reliability of each local estimate.
These are estimates from displaced forces, separate from analytic PBE0 Hessians.
"""

from collections import defaultdict
import hashlib
import pickle
import sqlite3
import re
import numpy as np
from rdkit import Chem
from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import DB
from experiments.reactivity.stationary_fit import internal_transform
from experiments.reactivity.frequency_validation import TO_CM1


def symmetric_response(u, v, weight):
    """Solve symmetric ridge regression by diagonalizing the normal operator."""
    a = (u * weight[:, None]).T @ u
    b = (u * weight[:, None]).T @ v
    eig, axes = np.linalg.eigh(a)
    ridge = max(0.000002, 0.0008 * np.trace(a) / len(a))
    inner = axes.T @ (b + b.T) @ axes
    return axes @ (inner / (eig[:, None] + eig[None, :] + 2 * ridge)) @ axes.T


def solve_group(name, records):
    center = next(row for row in records if row[0].endswith("-opt"))
    _, z, ref, f0 = center
    others = [row for row in records if row is not center]
    mass = np.array(
        [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(a)) for a in z]
    )
    root = np.sqrt(mass)[:, None]
    tf = internal_transform(mass, ref)
    internal = root.repeat(3, axis=1).ravel()[:, None] * tf
    r0 = ref - np.average(ref, axis=0, weights=mass)
    u = []
    v = []
    amplitudes = []
    for _, numbers, r, f in others:
        np.testing.assert_array_equal(z, numbers)
        xc = r - np.average(r, axis=0, weights=mass)
        left, _, right = np.linalg.svd((xc * mass[:, None]).T @ r0)
        if np.linalg.det(left @ right) < 0:
            left[:, -1] *= -1
        rotation = left @ right
        delta = xc @ rotation - r0
        u.append((delta * root).ravel() @ internal)
        v.append(((f0 - f @ rotation) / root).ravel() @ internal)
        amplitudes.append(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    u = np.asarray(u)
    v = np.asarray(v)
    amplitudes = np.asarray(amplitudes)
    if not len(u):
        return None
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    )
    perm = rng.permutation(len(u))
    train, test = perm[::], perm[:20]
    train = train[20:]
    # Downweight the largest distortions to reduce anharmonic bias.
    w = 1.0 / (1 + (amplitudes / 0.10) ** 2)
    a = symmetric_response(u[train], v[train], w[train])
    pred = u[test] @ a
    r2 = 1.0 - np.sum((pred - v[test]) ** 2) / max(np.sum(v[test] ** 2), 1.0e-12)
    b = symmetric_response(u[perm[::2]], v[perm[::2]], w[perm[::2]])
    c = symmetric_response(u[perm[1::2]], v[perm[1::2]], w[perm[1::2]])
    result = symmetric_response(u, v, w)
    eig, axes = np.linalg.eigh(result)
    # Compare two independent force fits along identical final directions.
    diag = lambda k: np.einsum("ij,ij->j", axes, k @ axes)
    frequency = lambda x: TO_CM1 * np.sign(x) * np.sqrt(abs(x))
    split = np.abs(frequency(diag(b)) - frequency(diag(c)))
    return (
        name,
        z,
        ref,
        f0,
        tf,
        axes,
        eig,
        split,
        float(r2),
        len(others),
        float(np.median(amplitudes)),
    )


def run():
    groups = defaultdict(list)
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        for name, z, r, f, q, m in db.execute(
            "select name,z,r,force,q,m from samples where source='QM7X' and split='train' order by name"
        ):
            if (q, m) != (0, 1):
                raise ValueError("Unexpected electronic state in QM7-X")
            key = re.sub(r"-(?:opt|d\d+)$", "", name)
            groups[key].append(
                (
                    name,
                    np.frombuffer(z, np.uint8),
                    np.frombuffer(r, float).reshape(-1, 3),
                    np.frombuffer(f, float).reshape(-1, 3),
                )
            )
    fits = [r for k, v in groups.items() if (r := solve_group(k, v)) is not None]
    with (kt.WORK / "qm7x-local-hessians.pkl").open("wb") as f:
        pickle.dump(fits, f)
    data = dict(
        conformers=len(fits),
        force_geometries=sum(r[9] + 1 for r in fits),
        test_force_r2_percentiles=np.percentile(
            [r[8] for r in fits], [10, 50, 90]
        ).tolist(),
        median_split_mode_difference_cm1=float(
            np.median(np.concatenate([r[7] for r in fits]))
        ),
        conformers_with_test_r2_above_080=sum(r[8] > 0.8 for r in fits),
    )
    kt.write_json(kt.WORK / "qm7x-local-hessians.json", data)
    print("QM7-X force-change estimates", data, flush=True)


if __name__ == "__main__":
    run()
