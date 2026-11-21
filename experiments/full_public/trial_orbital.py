"""Train sparse kernels on sigma/pi orbital paths and score them on the Korny split.

Centers and projections are fitted only on eligible training formulas. Two
kernel widths share one set of geometry descriptors. Every original HDF5
energy / projected force / curvature anchor and radical displacement remains
in its original order. Each file is published after all workers finish.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt, trial_sites, trial_aux
from experiments.full_public.trial_messages import prototypes
from experiments.reactivity.reaction_classes import lookup
from src.forcefield import ForceField
from src.orbital_moments import OrbitalMomentKernel, environments, CHANNELS, POWERS

WORK = kt.WORK
LENGTHS = (0.8, 1.5)


def learn(gs, rows, classes):
    model = ForceField.load(WORK / "p25-public-high.npz")
    groups = model.groups.copy()
    groups[1] = 0
    radii = model.radii
    classes.update(lookup())
    keys = trial_sites.candidates(gs, rows, classes)
    sites = {}
    rng = np.random.default_rng(65883)
    for key in keys:
        g = gs[key]
        x = environments(g.z, g.r, groups, radii)[0]
        for t in np.unique(groups[g.z]):
            sites.setdefault(int(t), []).append(x[groups[g.z] == t])
    centers = {}
    proj = {}
    for g in sorted(sites):
        x = np.vstack(sites.pop(g))
        if len(x) > 18000:
            x = x[rng.choice(len(x), 18000, replace=False)]
        if len(x) < 3:
            continue
        split = (2 * CHANNELS, 2 * CHANNELS + 2 * len(POWERS))
        blocks = []
        for data in np.split(x, split, axis=1):
            # Normalize each block's second moment before PCA so higher
            # powers do not suppress the low-order angular information.
            scale = np.maximum(np.std(data, axis=0), 0.035)
            val, v = np.linalg.eigh(np.cov(data / scale, rowvar=False))
            active = val > 1.0e-6
            if not active.any():
                blocks.append(np.empty((len(scale), 0)))
                continue
            blocks.append(
                (v[:, active] / scale[:, None])
                / np.sqrt((val[active] + 0.06) * active.sum())
            )
        proj[g] = block_diag(*blocks)
        if proj[g].shape[1] == 0:
            del proj[g]
            continue
        projected = x @ proj[g]
        count = min(1100, max(32, int(13 * np.sqrt(len(x)))), len(x))
        centers[g] = prototypes(projected, count, rng)
    basis = OrbitalMomentKernel(
        groups,
        radii,
        centers,
        proj,
        1.0,
        np.zeros(sum(len(c) for c in centers.values())),
    )
    np.savez_compressed(WORK / "orbital-basis.npz", **basis.fields())
    print(
        "Training-only orbital moment centers",
        {g: len(c) for g, c in centers.items()},
        flush=True,
    )
    return basis


def batch(geometries):
    types = np.concatenate([basis.groups[z] for z, r in geometries])
    owner = np.repeat(np.arange(len(geometries)), [len(z) for z, r in geometries])
    raw = np.vstack(
        [environments(z, r, basis.groups, basis.radii)[0] for z, r in geometries]
    )
    out = [np.zeros((len(geometries), basis.size)) for w in LENGTHS]
    start = 0
    for g, c in basis.centers.items():
        mask = types == g
        if mask.any():
            u = raw[mask] @ basis.projections[g]
            sq = np.einsum("ij,ij->i", c, c)
            d2 = np.maximum(0, np.einsum("ij,ij->i", u, u)[:, None] + sq - 2 * u @ c.T)
            who, idx = np.unique(owner[mask], return_index=True)
            for mat, w in zip(out, LENGTHS):
                values = np.exp(-0.5 * d2 / w**2) - np.exp(-0.5 * sq / w**2)
                mat[who, start : start + len(c)] = np.add.reduceat(values, idx, axis=0)
        start += len(c)
    return out


def work(i):
    sl = slice(i, min(i + 40, len(points)))
    for a, v in zip(arrays, batch(points[sl])):
        a[sl] = v


def cache(side, data):
    global points, arrays
    points = data
    paths = [WORK / f"orbital-{w:g}-{side}.npy" for w in LENGTHS]
    if all(p.exists() for p in paths):
        return
    arrays = [
        np.lib.format.open_memmap(
            p.with_suffix(".pending"),
            mode="w+",
            dtype=float,
            shape=(len(points), basis.size),
        )
        for p in paths
    ]
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        for j, _ in enumerate(pool.map(work, range(0, len(points), 40), buffersize=16)):
            if (j + 1) % 2300 == 0:
                print("Orbital geometries encoded", side, (j + 1) * 40, flush=True)
    for a, p in zip(arrays, paths):
        a.flush()
        p.with_suffix(".pending").replace(p)
    print("Completed orbital moment columns", side, len(points), flush=True)


def prepare():
    global basis
    gs, rows, classes = kt.inputs()
    basis = learn(gs, rows, classes)
    with (WORK / "ids.pkl").open("rb") as f:
        _, keys = pickle.load(f)
    data = [(gs[k].z, gs[k].r) for k in keys]
    rng = np.random.default_rng(591)
    for i in rng.choice(len(keys), 8, replace=False):
        z, r = data[i]
        v = rng.normal(size=r.shape)
        v /= np.linalg.norm(v)
        results = batch([(z, r), (z, r - 1.0e-5 * v), (z, r + 1.0e-5 * v)])
        for w, x in zip(LENGTHS, results):
            basis.length = w
            ref, j = basis.design(z, r, True)
            np.testing.assert_allclose(x[0], ref, atol=1.0e-9)
            np.testing.assert_allclose(
                j.T @ v.ravel(), (x[1] - x[2]) / 2.0e-5, atol=2.0e-6
            )
    print("Verified analytic orbital-moment forces", flush=True)
    for _, z, r, v in trial_sites.hdf():
        data.extend([(z, r), (z, r + 0.001 * v), (z, r - 0.001 * v)])
    cache("main", data)
    aux, npriv, _ = trial_aux.read_frames()
    cache("rad", aux[npriv:])


if __name__ == "__main__":
    from experiments.full_public.trial_curvature_fit import fit
    from experiments.full_public.trial_spectra import evaluate_trial

    with threadpool_limits(1):
        prepare()
    with threadpool_limits(5):
        fit(27)
    with threadpool_limits(1):
        evaluate_trial("p27-public")
