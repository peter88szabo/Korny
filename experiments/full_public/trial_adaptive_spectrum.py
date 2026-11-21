"""Optimize Fourier frequencies on training reactions, then fit the energy/force readout.

The nonlinear pilot uses only folds 2 through 7, with a shared forbidden-geometry
check across data sources. A fixed PBE0+D3 prior supplies residual targets and the
two electronic reference levels have independent linear pilot coefficients.
An identical seeded, frozen spectrum supplies a controlled ablation. Final kernel
weights in both arms use the full existing split, force and curvature constraints.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from scipy.sparse import vstack
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_fidelity import rdb_dft
from experiments.full_public.trial_sites import ACTIVE
from src.reactive_basis import ReactiveBasis
from src.adaptive_spectrum import SpectralKernel

WORK = kt.WORK
WIDTHS = (0.85, 1.3)
SIZE = 112


def tables(auxiliary=False):
    prefix = "rich-aux" if auxiliary else "rich"
    return {
        s: (
            np.load(WORK / f"{prefix}-off-{s}.npy", mmap_mode="r"),
            np.load(WORK / f"{prefix}-sites-{s}.npy", mmap_mode="r"),
        )
        for s in ACTIVE
    }


def geometry_atoms(offsets, ids):
    begin = offsets[ids]
    counts = offsets[ids + 1] - begin
    present = np.flatnonzero(counts)
    starts = np.cumsum(counts) - counts
    owner = np.repeat(np.arange(len(ids)), counts)
    rows = np.arange(counts.sum()) + np.repeat(begin - starts, counts)
    return rows, owner, present, starts[present]


def observations():
    high, rad = (kt.equation_data(t) for t in kt.TARGETS)
    dft = rdb_dft(high)
    forbidden = set()
    for d in (high, rad):
        forbidden.update(map(int, d[0][d[3] < 2].indices))
    base = np.load(WORK / "prior.npy")
    slots = []
    targets = []
    which = []
    groups = []
    for head, d, source, label in (
        (0, high, "RDB7", "RDB7 CC"),
        (0, high, "BH9", "BH9"),
        (0, high, "OSOS", "OSOS"),
        (1, dft, "RDB7", "RDB7 DFT"),
        (1, rad, "Grambow", "Grambow"),
    ):
        mask = (d[3] >= 2) & (d[2] == source)
        s = d[0][mask]
        safe = np.asarray(s[:, sorted(forbidden)].getnnz(axis=1) == 0).ravel()
        s = s[safe]
        y = d[1][mask][safe] - s @ base
        slots.append(s)
        targets.append(y)
        which.extend([head] * len(y))
        groups.extend([label] * len(y))
    return (
        vstack(slots, format="csr"),
        np.concatenate(targets),
        np.array(which),
        np.array(groups),
    )


def optimize():
    geom = tables()
    matrix, y, head, group = observations()
    s0 = np.random.default_rng(740)
    with np.load(WORK / "rich-basis.npz") as f:
        source = ReactiveBasis.load(f)
    freq = {
        s: s0.normal(0, 0.7, (source.projections[s].shape[1], SIZE)) for s in ACTIVE
    }
    initial = {s: v.copy() for s, v in freq.items()}
    coef = {s: s0.normal(0, 0.025, (2 * SIZE, 2)) for s in ACTIVE}
    moments = {
        (kind, s): [np.zeros_like(v), np.zeros_like(v)]
        for kind, vals in (("omega", freq), ("coef", coef))
        for s, v in vals.items()
    }
    pools = [np.flatnonzero(group == label) for label in np.unique(group)]
    print(
        "Nonlinear pilot equations", [(group[p[0]], len(p)) for p in pools], flush=True
    )

    def adam(k, value, gradient, step, rate):
        first, second = moments[k]
        first *= 0.9
        first += 0.1 * gradient
        second *= 0.995
        second += 0.005 * gradient**2
        value -= (
            rate
            * (first / (1 - 0.9**step))
            / (np.sqrt(second / (1 - 0.995**step)) + 1.0e-7)
        )

    history = []
    for step in range(1, 1001):
        ids = np.concatenate(
            [s0.choice(pool, min(28, len(pool)), replace=False) for pool in pools]
        )
        mat = matrix[ids]
        geometries, inv = np.unique(mat.indices, return_inverse=True)
        # Compact this sparse batch in place. Its row order (and thus its signs)
        # stays identical to the original stoichiometric energy equations.
        from scipy.sparse import csr_matrix

        local = csr_matrix(
            (mat.data, inv, mat.indptr), shape=(len(ids), len(geometries))
        )
        energy = np.zeros((len(geometries), 2))
        cache = {}
        for s, (offset, design) in geom.items():
            rows, owner, present, start = geometry_atoms(offset, geometries)
            if not len(rows):
                continue
            u = design[rows]
            phase = u @ freq[s]
            c = np.cos(phase)
            sin = np.sin(phase)
            z = np.column_stack((c - 1.0, sin))
            energy[present] += np.add.reduceat(z @ coef[s], start, axis=0)
            cache[s] = u, c, sin, z, owner
        resid = (local @ energy)[np.arange(len(ids)), head[ids]] - y[ids]
        # Smooth absolute loss limits outlier leverage. The gradient below is
        # evaluated through both the atom sum and the reaction stoichiometry.
        yy = np.zeros((len(ids), 2))
        yy[np.arange(len(ids)), head[ids]] = resid / np.hypot(resid, 0.28) / len(ids)
        upstream = local.T @ yy
        for s, (u, c, sin, z, owner) in cache.items():
            adj = upstream[owner]
            dphase = -(adj @ coef[s][:SIZE].T) * sin + (adj @ coef[s][SIZE:].T) * c
            dw = u.T @ dphase + 0.0006 * (freq[s] - initial[s])
            da = z.T @ adj + 0.0008 * coef[s]
            rate = 1.0 / (1.0 + step / 1400)
            adam(("omega", s), freq[s], dw, step, 0.0040 * rate)
            adam(("coef", s), coef[s], da, step, 0.0090 * rate)
        history.append(float(np.mean(abs(resid)) * kt.KCAL))
        if step % 250 == 0:
            print(
                "Supervised spectrum",
                step,
                "recent pilot batch MAE",
                np.mean(history[-60:]),
                flush=True,
            )
    for tag, values in (("adaptive", freq), ("frozen", initial)):
        b = SpectralKernel(
            source.species,
            source.spec,
            source.projections,
            values,
            1.0,
            np.zeros(2 * SIZE * len(ACTIVE)),
        )
        np.savez_compressed(WORK / f"{tag}-basis.npz", **b.fields())
    kt.write_json(
        WORK / "adaptive-frequency-fit.json",
        dict(
            equations={str(group[p[0]]): int(len(p)) for p in pools},
            steps=step,
            initial_batch_mae_kcal=float(np.mean(history[:60])),
            final_batch_mae_kcal=float(np.mean(history[-60:])),
            mean_frequency_displacement={
                str(s): float(
                    np.linalg.norm(freq[s] - initial[s]) / np.sqrt(freq[s].size)
                )
                for s in ACTIVE
            },
        ),
    )


def encode(bounds):
    lo, hi = bounds
    ids = np.arange(lo + skip, hi + skip)
    for j, s in enumerate(ACTIVE):
        offsets, x = atoms[s]
        ix, _, present, start = geometry_atoms(offsets, ids)
        if not len(ix):
            continue
        phase = x[ix] @ model.frequencies[s]
        for w, mat in zip(WIDTHS, arrays):
            v = phase / w
            sl = slice(j * 2 * SIZE, (j + 1) * 2 * SIZE)
            mat[lo + present, sl] = np.add.reduceat(
                np.column_stack((np.cos(v) - 1.0, np.sin(v))), start, axis=0
            )


def cache(tag):
    global model, arrays, atoms, skip
    with np.load(WORK / f"{tag}-basis.npz") as f:
        model = SpectralKernel.load(f)
    npriv = json_count()
    for side in ("main", "rad"):
        paths = [WORK / f"{tag}-{w:g}-{side}.npy" for w in WIDTHS]
        if all(p.exists() for p in paths):
            continue
        atoms = tables(side == "rad")
        skip = npriv if side == "rad" else 0
        n = len(atoms[1][0]) - 1 - skip
        arrays = [
            np.lib.format.open_memmap(
                p.with_suffix(".pending"), mode="w+", dtype=float, shape=(n, model.size)
            )
            for p in paths
        ]
        with ProcessPoolExecutor(
            max_workers=5, mp_context=mp.get_context("fork")
        ) as pool:
            for _ in pool.map(
                encode, [(i, min(i + 200, n)) for i in range(0, n, 200)], buffersize=14
            ):
                pass
        for arr, p in zip(arrays, paths):
            arr.flush()
            p.with_suffix(".pending").replace(p)
        print("Completed spectrum features", tag, side, n, flush=True)
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, keys = pickle.load(f)
    rng = np.random.default_rng(158)
    for idx in rng.choice(len(keys), 6, replace=False):
        g = gs[keys[idx]]
        z, r = g.z, g.r
        v = rng.normal(size=r.shape)
        v /= np.linalg.norm(v)
        for w in WIDTHS:
            model.length = w
            a, j = model.design(z, r, True)
            stored = np.load(WORK / f"{tag}-{w:g}-main.npy", mmap_mode="r")[idx]
            np.testing.assert_allclose(a, stored, atol=3.0e-8)
            diff = (
                model.design(z, r - 1.0e-5 * v)[0] - model.design(z, r + 1.0e-5 * v)[0]
            ) / 2.0e-5
            np.testing.assert_allclose(j.T @ v.ravel(), diff, atol=4.0e-6)


def json_count():
    import json

    return int(json.loads((WORK / "base-aux-ready.json").read_text())["private"])


if __name__ == "__main__":
    from experiments.full_public.trial_curvature_fit import fit
    from experiments.full_public.trial_spectra import evaluate_trial

    with threadpool_limits(1):
        if not (WORK / "adaptive-frequency-fit.json").exists():
            optimize()
    for tag, stage in (("adaptive", 34), ("frozen", 35)):
        with threadpool_limits(1):
            cache(tag)
        with threadpool_limits(4):
            fit(stage)
        with threadpool_limits(1):
            evaluate_trial(f"p{stage}-public")
