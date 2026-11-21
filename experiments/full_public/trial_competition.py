"""Fit a differentiable local kernel on competing bonds and both sides of a barrier."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_sites, trial_aux
from experiments.full_public.trial_messages import prototypes
from experiments.reactivity.reaction_classes import lookup
from src.forcefield import ForceField
from src.valence_competition import CompetitionKernel, events

WORK = kt.WORK
LENGTHS = (0.75, 1.35)


def learn(gs, rows, classes):
    parent = ForceField.load(WORK / "p25-public-high.npz")
    groups = parent.groups.copy()
    groups[1] = 0
    radii = parent.radii
    classes.update(lookup())
    keys = trial_sites.candidates(gs, rows, classes)
    examples = {}
    for key in keys:
        g = gs[key]
        types, values, w, *_ = events(g.z, g.r, groups, radii)
        for t in np.unique(types):
            use = (types == t) & (w > 0.025)
            if use.any():
                examples.setdefault(int(t), []).append(values[use])
    centers = {}
    projections = {}
    rng = np.random.default_rng(9492)
    for t, records in sorted(examples.items()):
        x = np.vstack(records)
        if len(x) < 3:
            continue
        if len(x) > 14000:
            x = x[rng.choice(len(x), 14000, replace=False)]
        # Retain physical length/angle variation even when equilibrium training
        # samples give those directions little variance.
        eig, vec = np.linalg.eigh(np.cov(x, rowvar=False))
        use = eig > 1.0e-7
        if not np.any(use):
            continue
        p = vec[:, use] / np.sqrt((eig[use] + 0.046) * np.count_nonzero(use))
        count = min(150, max(4, int(1.65 * np.sqrt(len(x)))), len(x))
        centers[t] = prototypes(x @ p, count, rng)
        projections[t] = p
    model = CompetitionKernel(
        groups,
        radii,
        centers,
        projections,
        1.0,
        np.zeros(sum(map(len, centers.values()))),
    )
    np.savez_compressed(WORK / "competition-basis.npz", **model.fields())
    print(
        "Competing-bond centers",
        model.size,
        "chemical triad groups",
        len(centers),
        flush=True,
    )
    return model


def batch(points):
    rows = [events(z, r, model.groups, model.radii) for z, r in points]
    types = np.concatenate([d[0] for d in rows])
    x = np.vstack([d[1] for d in rows])
    w = np.concatenate([d[2] for d in rows])
    owner = np.repeat(np.arange(len(points)), [len(d[0]) for d in rows])
    order = np.argsort(types, kind="stable")
    keys, starts, counts = np.unique(
        types[order], return_index=True, return_counts=True
    )
    out = [np.zeros((len(points), model.size)) for width in LENGTHS]
    for g, start, count in zip(keys, starts, counts):
        if g not in model.centers:
            continue
        ix = order[start : start + count]
        u = x[ix] @ model.projections[g]
        c = model.centers[g]
        d2 = np.maximum(
            0.0,
            np.einsum("ij,ij->i", u, u)[:, None]
            + np.einsum("ij,ij->i", c, c)
            - 2 * u @ c.T,
        )
        who, indices = np.unique(owner[ix], return_index=True)
        for width, mat in zip(LENGTHS, out):
            mat[who, columns[g]] = np.add.reduceat(
                w[ix, None] * np.exp(-0.5 * d2 / width**2), indices, axis=0
            )
    return out


def chunk(i):
    s = slice(i, min(i + 32, len(data)))
    for arr, m in zip(arrays, batch(data[s])):
        arr[s] = m


def cache(side, points):
    global arrays, data
    data = points
    paths = [WORK / f"competition-{w:g}-{side}.npy" for w in LENGTHS]
    if all(p.exists() for p in paths):
        return
    arrays = [
        np.lib.format.open_memmap(
            p.with_suffix(".pending"),
            mode="w+",
            dtype=float,
            shape=(len(points), model.size),
        )
        for p in paths
    ]
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        for j, _ in enumerate(
            pool.map(chunk, range(0, len(points), 32), buffersize=16)
        ):
            if (j + 1) % 3000 == 0:
                print(
                    "Encoded competing-bond geometries", side, (j + 1) * 32, flush=True
                )
    for arr, path in zip(arrays, paths):
        arr.flush()
        path.with_suffix(".pending").replace(path)
    print("Completed competing-bond features", side, len(points), flush=True)


def prepare():
    global model, columns
    gs, rows, classes = kt.inputs()
    model = learn(gs, rows, classes)
    columns = {}
    off = 0
    for k, c in model.centers.items():
        columns[k] = slice(off, off + len(c))
        off += len(c)
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    pts = [(gs[n].z, gs[n].r) for n in names]
    rng = np.random.default_rng(739)
    for i in rng.choice(len(pts), 10, replace=False):
        z, r = pts[i]
        v = rng.normal(size=r.shape)
        v /= np.linalg.norm(v)
        res = batch([(z, r), (z, r - 1.0e-5 * v), (z, r + 1.0e-5 * v)])
        for w, a in zip(LENGTHS, res):
            model.length = w
            features, jac = model.design(z, r, True)
            np.testing.assert_allclose(features, a[0], atol=1.0e-9)
            np.testing.assert_allclose(
                jac.T @ v.ravel(), (a[1] - a[2]) / 2.0e-5, atol=2.0e-6
            )
    print("Verified conservative exchange forces", flush=True)
    for _, z, r, v in trial_sites.hdf():
        pts.extend([(z, r), (z, r + 0.001 * v), (z, r - 0.001 * v)])
    cache("main", pts)
    aux, npriv, _ = trial_aux.read_frames()
    cache("rad", aux[npriv:])


def strengthen(mat, rhs, values, parent, meta, saddle, train):
    """Reweight true saddle curvatures by their signed wavenumber sensitivity.

    The secant to sign(k)*sqrt(abs(k)) accounts for the error caused by a
    positive curvature where the reference has a negative eigenvalue. A floor
    bounds the leverage of near-zero modes. Only training folds enter the solve.
    """
    from scipy.linalg import cho_factor, cho_solve
    from experiments.reactivity.frequency_validation import TO_CM1

    idx = np.flatnonzero(saddle & train & (meta[:, 2] < 0))
    x = (
        (values[idx, 3] - values[idx, 0]) + (values[idx, 4] - values[idx, 0])
    ) / 0.004**2
    k0 = (
        (parent[idx, 3] - parent[idx, 0]) + (parent[idx, 4] - parent[idx, 0])
    ) / 0.004**2
    target = meta[idx, 2]

    def freq(k):
        return TO_CM1 * np.sign(k) * np.sqrt(abs(k) / meta[idx, 4])

    truth = freq(target)
    weights = cho_solve(cho_factor(mat, check_finite=False), rhs, check_finite=False)
    for _ in range(3):
        guess = k0 + x @ weights
        slope = np.abs(freq(guess) - truth) / np.maximum(abs(guess - target), 1.0e-9)
        # 65 cm^-1 per eV/angstrom^2 is a typical magnitude near these modes.
        w = np.clip(slope / 65.0, 0.3, 3.0) * np.where(guess >= 0, 1.4, 1.0)
        a = x * w[:, None]
        weights = cho_solve(
            cho_factor(mat + a.T @ a, check_finite=False),
            rhs + a.T @ (w * (target - k0)),
            check_finite=False,
        )
    return weights


if __name__ == "__main__":
    from experiments.full_public.trial_curvature_fit import fit
    from experiments.full_public.trial_spectra import evaluate_trial

    with threadpool_limits(1):
        prepare()
    with threadpool_limits(5):
        fit(30)
    with threadpool_limits(1):
        evaluate_trial("p30-public")
