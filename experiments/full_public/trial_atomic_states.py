"""Fit reaction energies using locally predicted electronic properties and their environments.

The atom-property predictor and Gaussian centers use training structures only.
The second stage fits energies on the existing physical and reaction constraints.
Both fitted widths use the same centers. Charge and multiplicity accompany every
geometry including displaced HDF5 and open-shell reference structures.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import (
    kernel_trials as kt,
    trial_sites,
    trial_aux,
    trial_periodic,
)
from experiments.full_public.trial_charge_teacher import load_teacher, projected
from experiments.full_public.trial_messages import prototypes
from experiments.reactivity.reaction_classes import lookup
from src.atomic_states import AtomicStateKernel

WORK = kt.WORK
LENGTHS = (0.8, 1.5)


def environments(teacher, z, r, q, m):
    u = projected(teacher, z, r)
    env = teacher.context(z, r, q, m, u)[0]
    groups = teacher.periodic.groups[z]
    return {g: (v, env[groups == g]) for g, v in u.items()}


def learn(gs, rows, classes):
    teacher = load_teacher()
    classes.update(lookup())
    keys = set(trial_sites.candidates(gs, rows, classes))
    with (WORK / "atom-state-labels.pkl").open("rb") as f:
        keys.update(
            k
            for k, split, *_ in pickle.load(f)
            if split == "fit" and k.startswith("Grambow/")
        )
    rng = np.random.default_rng(4598)
    local = {}
    for key in sorted(keys):
        g = gs[key]
        for group, row in environments(teacher, g.z, g.r, *g.state).items():
            local.setdefault(group, []).append(row)
    centers = {}
    projections = {}
    for g, values in sorted(local.items()):
        u = np.vstack([a for a, b in values])
        field = np.vstack([b for a, b in values])
        if len(u) > 16000:
            idx = rng.choice(len(u), 16000, replace=False)
            u, field = u[idx], field[idx]
        eig, axes = np.linalg.eigh(np.cov(field, rowvar=False))
        # A floor keeps small electronic-state variations at unseen chemistries bounded.
        keep = eig > 1.0e-6
        proj = axes[:, keep] / np.sqrt((eig[keep] + 0.035) * max(1, keep.sum()))
        projections[g] = proj
        sites = np.column_stack((0.65 * u, field @ proj))
        count = min(950, len(sites), max(32, int(12 * np.sqrt(len(sites)))))
        centers[g] = prototypes(sites, count, rng)
    if set(centers) != set(teacher.coefficients):
        raise ValueError("Training center groups do not match property groups")
    model = AtomicStateKernel(
        teacher, centers, projections, 1.0, np.zeros(sum(map(len, centers.values())))
    )
    np.savez_compressed(WORK / "state-basis.npz", **model.fields())
    print(
        "Atomic-state basis centers",
        {g: len(v) for g, v in centers.items()},
        flush=True,
    )
    return model


def batch(points):
    rows = [environments(model.teacher, *p) for p in points]
    result = [np.zeros((len(points), model.size)) for _ in LENGTHS]
    start = 0
    for g, c in model.centers.items():
        indices = [i for i, d in enumerate(rows) if g in d]
        if indices:
            u = np.vstack([rows[i][g][0] for i in indices])
            field = np.vstack([rows[i][g][1] for i in indices])
            sites = np.column_stack((0.65 * u, field @ model.projections[g]))
            offsets = np.r_[0, np.cumsum([len(rows[i][g][0]) for i in indices])[:-1]]
            d2 = np.maximum(
                0.0,
                (sites * sites).sum(axis=1)[:, None]
                + (c * c).sum(axis=1)
                - 2 * sites @ c.T,
            )
            for w, out in zip(LENGTHS, result):
                out[indices, start : start + len(c)] = np.add.reduceat(
                    np.exp(-0.5 * d2 / w**2), offsets, axis=0
                )
        start += len(c)
    return result


def compute_chunk(i):
    s = slice(i, min(i + 36, len(data)))
    for a, x in zip(arrays, batch(data[s])):
        a[s] = x


def cache(side, points):
    global data, arrays
    data = points
    paths = [WORK / f"state-{w:g}-{side}.npy" for w in LENGTHS]
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
        for i, _ in enumerate(
            pool.map(compute_chunk, range(0, len(points), 36), buffersize=20)
        ):
            if (i + 1) % 2000 == 0:
                print("State-aware geometries encoded", side, (i + 1) * 36, flush=True)
    for arr, path in zip(arrays, paths):
        arr.flush()
        path.with_suffix(".pending").replace(path)
    print("Completed atomic-state features", side, len(points), flush=True)


def prepare():
    global model
    trial_periodic.setup()
    gs, rows, classes = kt.inputs()
    model = learn(gs, rows, classes)
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    points = [(gs[k].z, gs[k].r, *gs[k].state) for k in names]
    rng = np.random.default_rng(379)
    for z, r, q, m in [
        next(p for p in points if p[-2] != 0 and len(p[0]) > 1),
        next(p for p in points if p[-1] == 2 and len(p[0]) > 1),
        *points[36:39],
    ]:
        v = rng.normal(size=r.shape)
        v /= np.linalg.norm(v)
        arr = batch(
            [(z, r, q, m), (z, r - 1.0e-5 * v, q, m), (z, r + 1.0e-5 * v, q, m)]
        )
        for a, w in zip(arr, LENGTHS):
            model.length = w
            x, j = model.design_state(z, r, q, m, True)
            np.testing.assert_allclose(a[0], x, atol=5.0e-8)
            np.testing.assert_allclose(
                (a[1] - a[2]) / 2.0e-5, j.T @ v.ravel(), atol=3.0e-6
            )
    print(
        "Analytic forces verified on charged, radical and neutral structures",
        flush=True,
    )
    for _, z, r, v, q, m in trial_sites.hdf(include_state=True):
        points.extend(
            [(z, r, q, m), (z, r + 0.001 * v, q, m), (z, r - 0.001 * v, q, m)]
        )
    cache("main", points)
    with (WORK / "radical-modes.pkl").open("rb") as f:
        states = [entry[1][3] for entry in pickle.load(f) for _ in range(7)]
    aux, nprivate, _ = trial_aux.read_frames()
    cache(
        "rad",
        [(z, r, *state) for (z, r), state in zip(aux[nprivate:], states, strict=True)],
    )


if __name__ == "__main__":
    from experiments.full_public.trial_curvature_fit import fit
    from experiments.full_public.trial_spectra import evaluate_trial

    with threadpool_limits(1):
        prepare()
    with threadpool_limits(5):
        fit(28)
    with threadpool_limits(1):
        evaluate_trial("p28-public")
