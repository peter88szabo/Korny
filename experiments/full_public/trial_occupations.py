"""Train and validate a sparse energy kernel using spin-resolved occupation spectra."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_sites, trial_aux
from experiments.full_public.trial_messages import prototypes
from experiments.reactivity.reaction_classes import lookup
from src.forcefield import ForceField
from src.occupied_orbitals import OccupationKernel, environments as populations
from src.orbital_bonds import environments as bond_populations

WORK = kt.WORK
WIDTHS = (0.74, 1.35)


def learn(gs, rows, classes, bonds):
    environments = bond_populations if bonds else populations
    backbone = ForceField.load(WORK / "p28-public-high.npz")
    groups = backbone.groups.copy()
    groups[1] = 0
    radii = backbone.radii
    classes.update(lookup())
    keys = trial_sites.candidates(gs, rows, classes)
    tables = {}
    rng = np.random.default_rng(10019)
    for key in keys:
        g = gs[key]
        raw = environments(g.z, g.r, *g.state, groups, radii)[0]
        for group in np.unique(groups[g.z]):
            tables.setdefault(int(group), []).append(raw[groups[g.z] == group])
    centers = {}
    proj = {}
    for group, items in sorted(tables.items()):
        x = np.vstack(items)
        if len(x) > 16000:
            x = x[rng.choice(len(x), 16000, replace=False)]
        if len(x) < 3:
            continue
        # Regulate whitening so the single-electron states retain their influence
        # without making a small error in an occupation dominate neighboring sites.
        scale = np.maximum(np.std(x, axis=0), 0.075)
        eig, axes = np.linalg.eigh(np.cov(x / scale, rowvar=False))
        keep = eig > 1.0e-7
        if not keep.any():
            continue
        p = (axes[:, keep] / scale[:, None]) / np.sqrt((eig[keep] + 0.12) * keep.sum())
        count = min(850, max(28, int(11 * np.sqrt(len(x)))), len(x))
        centers[group] = prototypes(x @ p, count, rng)
        proj[group] = p
    result = OccupationKernel(
        groups,
        radii,
        centers,
        proj,
        1.0,
        np.zeros(sum(map(len, centers.values()))),
        bonds,
    )
    np.savez_compressed(WORK / f"{family}-basis.npz", **result.fields())
    print(
        "Spin-resolved orbital centers",
        result.size,
        "by group",
        {k: len(c) for k, c in centers.items()},
        flush=True,
    )
    return result


def batch(pts):
    raw = np.vstack(
        [
            model.environments(z, r, q, m, model.groups, model.radii)[0]
            for z, r, q, m in pts
        ]
    )
    owner = np.repeat(np.arange(len(pts)), [len(z) for z, *_ in pts])
    types = np.concatenate([model.groups[z] for z, *_ in pts])
    result = [np.zeros((len(pts), model.size)) for _ in WIDTHS]
    offset = 0
    for g, c in model.centers.items():
        take = types == g
        if take.any():
            u = raw[take] @ model.projections[g]
            d2 = np.maximum(
                0.0, (u * u).sum(axis=1)[:, None] + (c * c).sum(axis=1) - 2 * u @ c.T
            )
            who, idx = np.unique(owner[take], return_index=True)
            for w, out in zip(WIDTHS, result):
                out[who, offset : offset + len(c)] = np.add.reduceat(
                    np.exp(-0.5 * d2 / w**2), idx, axis=0
                )
        offset += len(c)
    return result


def chunk(i):
    s = slice(i, min(i + 36, len(data)))
    for ar, x in zip(arrays, batch(data[s])):
        ar[s] = x


def cache(side, points):
    global arrays, data
    data = points
    paths = [WORK / f"{family}-{w:g}-{side}.npy" for w in WIDTHS]
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
            pool.map(chunk, range(0, len(points), 36), buffersize=16)
        ):
            if (j + 1) % 2000 == 0:
                print(
                    "Spin-resolved orbital geometries encoded",
                    side,
                    (j + 1) * 36,
                    flush=True,
                )
    for a, p in zip(arrays, paths):
        a.flush()
        p.with_suffix(".pending").replace(p)
    print("Completed orbital occupations", side, len(points), flush=True)


def prepare(bonds=False):
    global model, family
    family = "orbital-bond" if bonds else "occupied"
    gs, rows, classes = kt.inputs()
    model = learn(gs, rows, classes, bonds)
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    pts = [(gs[k].z, gs[k].r, *gs[k].state) for k in names]
    rng = np.random.default_rng(9874)
    for z, r, q, m in [
        next(p for p in pts if 3 < len(p[0]) < 12 and p[-1] == mult)
        for mult in (1, 2, 3)
    ] + [next(p for p in pts if p[-2] != 0 and 1 < len(p[0]) < 10)]:
        dr = rng.normal(size=r.shape)
        dr /= np.linalg.norm(dr)
        ans = batch(
            [(z, r, q, m), (z, r - 1.0e-5 * dr, q, m), (z, r + 1.0e-5 * dr, q, m)]
        )
        for w, x in zip(WIDTHS, ans):
            model.length = w
            value, jac = model.design_state(z, r, q, m, True)
            np.testing.assert_allclose(value, x[0], atol=2.0e-8)
            np.testing.assert_allclose(
                jac.T @ dr.ravel(), (x[1] - x[2]) / 2.0e-5, atol=4.0e-6
            )
    print(
        "Verified analytic orbital response for singlet, doublet, triplet and ion",
        flush=True,
    )
    for _, z, r, v, q, m in trial_sites.hdf(include_state=True):
        pts.extend([(z, r, q, m), (z, r + 0.001 * v, q, m), (z, r - 0.001 * v, q, m)])
    cache("main", pts)
    aux, npr, _ = trial_aux.read_frames()
    with (WORK / "radical-modes.pkl").open("rb") as f:
        states = [row[1][3] for row in pickle.load(f) for _ in range(7)]
    cache("rad", [(z, r, *st) for (z, r), st in zip(aux[npr:], states, strict=True)])


if __name__ == "__main__":
    from experiments.full_public.trial_curvature_fit import fit
    from experiments.full_public.trial_spectra import evaluate_trial

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bonds", action="store_true")
    bonds = parser.parse_args().bonds
    stage = 36 if bonds else 33
    with threadpool_limits(1):
        prepare(bonds)
    with threadpool_limits(4):
        fit(stage)
    with threadpool_limits(1):
        evaluate_trial(f"p{stage}-public")
