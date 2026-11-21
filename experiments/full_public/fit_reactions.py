"""Fit the complete requested reaction corpus and score the held-out equations.

Coefficients are fitted in three stages with distinct, explicitly labelled
checkpoints. All share an energy-conserving coordinate representation. The
higher-level corrections start at the fitted PBE0 solution. Checkpoints store
all parameters needed for prediction and can be opened with ForceField.load.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from time import monotonic

import numpy as np
from rdkit import Chem
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public.data import fingerprint, load_reaction_data
from experiments.full_public.specified import OUT
from experiments.reactivity.matern_fit import basis
from experiments.reactivity.reaction_classes import lookup
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from experiments.reactivity.stationary_fit import POINTS, MASSES
from experiments.reactivity.larger_vibrations import references
from src.reaction_surface import ReactionSurface
from src.forcefield import ForceField
from src.dispersion.d3 import d3_energy_forces

BASE = OUT / "pbe0-stationary-core.npz"


def precompute(i):
    g = geoms[unique[i]]
    z, r = g.z, g.r
    q, m = g.state
    v = surface.design(z, r, q, m)[0]
    energy = surface.prior(z, r, q, m)[0]
    dispersion = d3_energy_forces(z, r)[0] if len(z) > 1 else 0.0
    return i, v, energy, dispersion


def basis_seeds(rows, geoms, classes):
    """Allocate sparse-center candidates across reaction classes without changing the training set."""
    pool = defaultdict(set)
    for target, source, split, name, eq, y, kind in rows:
        if split != "train":
            continue
        family = classes.get(name, source)
        if kind == "forward":
            keys = [eq[-1][0]]
        elif kind in ("atomization", "energy_difference"):
            keys = [k for k, c in eq]
        else:
            continue
        for k in keys:
            if set(geoms[k].z) <= set((1, 6, 7, 8, 9, 15, 16, 17)):
                pool[family].add(k)
    for family, keys in sorted(pool.items()):
        # The fixed sparse basis controls computational cost; every labelled
        # training equation enters the later regression, irrespective of class.
        for k in sorted(keys, key=lambda k: hashlib.sha256(k.encode()).digest())[:24]:
            yield geoms[k]


def locks(surface, used):
    """Preserve first and second derivatives at the NIST training minima.

    A force at an equilibrium and the change in force under a displacement
    constrain different derivatives of the same potential.
    """
    pts = json.loads(POINTS.read_text())
    pts.update(references())
    locks = []
    for name in used:
        p = pts[name]
        z = np.array([MASSES[s][0] for s in p["atoms"]])
        r = np.array(p["positions_A"])
        j = surface.design(z, r, 0, 1, True)[1]
        locks.append(2 * j / np.sqrt(r.size))
        for d in np.eye(r.size).reshape(-1, *r.shape):
            a = surface.design(z, r + 0.001 * d, 0, 1, True)[1]
            b = surface.design(z, r - 0.001 * d, 0, 1, True)[1]
            locks.append(0.12 * (a - b) / (0.002 * r.size))
    return np.vstack(locks)


def run(joint=False):
    global geoms, unique, surface
    start = monotonic()
    geoms, rows, classes = load_reaction_data()
    base = ForceField.load(BASE)
    classes.update(lookup(include_trip50=False))
    kernel = (
        ForceField.load(OUT / "pbe0-reaction.npz").kernel
        if joint
        else basis(basis_seeds(rows, geoms, classes), base.pair.species, max_centers=64)
    )
    kernel.family = "double_matern52"
    pt = Chem.GetPeriodicTable()
    elements = sorted(
        {int(z) for g in geoms.values() for z in g.z} | set(base.pair.species)
    )
    radii = np.array([0.0] + [pt.GetRcovalent(s) for s in range(1, 119)])
    groups = np.array([0] + [min(8, pt.GetNOuterElecs(s)) for s in range(1, 119)])
    states = sorted({g.state for g in geoms.values()} | set(base.supported_states))
    surface = ReactionSurface(base, kernel, elements, radii, groups, states, [], "PBE0")
    if joint:
        from itertools import combinations_with_replacement
        from src.repulsion import NuclearRepulsion

        with np.load(OUT / "short-range-pairs.npz") as d:
            pairs = {
                tuple(p): v for p, v in zip(d["pairs"], d["coefficients"], strict=True)
            }
        surface.extension_core = NuclearRepulsion(
            elements,
            [pairs[p] for p in combinations_with_replacement(elements, 2)],
            radii,
        )
    unique = []
    indices = {}
    by_fp = {}
    for k, g in geoms.items():
        fp = (fingerprint(g.z, g.r), g.state)
        if fp not in by_fp:
            by_fp[fp] = len(unique)
            unique.append(k)
        indices[k] = by_fp[fp]
    g = geoms[unique[0]]
    dim = len(surface.design(g.z, g.r, *g.state)[0])
    x = np.empty((len(unique), dim))
    p = np.empty(len(unique))
    d3 = np.empty(len(unique))
    print("feature assembly", len(unique), "geometries", dim, "features", flush=True)
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("fork")) as pool:
        for i, v, e, d in pool.map(
            precompute, range(len(unique)), chunksize=32, buffersize=12
        ):
            x[i] = v
            p[i] = e
            d3[i] = d
            if (i + 1) % 5000 == 0:
                print(
                    "described",
                    i + 1,
                    "of",
                    len(unique),
                    "elapsed",
                    round(monotonic() - start),
                    flush=True,
                )
    anchors = locks(
        surface, json.loads((OUT / "stationary.json").read_text())["training"]
    )
    print("NIST derivative constraints", len(anchors), flush=True)
    prefix = "joint-" if joint else ""
    full_data = np.load(OUT / "joint-gram.npy", mmap_mode="r") if joint else None
    geometry_data = (
        np.load(OUT / "nist-derivative-gram.npy", mmap_mode="r") if joint else None
    )
    checkpoints = {}
    scores = {}
    pbe_weights = np.zeros(dim)
    for target in ("pbe0", "high", "wb97xd3"):
        which = [row for row in rows if row[0] == target]
        groups = Counter(r[1] for r in which if r[2] == "train")
        prior = pbe_weights.copy()
        energies = p + (d3 if target != "pbe0" else 0.0)
        gram = np.zeros((dim, dim))
        rhs = np.zeros(dim)
        cache = []

        def flush():
            if not cache:
                return
            a, b = zip(*cache)
            a = np.array(a)
            b = np.array(b)
            gram[:] += a.T @ a
            rhs[:] += a.T @ b
            cache.clear()

        def equation(terms, ref):
            ii = np.array([indices[k] for k, _ in terms])
            nu = np.array([c for _, c in terms])
            return nu @ x[ii], ref - nu @ energies[ii]

        with threadpool_limits(limits=3, user_api="blas"):
            for _, src, split, name, eq, ref, kind in which:
                if split != "train":
                    continue
                a, y = equation(eq, ref)
                # Balance the supervision from large and small collections;
                # forward, reverse and reaction energies keep their physical units.
                w = min(1.0, np.sqrt(4500 / groups[src]))
                cache.append((w * a, w * (y - a @ prior)))
                if len(cache) >= 512:
                    flush()
            flush()
            gram += anchors.T @ anchors
            rhs -= anchors.T @ (anchors @ prior)
            if joint:
                # Keep the entire HDF5 objective in the last bare-PBE0 fit.
                # On the distinct reference levels, anchor changes to the PBE0
                # surface over the same training distribution, without using
                # PBE0 energies as if they were coupled-cluster observations.
                if target == "pbe0":
                    gram += full_data[:-1, :-1]
                    rhs += full_data[:-1, -1]
                else:
                    gram += 0.15 * full_data[:-1, :-1]
                if target == "wb97xd3":
                    hd = np.load(OUT / "radical-derivative-gram.npy", mmap_mode="r")
                    gram += hd[:-1, :-1]
                    rhs += hd[:-1, -1] - hd[:-1, :-1] @ prior
                if target == "pbe0":
                    gram += geometry_data[:-1, :-1]
                    rhs += geometry_data[:-1, -1]
            gram.flat[:: dim + 1] += 0.08
            coeff = prior + cho_solve(cho_factor(gram), rhs)
        if target == "pbe0":
            pbe_weights = coeff
        surface.prediction_target = {
            "pbe0": "PBE0",
            "high": "high-level electronic references, calibrated with PBE0-D3(BJ)",
            "wb97xd3": "wB97X-D3 electronic references, calibrated with PBE0-D3(BJ)",
        }[target]
        surface.weights = coeff
        filename = OUT / f"{prefix}{target}-reaction.npz"
        surface.save_with_base(BASE, filename)
        disp = "off" if target == "pbe0" else "d3"
        loaded = ForceField.load(filename, dispersion=disp)
        # Exercise the real prediction interface and check force signs against
        # a centered difference in a deterministic displacement.
        g = next(g for g in geoms.values() if len(g.z) == 8 and g.state == (0, 1))
        en, force = loaded.predict(g.z, g.r, *g.state)
        direction = np.random.default_rng(73).normal(size=g.r.shape)
        direction /= np.linalg.norm(direction)
        step = 0.00005
        fd = (
            loaded.energy(g.z, g.r - step * direction, *g.state)
            - loaded.energy(g.z, g.r + step * direction, *g.state)
        ) / (2 * step)
        np.testing.assert_allclose(fd, (force * direction).sum(), rtol=2e-5, atol=5e-5)
        pred = energies + x @ coeff
        bins = defaultdict(list)
        for _, src, split, name, terms, ref, kind in which:
            if split == "validation":
                err = (sum(c * pred[indices[k]] for k, c in terms) - ref) * EV_TO_KCAL
                bins[src, kind].append(err)
        table = {
            "/".join(key): dict(
                n=len(a),
                mae=float(np.abs(a).mean()),
                rmse=float(np.sqrt(np.mean(np.square(a)))),
            )
            for key, a in sorted(bins.items())
        }
        scores[target] = table
        checkpoints[target] = filename.name
        (OUT / (prefix + "reaction-validation.json")).write_text(
            json.dumps(
                dict(
                    seconds=monotonic() - start,
                    unique_geometries=len(unique),
                    features=dim,
                    checkpoints=checkpoints,
                    validation=scores,
                ),
                indent=2,
            )
            + "\n"
        )
        print("fit and validated", target, table, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", action="store_true")
    run(parser.parse_args().joint)
