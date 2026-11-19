"""Fit a state-dependent residual on published reaction and barrier energies.

Only the high-level experimental checkpoint has this target. The PBE0 model,
Hirshfeld volume fit and runtime choice of dispersion remain independently usable.
"""
from collections import Counter
import argparse
import hashlib
from io import BytesIO
import json
from time import perf_counter

import numpy as np
from scipy.linalg import eigh

from experiments.reactivity.augmented import OUT
from experiments.reactivity.spin_data import reactions
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, score
from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.stationary_fit import POINTS
from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField

KINDS = ("forward", "reverse", "reaction")
TARGET = "BH9/TRIP50 electronic reaction and barrier differences, calibrated with PBE0-D3(BJ)"
BACKBONE = "pbe0-core-9325855e31063235.npz"


def partition(z):
    """Hold out complete TS formulas across sources, regardless of charge or spin."""
    key = ",".join(map(str, sorted(map(int, z))))
    fold = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return "test" if fold < 2 else "development" if fold < 4 else "train"


def shape_constraints(base, features, geoms, train_keys, states, *, auxiliary=False, held_formulas=()):
    """Enforce stationarity at training TSs/minima and preserve fitted equilibrium curvature.

    Two force projections per training geometry control memory. One follows the
    residual force, one probes an independent Cartesian direction. At NIST training
    minima, preserve the PBE0 backbone forces and Hessian responses more strongly.
    """
    rng = np.random.default_rng(391)
    blocks, targets = [], []
    def add(state, design, target):
        matrix = np.zeros((len(design), 1+len(states), features.size))
        matrix[:, 0] = design
        matrix[:, 1+states.index(state)] = design
        blocks.append(matrix.reshape(len(design), -1))
        targets.extend(target)
    for k in sorted(train_keys):
        g = geoms[k]
        force = base.predict(g.z, g.r, *g.state)[1].ravel()
        if np.linalg.norm(force) < 1e-8:
            continue
        dirs = np.vstack((force, rng.normal(size=len(force))))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        # Roughly 0.1 A motion from the stationary point in a projected direction.
        dirs *= .10
        add(g.state, dirs @ features.design(g.z, g.r)[1], -(dirs @ force))
    for point in json.loads(POINTS.read_text()).values():
        if point["split"] != "train":
            continue
        z = np.array([ELEMENTS[a] for a in point["atoms"]])
        xyz = np.array(point["positions_A"])
        f = features.design(z, xyz)[1]
        add((0, 1), 1.5*f, np.zeros(len(f)))
        for u in rng.normal(size=(4, *xyz.shape)):
            u /= np.linalg.norm(u)
            secant = (features.design(z, xyz + .002*u)[1] -
                      features.design(z, xyz - .002*u)[1]) / .004
            add((0, 1), .30*secant, np.zeros(len(secant)))
    sources=[]
    if auxiliary:
        from experiments.reactivity.cluster_forces import projected_rows
        from experiments.reactivity.radical_curvatures import curvature_samples
        from experiments.reactivity.stationary_fit import curvature_rows, internal_transform
        from copy import copy
        bare = copy(base)
        bare.correction = None
        cluster, _, _ = projected_rows(bare, features, POINTS.parents[1]/"QCell/qcell_ions_water.h5")
        secants, _ = curvature_rows(bare, features)
        def compressed_rows(data, weight):
            # These readers historically use positive valence gradients, whereas
            # this feature object returns negative energy derivatives. Negate RHS.
            active = np.any(data!=0.,axis=0)
            small = np.linalg.qr(data[:,active], mode="r")
            rows = np.zeros((len(small), data.shape[1]))
            rows[:,active] = small
            add((0,1), weight * rows[:,:-1], -weight * rows[:,-1])
        compressed_rows(cluster["train"], .45)
        compressed_rows(secants, .15)
        for name,z,r,state,h in curvature_samples(held_formulas):
            sources.append(name)
            # Remove rigid motions before selecting unstable, bending and stretching
            # directions. Orthogonal Cartesian vectors keep all labels in eV/A^2.
            u = internal_transform(np.ones(len(z)),r)
            _,v = np.linalg.eigh(u.T @ h @ u)
            modes = (u @ v).T
            stiff, rhs = [], []
            for mode in modes[[0,len(modes)//2,-1]]:
                step = .001*mode.reshape(r.shape)
                fp = features.design(z,r+step)[1]
                fm = features.design(z,r-step)[1]
                base_diff = (base.predict(z,r+step,*state)[1]-base.predict(z,r-step,*state)[1]).ravel()
                stiff.append(mode @ (fm-fp)/.002)
                rhs.append(mode @ h @ mode + mode @ base_diff/.002)
            add(state, .18*np.array(stiff), .18*np.array(rhs))
            f = base.predict(z,r,*state)[1].ravel()
            add(state, .12*modes[[0,-1]] @ features.design(z,r)[1], -.12*modes[[0,-1]] @ f)
    return np.vstack(blocks), np.array(targets), sources


def fit(*, shape=True, auxiliary=False, backbone=BACKBONE, out="electronic_fit.json"):
    start = perf_counter()
    base = ForceField.load(OUT / backbone, dispersion="d3")
    geoms, records, excluded = reactions(base.charge_offsets)
    splits = [partition(geoms[row.ts].z) for row in records]
    train_keys = {key for r, s in zip(records, splits) if s == "train"
                  for key in (r.ts, *r.products, *r.reactants)}
    states = sorted({geoms[k].state for k in train_keys})
    # A zero state head exposes the common prior on the recorded physical state.
    # It does not supply fictitious closed-shell labels for odd-electron systems.
    features = ElectronicFeatures(base.pair.species, environment=shape)
    prior_weights = np.zeros((len(states), features.size))
    if base.electronic is not None:
        if base.electronic.features.environment != shape:
            raise ValueError("The state basis must match the bare PBE0 checkpoint")
        prior_weights = np.array([base.electronic.weights[k] for k in states])
    base.electronic = ElectronicCorrection(base.pair.species, states, prior_weights, base.target,
                                           environment=shape)
    cache = {}
    counts = Counter()
    def state(k):
        if k not in cache:
            g = geoms[k]
            cache[k] = (base.energy(g.z, g.r, *g.state), features.design(g.z, g.r)[0])
        return cache[k]
    x, y, baseline, meta = [], [], [], []
    for row, split in zip(records, splits, strict=True):
        if any(geoms[k].state not in states for k in (row.ts, *row.reactants, *row.products)):
            counts["state_absent_from_train"] += 1
            continue
        counts[split] += 1
        for kind, eq, ref in zip(KINDS, row.equations(), row.reference, strict=True):
            blocks = np.zeros((len(states)+1, features.size))
            prior = 0.
            for k, c in eq:
                value, phi = state(k)
                prior += c * value
                blocks[0] += c * phi
                blocks[1+states.index(geoms[k].state)] += c * phi
            x.append(blocks.ravel())
            baseline.append(prior)
            y.append(ref / EV_TO_KCAL - prior)
            meta.append(dict(name=row.name, family=row.family, split=split, kind=kind,
                             state=geoms[row.ts].state, reference=ref))
    x, y = np.array(x), np.array(y)
    held_formulas = {tuple(sorted(geoms[k].z)) for r,s in zip(records,splits) if s!="train"
                     for k in [r.ts,*r.reactants,*r.products]}
    constraints, target, auxiliary_sources = (shape_constraints(base, features, geoms, train_keys, states,
        auxiliary=auxiliary,held_formulas=held_formulas) if shape else (np.zeros((0,x.shape[1])),np.empty(0),[]))
    prior_seconds = perf_counter()-start
    print("reaction groups", counts, "state combinations", states, "feature seconds", prior_seconds, flush=True)
    train = np.array([m["split"]=="train" for m in meta])
    dev = np.array([m["split"]=="development" for m in meta])
    def solve(relative_scale):
        # Shared shape coefficients transfer between electronic states. A separate
        # block for each observed state learns residual changes in bond energetics.
        scale = np.r_[np.ones(features.size), np.full(len(states)*features.size, relative_scale)]
        a = np.vstack((x[train], constraints)) * scale
        b = np.r_[y[train], target]
        # Solve on active columns only. Reuse the spectrum across all ridge values.
        active = np.any(a != 0., axis=0)
        aa = a[:, active]
        if len(aa.T) < len(aa):
            vals, vecs = eigh(aa.T @ aa)
            proj = vecs.T @ (aa.T @ b)
            left = vecs
        else:
            vals, vecs = eigh(aa @ aa.T)
            proj = vecs.T @ b
            left = aa.T @ vecs
        trials = []
        for ridge in (.0003, .003, .03, .3, 3., 30.):
            w = np.zeros(x.shape[1])
            w[active] = (left @ (proj/(np.maximum(vals, 0)+ridge))) * scale[active]
            loss = np.abs(x[dev] @ w-y[dev]).mean()
            trials.append((loss, ridge, w))
        return min(trials, key=lambda t:t[0])
    blind_loss, blind_ridge, blind = solve(0.)
    candidates = [(s, *solve(s)) for s in (.25, .75, 1.5)]
    coupling, _, ridge, w = min(candidates, key=lambda t:t[1])
    weights = w.reshape(len(states)+1, features.size)
    base.electronic = ElectronicCorrection(base.pair.species, states, prior_weights+weights[1:]+weights[0], TARGET, environment=shape)
    stream = BytesIO()
    base.save(stream)
    data = stream.getvalue()
    name = "electronic-"+hashlib.sha256(data).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(data)
    elapsed = perf_counter()-start
    for m, b, db, ds in zip(meta, baseline, x@blind, x@w, strict=True):
        for key, val in (("backbone", b), ("state_blind_fit", b+db), ("state_fit", b+ds)):
            m[key] = float(val*EV_TO_KCAL)
    metrics = {}
    for split in ("train","development","test"):
        metrics[split] = {}
        for m in (0,1,2,3):
            group = {}
            for kind in ("reaction","barrier"):
                subset = [r for r in meta if r["split"]==split and (not m or m==r["state"][1])
                          and (r["kind"]=="reaction")==bool(kind=="reaction")]
                if subset:
                    group[kind] = {k:score([r[k]-r["reference"] for r in subset])
                                   for k in ("backbone", "state_blind_fit", "state_fit")}
            metrics[split][str(m) if m else "all"] = group
    report = dict(checkpoint=name, backbone=backbone, calibration_target=TARGET,
                  states=states, counts=counts, excluded=excluded,
                  split="sha256 of full TS composition; common small reagents can recur",
                  auxiliary_radical_hessians=auxiliary_sources, shape_constraints=len(constraints), fit_seconds=elapsed, feature_seconds=prior_seconds, ridge=ridge,
                  state_feature_scale=coupling, blind_ridge=blind_ridge, metrics=metrics,
                  train_geometries_by_state=dict(Counter(str(geoms[k].state) for k in train_keys)),
                  fixed_singlet_panel=assess(base), rows=meta)
    (OUT/out).write_text(json.dumps(report, indent=2)+"\n")
    print("test", metrics["test"]["all"], "checkpoint", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--out", default="electronic_fit.json")
    parser.add_argument("--auxiliary-curvature", action="store_true")
    args=parser.parse_args()
    fit(shape=not args.energy_only, auxiliary=args.auxiliary_curvature, backbone=args.backbone, out=args.out)
