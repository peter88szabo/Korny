"""Direct electronic-reference training and validation for the H-transfer bases.

Cache parent energies on the archived geometries. Differences of these scalar
predictions and of the new features use the same forward/reverse orientation.
Each auxiliary Hamiltonian can have its own correction around a shared vector.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sqlite3

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import DATA
from experiments.full_public.trial_h_transfer import (
    ROOT,
    ANGULAR,
    STAGES,
    bases,
    dual_features,
)
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.irc_learning_curve import direction
from src.forcefield import ForceField

PAPER = DATA / "paper_references"
RADICAL = IRC.parent / "multiscale-path67"
FAMILIES = tuple(STAGES)


def setup(angular=False):
    global models, backbone, operators
    models = bases(angular)
    backbone = {
        t: ForceField.load(kt.WORK / f"p68-public-{t}.npz", dispersion="d3")
        for t in kt.TARGETS
    }
    operators = {}
    for f, (p, _) in STAGES.items():
        for t in kt.TARGETS:
            parent = ForceField.load(kt.WORK / f"p{p}-public-{t}.npz", dispersion="d3")
            operators[f, t] = parent.additional_bases[-1]


def parents_at(z, r, q, m, target):
    """The two parents share p68 and differ by one geometry-only energy basis."""
    e = backbone[target].energy(z, r, q, m)
    return np.array(
        [
            e + b.design(z, r)[0] @ b.weights
            for b in (operators[f, target] for f in FAMILIES)
        ]
    )


def publish(path, **kwargs):
    temp = path.with_suffix(".pending")
    with temp.open("wb") as f:
        np.savez(f, **kwargs)
    temp.replace(path)


def path_batch(start):
    path = output_dir / f"rdb19-{start:05d}.npz"
    if path.exists():
        return
    obs = []
    cached = None
    if output_dir == ANGULAR:
        with np.load(ROOT / path.name) as d:
            cached = d["rows"][:, :, :, -1]
    for idx, rec in enumerate(records[start : start + 24]):
        rx = rec["rxn"]
        sides = []
        for name in ("forw", "rev"):
            with np.load(IRC / f"{rx}-{name}.npz") as f:
                z = f["z"].astype(int)
                coords = f["r_A"]
                ener = f["energy_eV"]
                sides.append((coords[-1].astype(float), float(ener[-1])))
                if name == "forw":
                    ts = coords[0].astype(float), float(ener[0])
        # Native endpoint matching sets the direction of reported barriers.
        # Training gap and midpoint squared errors are orientation-independent.
        orientation = direction(rx) if rec["heldout"] else (0, 2)
        if orientation is None:
            raise ValueError(f"Missing held endpoint assignment {rx}")
        if orientation[0] != 0:
            sides.reverse()
        points = [sides[0], ts, sides[1]]
        x = dual_features([(z, r) for r, e in points], models)
        y = np.array([e for r, e in points])
        residual = (
            (
                y[:, None]
                - np.array([parents_at(z, r, 0, 2, "wb97xd3") for r, e in points])
            ).T
            if cached is None
            else cached[idx]
        )
        obs.append(
            [np.column_stack((x[f], residual[j])) for j, f in enumerate(FAMILIES)]
        )
    publish(path, rows=np.array(obs))


def path_cache(angular=False):
    global records, output_dir
    output_dir = ANGULAR if angular else ROOT
    output_dir.mkdir(exist_ok=True)
    setup(angular)
    records = json.loads((RADICAL / "snapshot.json").read_text())
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(
            pool.map(path_batch, range(0, len(records), 24), buffersize=12), 1
        ):
            if i % 50 == 0:
                print("RDB19-Rad paths encoded", i * 24, flush=True)
    print("Completed RDB19-Rad", len(records), "doublet paths", flush=True)


def paper_records():
    with sqlite3.connect(f"file:{PAPER/'electronic.sqlite'}?mode=ro", uri=True) as db:
        query = """select id,source,system,series,method,split,charge,mult,z,r,energy_eV
                 from geometries where duplicate_of is null and mult in (1,2)
                 and split in ('train','validation') order by id"""
        rec = {
            n: (
                src,
                sys,
                series,
                method,
                split,
                q,
                m,
                np.frombuffer(z, np.uint8),
                np.frombuffer(r, "<f8").reshape(-1, 3),
                e,
            )
            for n, src, sys, series, method, split, q, m, z, r, e in db.execute(query)
        }
        links = [
            (src, sys, meth, case, [a], t, [p] if p else [])
            for src, sys, meth, case, a, t, p in db.execute(
                "select * from channels order by source,system,method,channel"
            )
        ]
    frag = json.loads((PAPER / "fragment-channels.json").read_text())
    for channel in frag["channels"]:
        a, t, p = channel["reactant"], channel["ts"], channel["product"]
        method, split, sys = channel["method"], channel["split"], channel["system"]
        for n in [*a, t, *p]:
            if n in rec:
                continue
            g = frag["geometries"].get(n)
            if g and g["mult"] in (1, 2):
                rec[n] = (
                    "AutoMeKin",
                    sys,
                    "fragment",
                    method,
                    split,
                    g["charge"],
                    g["mult"],
                    np.array(g["z"], int),
                    np.array(g["r_A"]),
                    g["energy_eV"],
                )
        links.append(("AutoMeKin fragments", sys, method, channel["channel"], a, t, p))
    return rec, links


def paper_target(method):
    return "high" if "CCSD" in method or method.startswith("BASChem") else "wb97xd3"


def make_delta_plans():
    """Precompute coefficient differences for the exact p42 → p68 energy update."""
    old = {
        t: ForceField.load(kt.WORK / f"p42-public-{t}.npz", dispersion="d3")
        for t in kt.TARGETS
    }
    plans = {}
    for t, m68 in backbone.items():
        m42 = old[t]
        for spin in (1, 2):
            old_w, old_extra = m42.spin_readouts.get(spin, (m42.weights, {}))
            new_w, new_extra = m68.spin_readouts.get(spin, (m68.weights, {}))
            main = new_w - old_w
            size = m68.spin_size_readouts.get(spin)
            extra = []
            for i, b in enumerate(m68.additional_bases):
                w = new_extra.get(i, b.weights).copy()
                if i < len(m42.additional_bases):
                    w -= old_extra.get(i, m42.additional_bases[i].weights)
                if np.any(w):
                    extra.append((b, w))
            plans[t, spin] = (main, size, extra)
    return plans


def paper_parent(n, z, r, q, m, t):
    """Reuse archived p42 energies where its state and reference branch match."""
    cached_t = (
        "high"
        if "CCSD" in paper_meta[n][3]
        or m == 1
        and paper_meta[n][3].startswith("BASChem")
        else "wb97xd3"
    )
    if n not in old_index or t != cached_t:
        return parents_at(z, r, q, m, t)
    e = old_energy[old_index[n]]
    main, size, extra = plans[t, m]
    if size is not None or np.any(main):
        w = main
        if size is not None:
            scale, coef = size
            w = w + coef * (scale / (scale + np.count_nonzero(z > 1)))
        e += backbone[t].design(z, r, q, m)[0] @ w
    for b, w in extra:
        x = (
            b.design_state(z, r, q, m)[0]
            if hasattr(b, "design_state")
            else b.design(z, r)[0]
        )
        e += x @ w
    return np.array(
        [
            e + operators[f, t].design(z, r)[0] @ operators[f, t].weights
            for f in FAMILIES
        ]
    )


def paper_batch(lo):
    path = output_dir / f"paper-{lo:05d}.npz"
    if path.exists():
        return
    keys = paper_names[lo : lo + 48]
    points = [paper_meta[n][-3:-1] for n in keys]
    x = dual_features(points, models)
    if output_dir == ANGULAR:
        with np.load(ROOT / path.name) as d:
            np.testing.assert_array_equal(d["keys"], keys)
            publish(path, keys=keys, base=d["base"], **x)
        return
    base = []
    for n in keys:
        _, _, _, method, _, q, m, z, r, _ = paper_meta[n]
        base.append(paper_parent(n, z, r, q, m, paper_target(method)))
    publish(path, keys=keys, base=base, **x)


def paper_cache(angular=False):
    global paper_meta, paper_names, plans, old_index, old_energy, output_dir
    output_dir = ANGULAR if angular else ROOT
    setup(angular)
    output_dir.mkdir(exist_ok=True)
    paper_meta, _ = paper_records()
    paper_names = sorted(paper_meta)
    plans = make_delta_plans()
    d = PAPER / "features42"
    old_index = json.loads((d / "index.json").read_text())
    old_energy = np.concatenate(
        [np.load(p)["base"] for p in sorted(d.glob("block-*.npz"))]
    )
    for source in ("AutoMeKin", "AIMNet2-NSE supplement", "BASChem19"):
        for spin in (1, 2):
            n = next(
                (n for n, r in paper_meta.items() if r[0] == source and r[6] == spin),
                None,
            )
            if n is None:
                continue
            _, _, _, m, _, q, s, z, r, _ = paper_meta[n]
            t = paper_target(m)
            np.testing.assert_allclose(
                paper_parent(n, z, r, q, s, t),
                parents_at(z, r, q, s, t),
                rtol=0,
                atol=5e-6,
            )
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(
            pool.map(paper_batch, range(0, len(paper_names), 48), buffersize=8), 1
        ):
            if i % 80 == 0:
                print("Paper source geometries encoded", i * 48, flush=True)
    print(
        "Completed AutoMeKin, AIMNet2-NSE supplement and BASChem19",
        len(paper_names),
        flush=True,
    )


def external_rows(family, directory=ROOT):
    """Build native gap, midpoint, profile and forward-only equations in eV."""
    family_index = FAMILIES.index(family)
    train = defaultdict(list)
    held = defaultdict(list)
    counts = defaultdict(Counter)

    def add(name, target, split, a, t, p=None, case=""):
        counts[name][split] += 1
        forward = t - a
        if p is None:
            terms = [(forward, 64.0)]
        else:
            terms = [(p - a, 36.0), (t - (a + p) * 0.5, 96.0)]
        if split == "train":
            train[target, name].extend(terms)
        else:
            held[target, name].append((case, forward, p - a if p is not None else None))

    records = json.loads((RADICAL / "snapshot.json").read_text())
    for start in range(0, len(records), 24):
        with np.load(directory / f"rdb19-{start:05d}.npz") as pack:
            triples = pack["rows"][:, family_index]
        for rec, (a, t, p) in zip(
            records[start : start + len(triples)], triples, strict=True
        ):
            add(
                "RDB19-Rad",
                "wb97xd3",
                "validation" if rec["heldout"] else "train",
                a,
                t,
                p,
                (rec["rxn"], rec["family"]),
            )
    meta, links = paper_records()
    geoms = {}
    for path in sorted(directory.glob("paper-*.npz")):
        with np.load(path) as d:
            ids = d["keys"]
            residual = np.array([meta[n][-1] for n in ids]) - d["base"][:, family_index]
            rows = np.column_stack((d[family], residual))
            geoms.update(zip(ids, rows, strict=True))
    if geoms.keys() != meta.keys():
        raise ValueError("Incomplete electronic-reference geometry cache")
    for src, sys, method, case, rs, t, ps in links:
        names = [*rs, t, *ps]
        if not all(n in geoms for n in names):
            continue
        split = meta[t][4]
        if any(meta[n][4] != split for n in names):
            raise ValueError("Channel crosses the source split")
        add(
            src + " | " + method,
            paper_target(method),
            split,
            sum(geoms[n] for n in rs),
            geoms[t],
            sum(geoms[n] for n in ps) if ps else None,
            (sys, case),
        )
    # Per-scan centering removes the arbitrary absolute energy zero. Distinct
    # systems, multiplicities and native Hamiltonians keep separate offsets.
    scans = defaultdict(list)
    for n, (src, sys, series, method, split, q, m, *_) in meta.items():
        if src == "AIMNet2-NSE supplement":
            scans[method, split, sys, series, q, m].append(n)
    for (method, split, sys, series, q, m), ids in scans.items():
        name = "AIMNet2-NSE | " + method
        rows = np.stack([geoms[n] for n in ids])
        rows -= rows.mean(axis=0)
        counts[name][split] += len(rows)
        if split == "train":
            train["wb97xd3", name].extend(
                (row, min(1, 250 / len(rows))) for row in rows
            )
        else:
            held["wb97xd3", name].extend(
                ((sys, series, n), row, None) for n, row in zip(ids, rows)
            )
    return train, held, counts


def profile_native(rows, tau, dim):
    """Profile a source-specific coefficient offset out of the shared loss.

    Each source minimizes its residual after fitting a separate offset with
    ridge tau. Eliminating that offset analytically leaves a positive
    semidefinite contribution to the common energy-surface normal equations.
    """
    z = np.stack([r for r, _ in rows])
    scales = np.array([w for _, w in rows])
    z *= np.sqrt(scales * min(1, 3000 / len(z)))[:, None]
    gram = z.T @ z
    middle = cho_factor(gram[:-1, :-1] + tau * np.eye(dim))
    profiled = gram - gram[:, :-1] @ cho_solve(middle, gram[:-1, :])
    return (profiled + profiled.T) * 0.5


def external_scores(held, w_by_target):
    """Score the deployed shared coefficients, without native offsets."""
    out = {}

    def calc(entries, w):
        f = np.array([a[:-1] @ w - a[-1] for _, a, p in entries]) * kt.KCAL
        if entries[0][2] is None:
            return {"n": len(f), "single": float(np.abs(f).mean())}
        g = np.array([p[:-1] @ w - p[-1] for _, a, p in entries]) * kt.KCAL
        return dict(
            n=len(f),
            reaction=float(np.abs(g).mean()),
            forward=float(np.abs(f).mean()),
            reverse=float(np.abs(f - g).mean()),
            barriers=float((np.abs(f).mean() + np.abs(f - g).mean()) / 2),
        )

    for (target, name), entries in held.items():
        panels = {name: entries}
        if name == "RDB19-Rad":
            panels[name + " H transfers"] = [
                v
                for v in entries
                if v[0][1] in ("H Atom Abstraction", "intra_H_migration")
            ]
        for label, cases in panels.items():
            w = w_by_target[target]
            out[label] = {
                "parent": calc(cases, np.zeros_like(w)),
                "fit": calc(cases, w),
            }
    return out


def fit(family, angular=False):
    from experiments.full_public.trial_path_directions import fit_surfaces

    parent, stage = STAGES[family]
    directory = ANGULAR if angular else ROOT
    stage += 2 if angular else 0
    basis = bases(angular)[family]
    train, held, counts = external_rows(family, directory)
    constraints = {t: np.zeros((basis.size + 1,) * 2) for t in kt.TARGETS}
    for (t, name), rows in train.items():
        # Bound leverage of correlated scans and low-fidelity / reference-
        # specific differences, in the same parameter units for both bases.
        tau = 120.0 if name.startswith("AIMNet2") else 1200.0
        constraints[t] += profile_native(rows, tau, basis.size)
    with np.load(directory / "reoptimized-directions.npz") as f:
        directions = f[family]
    # The three unit vectors are derived only from reoptimized training
    # reactions. Add modest prior variance along each native energy direction.
    covariance = np.eye(basis.size) + directions.T @ directions
    ridge = 60.0 * cho_solve(cho_factor(covariance), np.eye(basis.size))
    info = dict(
        operator=family,
        connectivity="H-centered sharing between two heavy atoms",
        native_training_counts={k: dict(v) for k, v in counts.items()},
        reoptimized=json.loads((directory / "reoptimized-counts.json").read_text()),
        angular_window_degrees=[110, 160] if angular else None,
        scan_offset_ridge=120,
        channel_offset_ridge=1200,
        ridge=60,
        columns=basis.size,
    )
    print(
        "Loaded direct native reference equations for",
        family,
        {k: v["train"] for k, v in counts.items()},
        flush=True,
    )
    # Write energies and source holdouts before the independent frequency jobs.
    main = np.load(directory / f"main-{family}.npy", mmap_mode="r")
    rad = np.load(directory / f"radical-{family}.npy", mmap_mode="r").reshape(
        -1, 7, basis.size
    )
    fit_surfaces(
        [(basis, None)],
        main,
        rad,
        info,
        f"p{stage}-public",
        f"htransfer{stage}",
        parent_stage=f"p{parent}-public",
        ridge_penalty=ridge,
        reference_constraints=constraints,
        relax_spectra=False,
    )
    weights = {
        t: ForceField.load(kt.WORK / f"p{stage}-public-{t}.npz", dispersion="d3")
        .additional_bases[-1]
        .weights
        for t in kt.TARGETS
    }
    stats = external_scores(held, weights)
    kt.write_json(directory / f"{family}-validation.json", stats)
    print("Completed native validation", family, stats["RDB19-Rad"], flush=True)
    from experiments.full_public.trial_spectra import evaluate_trial

    evaluate_trial(f"p{stage}-public")
    print("Completed frequencies", family, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("rdb19", "papers"))
    p.add_argument("--angular", action="store_true")
    args = p.parse_args()
    (path_cache if args.action == "rdb19" else paper_cache)(args.angular)
