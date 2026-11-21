"""Fit the expanded reaction correction, profiling centers and nonlinear scales.

Public energy differences enter every fit. An explicit comparison can also
include eligible private CC/DFT energies in the high-level fit. DFT observations have separate nuisance coefficients at each
archived level. Eliminating them by a Schur complement shares information with
the CC target without treating DFT energies as CC labels.
"""

import json
import pickle
from collections import defaultdict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_features import cache, label
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_spin_fit import selection_mask
from experiments.reactivity.leuven_hshift import DIFF
from src.reactive_basis import ReactiveBasis
from src.forcefield import ForceField

WORK, OUT, KCAL = kt.WORK, kt.OUT, kt.KCAL


def private_blocks(x, base, labels, paths, cols):
    """Center energies only within a common method, formula, charge and spin.

    A DFT level can deviate from the shared surface at a quadratic cost. The
    eliminated level-specific vectors never enter the deployed CC potential.
    """
    groups = defaultdict(list)
    p = len(cols)
    for i, g, role, e, kind, file in labels:
        if role == "train":
            groups[g].append((i, e))
    totals = defaultdict(lambda: np.zeros((p + 1, p + 1)))
    for g, rows in groups.items():
        if len(rows) < 2:
            continue
        ids, energy = zip(*rows)
        ix = np.array(ids)
        a = np.column_stack((x[np.ix_(ix, cols)], np.asarray(energy) - base[ix]))
        a -= a.mean(axis=0)
        # Limit the influence of a single long trajectory at one composition.
        w = min(1.0, 1000 / len(rows))
        name = "CC" if "ccsd" in g[0] else g[0]
        totals[name] += w * (a.T @ a)
    out = totals.pop("CC", np.zeros((p + 1, p + 1)))
    for term in totals.values():
        h = term[:-1, :-1] * 0.3
        b = term[:-1, -1] * 0.3
        # min_u ||A(w + u)-y||² + eta ||u||², at this electronic level only.
        eta = 3.0
        a = h.copy()
        a.flat[:: p + 1] += eta
        fac = cho_factor(a, check_finite=False)
        shared = eta * cho_solve(fac, h, check_finite=False)
        out[:-1, :-1] += 0.5 * (shared + shared.T)
        out[:-1, -1] += eta * cho_solve(fac, b, check_finite=False)
    for name, role, idx, ref in paths:
        if role != "train":
            continue
        a = DIFF @ x[np.ix_(idx, cols)]
        y = ref - DIFF @ base[idx]
        rows = np.column_stack((a, y))
        out += 10 * (rows.T @ rows)
    return out


def private_score(v, base, paths):
    out = {}
    for role in ("train", "validation", "gamma_validation"):
        errors = [
            (DIFF @ (v[idx] + base[idx]) - ref) * KCAL
            for n, r, idx, ref in paths
            if r == role
        ]
        mae = np.mean(np.abs(errors), axis=0)
        out[role] = dict(
            n=len(errors),
            forward=mae[0],
            reverse=mae[1],
            reaction=mae[2],
            barriers=0.5 * (mae[0] + mae[1]),
        )
    return out


def rad_rows(x, b, meta, soft_weights=1.0):
    """Centered differences on the measured soft and stiff Hessian directions."""

    def construct(a):
        return np.stack(
            [
                (a[:, 1] - a[:, 2]) / 0.008,
                (a[:, 3] + a[:, 4] - 2 * a[:, 0]) / 0.004**2,
                (a[:, 5] + a[:, 6] - 2 * a[:, 0]) / 0.004**2,
            ],
            axis=1,
        )

    fm = construct(x)
    y = np.column_stack((np.zeros(len(meta)), meta[:, 2:4])) - construct(b)
    weight = np.column_stack((2 * meta[:, 1], np.full((len(meta), 2), 0.16)))
    weight[:, 1] *= soft_weights
    return (fm * weight[:, :, None]).reshape(-1, x.shape[-1]), (y * weight).ravel()


def reduce_gram(x, y, mask):
    a = np.column_stack((x[mask], y[mask]))
    return a.T @ a


def get_features(setting):
    tag, w, a, n, four, *extra = setting
    name = label(tag, w, a)
    from src.bond_environment import BondBasis

    bond = BondBasis.generate(*extra) if extra else None
    bond_name = (
        (f"bond-{extra[0]}-{extra[1]:g}" + ("-angle" if bond.angular else ""))
        if extra
        else ""
    )
    ids = np.concatenate([np.arange(j * 256, j * 256 + n) for j in range(4)])
    with np.load(WORK / f"{tag}-basis.npz") as f:
        b = ReactiveBasis.load(f)
    b = ReactiveBasis(
        b.species,
        [b.spec[k] for k in ("nr", "lmax", "ron", "rc")],
        b.projections,
        {s: c[:n] for s, c in b.centers.items()},
        b.radial_dims,
        w,
        a,
        np.zeros(4 * n + 12 * four + (bond.size if bond else 0)),
        four,
        bond,
    )
    primary = np.load(cache(tag, w, a), mmap_mode="r")
    aux = np.load(cache(tag, w, a, "aux"), mmap_mode="r")
    nbase = len(np.load(WORK / "prior.npy", mmap_mode="r"))
    nprivate = json.loads((WORK / f"{tag}-aux-ready.json").read_text())["private"]
    files = [primary, aux]
    bond_files = (
        [
            np.load(WORK / f"{bond_name}-{side}.npy", mmap_mode="r")
            for side in ("main", "aux")
        ]
        if bond
        else []
    )

    def block(k, lo, hi):
        out = np.asarray(files[k][lo:hi, ids])
        if four:
            other = np.load(
                WORK / ("four-aux.npy" if k else "four-main.npy"), mmap_mode="r"
            )
            out = np.column_stack((out, other[lo:hi]))
        if bond:
            out = np.column_stack((out, bond_files[k][lo:hi]))
        return np.ascontiguousarray(out)

    anchor = np.zeros((b.size, b.size))
    sources = np.load(WORK / "hdf-sources.npy")
    names, ct = np.unique(sources, return_counts=True)
    wt = {s: 4500 / c for s, c in zip(names, ct)}
    for lo in range(0, len(sources), 480):
        hi = min(lo + 480, len(sources))
        x = block(0, nbase + lo * 3, nbase + hi * 3).reshape(-1, 3, b.size)
        force = (x[:, 1] - x[:, 2]) * (4 / 0.002)
        wgt = np.sqrt([wt[s] for s in sources[lo:hi]])[:, None]
        e = x[:, 0] * wgt
        force *= wgt
        anchor += e.T @ e + force.T @ force
    nfile = WORK / f"{name}-{n}-{int(four)}{bond_name}-nist.npy"
    if not nfile.exists():
        parent = ForceField.load(OUT / "joint-pbe0-reaction.npz", dispersion="d3")
        rows = nist_rows(lambda z, r: b.design(z, r, True)[1], parent)
        np.save(nfile, rows[:, :-1].T @ rows[:, :-1])
    return (
        b,
        block(0, 0, nbase),
        block(1, 0, nprivate),
        block(1, nprivate, len(aux)),
        anchor,
        np.load(nfile),
    )


def fit(
    setting,
    context,
    include_leuven=False,
    robust=None,
    saddle_weight=None,
    isomer_weight=0.0,
    cache_parent=False,
):
    artifact = fmt(setting) + ("" if include_leuven else "-public")
    if robust:
        artifact += f"-h{robust[0]:g}-s{robust[1]:g}"
    if saddle_weight is not None:
        artifact += f"-soft{saddle_weight:g}"
    if isomer_weight:
        artifact += f"-isomer{isomer_weight:g}"
    b, pub, pvt, rad, anchor, nist = get_features(setting)
    results = {}
    labels, paths, pb = [], [], None
    private_file = WORK / "leuven-data.pkl"
    if len(pvt) and private_file.exists() and (WORK / "leuven-prior.npy").exists():
        with private_file.open("rb") as f:
            _, labels, paths = pickle.load(f)
        pb = np.load(WORK / "leuven-prior.npy")
        if len(pb) != len(pvt):
            raise ValueError(
                "Private feature cache and priors use different inventories"
            )
    if include_leuven and pb is None:
        raise ValueError("Prepare Leuven records before enabling private training")
    meta = np.load(WORK / "rad-meta.npy")
    rb = np.load(WORK / "rad-prior.npy")
    cfg = json.loads((WORK / "spin-choices.json").read_text())
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddles = np.array(
            [row[1][0].split("/")[-1].startswith("ts") for row in pickle.load(f)]
        )
    for target in context:
        d = context[target][0]
        S, y, _, fold, _, _ = d
        sel = context[target][1]
        x = S @ pub
        tuned = cfg[target]["best"]
        group_scores = []
        development_curvature = []
        for h in (0, 1, -1):
            col = h if h >= 0 else 2
            base = np.load(WORK / f"spin-energy-{target}-{h if h >= 0 else 'all'}.npy")
            if cache_parent and h < 0:
                # The final, published component is already fitted. Its readout
                # also provides an independent check on the recovered feature order.
                saved = ForceField.load(WORK / f"{artifact}-{target}.npz")
                coef = saved.reactive_basis.weights
                energy = base + pub @ coef
                with np.load(WORK / f"{artifact}-{target}-equations.npz") as f:
                    np.testing.assert_allclose(
                        ((S @ energy - y) * KCAL)[fold == -1], f["error_kcal"], atol=1e-7
                    )
                np.save(WORK / f"{artifact}-{target}-weights-{h}.npy", coef)
                np.save(WORK / f"{artifact}-{target}-parent-{h}.npy", energy)
                if target == "wb97xd3":
                    np.save(WORK / f"{artifact}-{target}-modes-{h}.npy",
                            rad.reshape(-1, 7, b.size) @ coef + rb[:, :, col])
                continue
            residual = y - S @ base
            inc = kt.grouped_moments(d, x, residual, (fold >= 0) & (fold != h))
            inc["other"][:-1, :-1] += 8 * nist
            if isomer_weight:
                from experiments.full_public.trial_isomers import gram

                inc["other"] += isomer_weight * gram(target, h, pub, base)
            if target == "high" and include_leuven:
                inc["other"] += private_blocks(
                    pvt, pb[:, col], labels, paths, np.arange(b.size)
                )
            elif target == "wb97xd3":
                xx, yy = rad_rows(
                    rad.reshape(-1, 7, b.size),
                    rb[:, :, col],
                    meta,
                    np.where(saddles, saddle_weight or 1.0, 1.0),
                )
                mask = np.repeat((meta[:, 0] >= 0) & (meta[:, 0] != h), 3)
                inc["other"] += reduce_gram(xx, yy, mask)
                del xx, yy
            if robust:
                from experiments.full_public import trial_robust

                coef = trial_robust.fit(
                    d,
                    x,
                    residual,
                    (fold >= 0) & (fold != h),
                    inc,
                    anchor,
                    tuned,
                    *robust,
                )
            else:
                coef = kt.solve(inc, anchor, tuned[1], tuned[2], tuned[0], 0.08)
            if h >= 0:
                if cache_parent:
                    # Inner-fold predictions must come from a model trained without
                    # that fold, including its radical Hessian observations.
                    np.save(WORK / f"{artifact}-{target}-weights-{h}.npy", coef)
                    np.save(WORK / f"{artifact}-{target}-parent-{h}.npy", base + pub @ coef)
                    if target == "wb97xd3":
                        np.save(WORK / f"{artifact}-{target}-modes-{h}.npy",
                                rad.reshape(-1, 7, b.size) @ coef + rb[:, :, col])
                    continue
                group_scores.append(
                    kt.score_value(kt.stats(d, x @ coef - residual, (fold == h) & sel))
                )
                if saddle_weight is not None and target == "wb97xd3":
                    from experiments.reactivity.frequency_validation import TO_CM1

                    use = (meta[:, 0] == h) & saddles
                    e = rad.reshape(-1, 7, b.size)[use] @ coef + rb[use, :, col]
                    curb = (e[:, 3] + e[:, 4] - 2 * e[:, 0]) / 0.004**2
                    ref = meta[use, 2]
                    mass = meta[use, 4]

                    def nu(k):
                        return TO_CM1 * np.sign(k) * np.sqrt(abs(k) / mass)

                    development_curvature.append(
                        float(np.abs(nu(curb) - nu(ref)).mean())
                    )
                continue
            public_validation = fold == -1
            np.savez_compressed(
                WORK / (artifact + f"-{target}-equations.npz"),
                source=d[2][public_validation],
                case=d[5][public_validation],
                kind=d[4][public_validation],
                error_kcal=(x @ coef - residual)[public_validation] * KCAL,
            )
            scores = kt.stats(d, x @ coef - residual, fold == -1)
            result = dict(
                inner=group_scores,
                score=float(np.mean(group_scores)),
                validation=scores,
            )
            if target == "high":
                if paths:
                    result["leuven"] = private_score(pvt @ coef, pb[:, 2], paths)
            else:
                e = rad.reshape(-1, 7, b.size) @ coef + rb[:, :, 2]
                curb = np.stack(
                    [
                        (e[:, 3] + e[:, 4] - 2 * e[:, 0]) / 0.004**2,
                        (e[:, 5] + e[:, 6] - 2 * e[:, 0]) / 0.004**2,
                    ],
                    axis=1,
                )
                ref = meta[:, 2:4]
                mass = meta[:, 4:6]
                from experiments.reactivity.frequency_validation import TO_CM1

                def freq(k):
                    return np.sign(k) * np.sqrt(np.abs(k) / mass) * TO_CM1

                result["radical_mode_mae_cm1"] = (
                    np.abs(freq(curb) - freq(ref))[meta[:, 0] == -1]
                    .mean(axis=0)
                    .tolist()
                )
            if development_curvature:
                result["development_saddle_mae_cm1"] = development_curvature
            results[target] = result
            b.weights = coef
            model = ForceField.load(WORK / f"spin-{target}.npz", dispersion="d3")
            model.reactive_basis = b
            model.save_with_base(kt.original.BASE, WORK / (artifact + f"-{target}.npz"))
            # Roundtrip and compare a scalar equation's gradient to central differences.
            restored = ForceField.load(
                WORK / (artifact + f"-{target}.npz"), dispersion="d3"
            )
            z, r = sample()
            v = np.random.default_rng(253).normal(size=r.shape)
            v /= np.linalg.norm(v)
            E, F = restored.predict(z, r, 0, 1)
            fd = (
                restored.energy(z, r - 0.00002 * v)
                - restored.energy(z, r + 0.00002 * v)
            ) / 0.00004
            np.testing.assert_allclose(fd, np.sum(v * F), atol=3e-5, rtol=3e-5)
            ix = sample.index
            np.testing.assert_allclose(
                E, base[ix] + pub[ix] @ coef, atol=1e-6, rtol=1e-10
            )
    if cache_parent:
        print("Verified fold-specific baseline", artifact, list(context), flush=True)
        return
    kt.write_json(WORK / (artifact + ".json"), results)
    print(
        "Finished candidate",
        setting,
        {s: round(r["score"], 2) for s, r in results.items()},
        flush=True,
    )
    return results


def sample():
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    i = next(
        i for i, n in enumerate(names) if gs[n].state == (0, 1) and len(gs[n].z) == 10
    )
    sample.index = i
    return gs[names[i]].z, gs[names[i]].r


def fmt(s):
    tag, w, a, n, four, *extra = s
    suffix = (
        (
            f"-bond{extra[0]}-l{extra[1]:g}"
            + ("-angle" if len(extra) > 2 and extra[2] else "")
        )
        if extra
        else ""
    )
    return f"{label(tag, w, a)}-n{n}-four{int(four)}" + suffix


def stage(stage, include_leuven=False):
    import shutil

    context = {t: (kt.equation_data(t), None) for t in kt.TARGETS}
    context = {t: (d, selection_mask(d, t)) for t, (d, _) in context.items()}
    if stage == 2:
        options = [("base", 1.0, 1.0, n, False) for n in (64, 128, 256)]
    else:
        previous_suffix = "" if include_leuven else "-public"
        prev = json.loads(
            (WORK / f"p{stage - 1}{previous_suffix}-settings.json").read_text()
        )["selected"]
        # Select one shared basis by giving the high-level singlet/doublet and
        # Grambow doublet development scores equal weight.
        tag, w, a, n, _ = prev
        options = [tuple(prev)]
        if stage == 3:
            options += [
                ("base", 0.55, 1.5, n, False),
                ("base", 1.65, 0.65, n, False),
                ("rich", 1.0, 1.0, n, False),
                ("rich", 0.65, 1.5, n, False),
            ]
        if stage == 4:
            options += [(tag, w, a, n, True)]
        if stage == 5:
            options += [(*prev, 48, length) for length in (1.0, 0.65)]
    suffix = "" if include_leuven else "-public"
    trials = {}
    for s in options:
        file = WORK / (fmt(s) + suffix + ".json")
        trials[s] = (
            json.loads(file.read_text())
            if file.exists()
            else fit(s, context, include_leuven)
        )
    best = min(options, key=lambda s: np.mean([r["score"] for r in trials[s].values()]))
    for target in kt.TARGETS:
        shutil.copyfile(
            WORK / (fmt(best) + suffix + f"-{target}.npz"),
            WORK / f"p{stage}{suffix}-{target}.npz",
        )
    kt.write_json(
        WORK / f"p{stage}{suffix}-settings.json",
        dict(
            selected=best,
            trials={
                fmt(s): [r["score"] for r in v.values()] for s, v in trials.items()
            },
        ),
    )
    kt.write_json(WORK / f"p{stage}{suffix}.json", trials[best])
    print("Completed priority", stage, best, flush=True)


def robust_stage():
    """Compare objective functions at fixed features, using only training folds for selection."""
    import shutil

    setting = json.loads((WORK / "p5-public-settings.json").read_text())["selected"]
    prefix = fmt(setting) + "-public"
    context = {
        t: (d, selection_mask(d, t)) for t in kt.TARGETS if (d := kt.equation_data(t))
    }
    trials = {}
    for opt in (None, (10.0, 1.0), (0.0, 4.0), (10.0, 4.0), (5.0, 4.0)):
        key = prefix if opt is None else prefix + f"-h{opt[0]:g}-s{opt[1]:g}"
        path = WORK / (key + ".json")
        trials[key] = (
            json.loads(path.read_text())
            if path.exists()
            else fit(setting, context, robust=opt)
        )
    winner = min(
        trials, key=lambda k: np.mean([v["score"] for v in trials[k].values()])
    )
    for t in kt.TARGETS:
        shutil.copyfile(WORK / f"{winner}-{t}.npz", WORK / f"p6-public-{t}.npz")
    kt.write_json(
        WORK / "p6-public-settings.json",
        dict(
            selected=winner,
            trials={
                k: [v["score"] for v in results.values()]
                for k, results in trials.items()
            },
        ),
    )
    kt.write_json(WORK / "p6-public.json", trials[winner])
    print("Completed robust objective", winner, flush=True)


def capacity_stage(stage=7):
    """Compare extra bond kernels and reaction weights on identical development folds."""
    import shutil

    setting = json.loads((WORK / "p5-public-settings.json").read_text())["selected"]
    context = {
        t: (d, selection_mask(d, t)) for t in kt.TARGETS if (d := kt.equation_data(t))
    }
    previous = json.loads((WORK / f"p{stage - 1}-public-settings.json").read_text())[
        "selected"
    ]
    trials = {previous: json.loads((WORK / f"{previous}.json").read_text())}
    options = (
        ((48, 16.0), (144, 4.0), (144, 16.0), (144, 64.0))
        if stage == 7
        else ((144, 4.0), (144, 16.0), (144, 64.0))
    )
    for count, strength in options:
        candidate = [*setting[:5], count, setting[6]]
        if stage == 8:
            candidate.append(True)
        key = fmt(candidate) + f"-public-h0-s{strength:g}"
        file = WORK / (key + ".json")
        trials[key] = (
            json.loads(file.read_text())
            if file.exists()
            else fit(candidate, context, robust=(0.0, strength))
        )
    winner = min(
        trials, key=lambda k: np.mean([v["score"] for v in trials[k].values()])
    )
    for t in kt.TARGETS:
        shutil.copyfile(WORK / f"{winner}-{t}.npz", WORK / f"p{stage}-public-{t}.npz")
    kt.write_json(
        WORK / f"p{stage}-public-settings.json",
        dict(
            selected=winner,
            trials={
                k: [v["score"] for v in results.values()]
                for k, results in trials.items()
            },
        ),
    )
    kt.write_json(WORK / f"p{stage}-public.json", trials[winner])
    print("Completed bond capacity experiment", winner, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=int, choices=(2, 3, 4, 5, 6, 7, 8))
    parser.add_argument(
        "--include-leuven",
        action="store_true",
        help="Opt in to private training for a controlled comparison",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=3, user_api="blas"):
        if args.stage in (6, 7, 8):
            if args.include_leuven:
                parser.error("The robust objective experiment uses the public dataset")
            if args.stage == 6:
                robust_stage()
            else:
                capacity_stage(args.stage)
        else:
            stage(args.stage, args.include_leuven)
