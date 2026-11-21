"""Fit a second atomic environment resolution to cross-fitted Korny residuals.

Every inner fit gets its own parent energies, including the parent's predictions
on displaced radical structures. An outer structure or label never enters the
linear systems or hyperparameter selection. The fixed parent retains D3.
"""
import json
import pickle
import shutil
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt, trial_fit as fit
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField

WORK = kt.WORK
PARENT = "p11-public"


def baseline(target, fold, observable, parent=PARENT):
    """Average the same two independent fits that form the deployed parent."""
    if parent != "p11-public":
        name = json.loads((WORK / f"{parent}-settings.json").read_text())["branches"][target]
        return np.load(WORK / f"{name}-{target}-{observable}-{fold}.npy")
    p8 = json.loads((WORK / "p8-public-settings.json").read_text())["selected"]
    p10 = json.loads((WORK / "p10-public-settings.json").read_text())["branches"][target]
    return .5 * sum(np.load(WORK / f"{s}-{target}-{observable}-{fold}.npy") for s in (p8, p10))


def metrics(d, errors, mask):
    """Measure paired reaction and barrier MAEs on the singlet/doublet inner IDs."""
    src, kind = d[2], d[4]
    mask = mask & np.isin(src, kt.BARRIERS)
    return [float(abs(errors[mask & (kind == k)]).mean() * kt.KCAL)
            for k in ("reaction", "forward", "reverse")]


def saddle_metrics(energy, use, meta):
    curv = (energy[use, 3] + energy[use, 4] - 2 * energy[use, 0]) / .004**2
    ref, m = meta[use, 2], meta[use, 4]
    def frequency(k):
        return TO_CM1 * np.sign(k) * np.sqrt(abs(k) / m)
    return [float(abs(frequency(curv) - frequency(ref)).mean()),
            float((np.sign(curv) == np.sign(ref)).mean())]


def run(stage=12):
    from experiments.full_public import trial_bond_order as order
    from src.bond_order_kernel import BondOrderKernel
    dest = f"p{stage}-public"
    parent = "p12-public" if stage == 16 else f"p{stage-1}-public"
    # The second residual adds distance-sensitive first- and second-shell
    # environments. Both stages share the split, shape anchors, and readout.
    bond = stage in (13, 16)
    family = "metric" if stage == 16 else "bo"
    options = [(w, 0.) for w in order.LENGTHS] if bond else [(1., 1.), (.65, 1.5)]
    registry = {}

    context = {t: kt.equation_data(t) for t in kt.TARGETS}
    primary = {t: fit.selection_mask(d, t) for t, d in context.items()}
    meta = np.load(WORK / "rad-meta.npy")
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array([r[1][0].split("/")[-1].startswith("ts") for r in pickle.load(f)])
    trials = {t: {} for t in context}
    # First vary environment length and angular weight, then the amount of
    # regularization and the relative emphasis on transition-state curvature.
    for width, angular in options:
        basis, pub, _, rad, anchor, nist = (order.features(width, family) if bond else fit.get_features(("rich", width, angular, 256, False)))
        rad = rad.reshape(-1, 7, basis.size)
        for t, d in context.items():
            s, y, src, fold, kind, case = d
            x = s @ pub
            for h in (0, 1, -1):
                base = baseline(t, h, "parent", parent)
                train = (fold >= 0) & (fold != h)
                residual = y - s @ base
                moments = kt.grouped_moments(d, x, residual, train)
                moments["other"] += isomer_gram(t, h, pub, base)
                # Penalize changes in forces and harmonic curvatures at all
                # training NIST minima while learning the new resolution.
                moments["other"][:-1, :-1] += 8 * nist
                if t == "wb97xd3":
                    base_modes = baseline(t, h, "modes", parent)
                    fx, fy = fit.rad_rows(rad, base_modes, meta, np.where(saddle, 3., 1.))
                    selected = np.repeat((meta[:, 0] >= 0) & (meta[:, 0] != h), 3)
                    moments["other"] += fit.reduce_gram(fx, fy, selected)
                    del fx, fy
                for ridge, scale in ((.02, .015), (.2, .1), (2., .5)):
                    key = f"{family if bond else 'ms'}-{width:g}-{angular:g}-r{ridge:g}-s{scale:g}"
                    coef = kt.solve(moments, anchor, 32, 64, scale, ridge)
                    registry[key] = (width, angular)
                    np.save(WORK / f"{key}-{t}-parent-{h}.npy", base + pub @ coef)
                    if t == "wb97xd3":
                        np.save(WORK / f"{key}-{t}-modes-{h}.npy", base_modes + rad @ coef)
                    if h >= 0:
                        errors = x @ coef - residual
                        item = {"maes": metrics(d, errors, (fold == h) & primary[t])}
                        if t == "wb97xd3":
                            check = (meta[:, 0] == h) & saddle
                            item["saddle"] = saddle_metrics(base_modes + rad @ coef, check, meta)
                            item["parent_saddle"] = saddle_metrics(base_modes, check, meta)
                        trials[t].setdefault(key, []).append(item)
                    else:
                        np.save(WORK / f"{key}-{t}-coef.npy", coef)
                        # Keep the held predictions private to this script until
                        # both folds have chosen the model. No outer metrics enter selection.
                        mask = fold == -1
                        np.savez_compressed(WORK / f"{key}-{t}-equations.npz",
                            source=src[mask], case=case[mask], kind=kind[mask],
                            error_kcal=((x @ coef - residual) * kt.KCAL)[mask])
                        if t == "wb97xd3":
                            held = meta[:, 0] == -1
                            np.save(WORK / f"{key}-held-modes.npy", (base_modes[held] + rad[held] @ coef))
            print("Finished development fits", t, width, angular, flush=True)
    selections = {}
    for t, options in trials.items():
        def criterion(key):
            folds = options[key]
            rxn, fwd, rev = np.mean([v["maes"] for v in folds], axis=0)
            score = .35 * rxn + .65 * (fwd + rev) / 2
            if t == "wb97xd3":
                # A 100 cm^-1 deterioration costs one kcal/mol in selection.
                score += max(0., np.mean([v["saddle"][0] - v["parent_saddle"][0] for v in folds])) / 100
            return score
        key = min(options, key=criterion)
        selections[t] = key
        width, angular = registry[key]
        from src.reactive_basis import ReactiveBasis
        with np.load(WORK / (f"{family}-basis.npz" if bond else "rich-basis.npz")) as file:
            basis = BondOrderKernel.load(file) if bond else ReactiveBasis.load(file)
        basis.length = width
        if not bond:
            basis.angular = angular
        basis.weights = np.load(WORK / f"{key}-{t}-coef.npy")
        model = ForceField.load(WORK / f"{parent}-{t}.npz", dispersion="d3")
        model.additional_bases.append(basis)
        path = WORK / f"{dest}-{t}.npz"
        model.save_with_base(kt.original.BASE, path)
        verify(path, t, basis, parent, bond, family)
        shutil.copyfile(WORK / f"{key}-{t}-equations.npz", WORK / f"{dest}-{t}-equations.npz")
        if t == "wb97xd3":
            shutil.copyfile(WORK / f"{key}-held-modes.npy", WORK / f"{dest}-held-modes.npy")
    kt.write_json(WORK / f"{dest}-settings.json", dict(selected=dest, branches=selections, development=trials))
    print("Selected on inner reactions and saddle frequencies", selections, flush=True)


def verify(path, target, basis, parent, bond, family):
    """Check the reload against cached energy and the analytical force against finite differences."""
    model = ForceField.load(path, dispersion="d3")
    z, r = fit.sample()
    ix = fit.sample.index
    from experiments.full_public.trial_features import cache
    feature_file = WORK/f"{family}-{basis.length:g}-main.npy" if bond else cache("rich", basis.length, basis.angular)
    x = np.load(feature_file, mmap_mode="r")[ix]
    np.testing.assert_allclose(model.energy(z, r), baseline(target, -1, "parent", parent)[ix] + x @ basis.weights, atol=1e-7, rtol=0)
    v = np.random.default_rng(42).normal(size=r.shape)
    v /= np.linalg.norm(v)
    fd = (model.energy(z, r - .00002*v) - model.energy(z, r + .00002*v)) / .00004
    np.testing.assert_allclose(np.sum(model.predict(z, r)[1] * v), fd, atol=3e-5)


if __name__ == "__main__":
    with threadpool_limits(3):
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--stage", type=int, choices=(12,13,16),default=12)
        run(parser.parse_args().stage)
