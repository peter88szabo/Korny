"""Test whether geometry alone can recover the RDB19-Rad family advantage.

Fit a local classifier on the atoms whose environments change most on the
training IRCs. Freeze its smooth probabilities and the native energy experts.
Compare three ways of mixing their energies on every native held-out formula,
then calibrate the geometry-only features to Korny's own electronic references.
"""

import json
from collections import defaultdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_periodic as kernel
from experiments.full_public.irc_learning_curve import direction
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.trial_path_directions import compact_banks
from experiments.full_public.trial_path_scales import ROOT as SOURCE
from experiments.full_public.trial_path_scales import metrics, observation_rows
from experiments.full_public.trial_spin_exchange import fit_doublets
from src.path_family import PathFamilyKernel

ROOT = IRC.parent / "local-family60"


def init_model():
    with np.load(SOURCE / "native.npz") as f:
        groups = f["groups"].tolist()
        pairs, _ = compact_banks(f["weights"])
    banks = [b for b, _ in pairs]
    native, logits = {}, {}
    lo = 0
    for key, centers in banks[0].centers.items():
        hi = lo + len(centers)
        native[key] = np.vstack([w[lo:hi] for _, w in pairs])
        logits[key] = np.zeros((len(native[key]) + 1, len(groups)))
        lo = hi
    return PathFamilyKernel(
        banks, native, logits, np.zeros(2 * len(groups) - 1)
    ), groups


def atomic_rows(model, z, r):
    b = model.banks[0]
    raw = kernel.raw(z, r)
    output = {}
    for key in np.unique(b.groups[z]):
        if key not in model.native:
            continue
        ids = np.flatnonzero(b.groups[z] == key)
        x = raw[ids]
        keep = np.flatnonzero(np.any(x != 0, axis=0))
        u = x[:, keep] @ b.projections[key][keep]
        output[key] = (ids, model.atomic(key, u)[0])
    return output


def fit_gate(model, groups, records):
    features, labels = defaultdict(list), defaultdict(list)
    for row in records:
        if row["heldout"]:
            continue
        family = groups.index(row["family"]) if row["family"] in groups else 0
        for side in ("forw", "rev"):
            with np.load(IRC / f"{row['rxn']}-{side}.npz") as f:
                z, r = f["z"], f["r_A"]
            views = [atomic_rows(model, z, r[i]) for i in (0, len(r) // 2, -1)]
            change = []
            for key, (ids, x) in views[0].items():
                # Ranking uses the observed path, never the energy or family
                # label. Inference will evaluate every atom independently.
                diff = views[2][key][1] - x
                speed = np.linalg.norm(diff, axis=1) / np.sqrt(x.shape[1])
                change.extend((float(v), int(key), j) for j, v in enumerate(speed))
            for speed, key, j in sorted(change, reverse=True)[:2]:
                if speed > 1e-7:
                    features[key].extend(v[key][1][j] for v in views)
                    labels[key].extend([family] * len(views))
    summary = {}
    for key, items in features.items():
        x = np.stack(items)
        y = np.array(labels[key])
        counts = np.bincount(y, minlength=len(groups))
        weight = 1 / np.sqrt(counts[y])
        weight /= weight.sum()

        def objective(params, x, y, weight):
            w = params.reshape(x.shape[1] + 1, len(groups))
            logits = x @ w[:-1] + w[-1]
            loss = weight @ (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
            error = softmax(logits, axis=1)
            error[np.arange(len(y)), y] -= 1
            error *= weight[:, None]
            gradient = np.vstack((x.T @ error, error.sum(0)))
            return loss + 0.0005 * np.sum(w * w), (gradient + 0.001 * w).ravel()

        ans = minimize(
            objective,
            model.logits[key].ravel(),
            args=(x, y, weight),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not ans.success:
            raise RuntimeError(f"Family classifier for group {key}: {ans.message}")
        model.logits[key] = ans.x.reshape(model.logits[key].shape)
        summary[int(key)] = {"environments": len(y), "iterations": ans.nit}
    kt.write_json(ROOT / "class-fit.json", summary)
    print("Fitted local radical classifiers", summary, flush=True)


def feature(model, z, r):
    total = np.zeros(model.size)
    for key, (_, x) in atomic_rows(model, z, r).items():
        total += model.features(key, x)[0].sum(0)
    return total


def native_validation(model, records):
    def encode(pts):
        return np.vstack([feature(model, z, r) for z, r in pts])

    rows = np.stack([observation_rows(r["rxn"], encode) for r in records])
    train = np.array([not r["heldout"] for r in records])
    held = np.zeros(len(records), bool)
    sign = []
    for i, r in enumerate(records):
        if r["heldout"] and (ends := direction(r["rxn"])) is not None:
            held[i] = True
            sign.append(1 if ends[0] == 0 else -1)
    f = (model.size + 1) // 2
    shared = np.zeros(model.size)
    shared[0] = 1
    checks = {}
    fitted = {}
    for name, ids in (
        ("shared", np.array([0])),
        ("global mixture", np.arange(f)),
        ("local family mixture", np.arange(model.size)),
    ):
        w = shared.copy()
        if name != "shared":
            a = rows[train].reshape(-1, model.size + 1)
            x = a[:, ids]
            reg = 0.003 * train.sum()
            w[ids] = cho_solve(
                cho_factor(x.T @ x + reg * np.eye(len(ids))),
                x.T @ a[:, -1] + reg * shared[ids],
            )
        test = rows[held, :2]
        errs = test[:, :, :-1] @ w - test[:, :, -1]
        checks[name] = metrics(errs[:, 0] * np.array(sign), errs[:, 1])
        fitted[name] = w
    archive = json.loads((SOURCE / "all-held-validation.json").read_text())[0]
    assert archive["n"] == int(held.sum())
    np.testing.assert_allclose(
        checks["shared"]["barriers"], archive["shared"]["barriers"], rtol=0, atol=5e-5
    )
    checks["given family"] = archive["family"]
    model.weights = fitted["local family mixture"]
    np.savez_compressed(ROOT / "native-geometry-only.npz", **model.fields())
    kt.write_json(
        ROOT / "native-evaluation.json",
        {"held": int(held.sum()), "training": int(train.sum()), "scores": checks},
    )
    print("Geometry-only native radical fit", int(held.sum()), checks, flush=True)


def run():
    kernel.setup()
    ROOT.mkdir(exist_ok=True)
    records = json.loads((SOURCE / "snapshot.json").read_text())
    if (ROOT / "basis.npz").exists():
        with np.load(ROOT / "basis.npz") as f:
            model = PathFamilyKernel.load(f)
    else:
        model, groups = init_model()
        fit_gate(model, groups, records)
        np.savez_compressed(ROOT / "basis.npz", **model.fields())
    native_validation(model, records)

    def encode(states):
        return np.vstack(
            [
                feature(model, z, r) if m == 2 else np.zeros(model.size)
                for z, r, q, m in states
            ]
        )

    fit_doublets(model, encode, "family60", "p60-public", ROOT)


if __name__ == "__main__":
    run()
