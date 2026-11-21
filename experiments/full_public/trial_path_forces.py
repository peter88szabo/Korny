"""Add independent Cartesian force directions to the full native radical fit.

IRC work constrains one tangent at a time. Here the same 1,996 atomic kernel
columns also see forces across independent displacements at the common saddle,
the two endpoints and one high-force interior point on each branch. All
hyperparameter comparisons are made on internal training-formula partitions.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_periodic as kernel
from experiments.full_public.irc_learning_curve import direction
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_path_scales import OPTIONS, metrics

SOURCE = IRC.parent / "multiscale-path61"
ROOT = IRC.parent / "multiscale-path63"
N_DIR = 4


def initialize():
    global cols
    kernel.setup()
    kernel.basis, cols = load_basis()
    kernel.OPTIONS = OPTIONS


def encode(rx):
    dest = ROOT / (rx + "-forces.npz")
    if dest.exists():
        return
    pts, labels = [], []
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(("transverse63/" + rx).encode()).digest()[:8], "big"
        )
    )
    step = 0.001
    for side in ("forw", "rev"):
        with np.load(IRC / f"{rx}-{side}.npz") as f:
            z, r, force = f["z"], f["r_A"].astype(float), f["force_eV_A"].astype(float)
        steep = (
            (
                1
                + np.linalg.norm(
                    force[1:-1].reshape(len(force) - 2, -1), axis=1
                ).argmax()
            )
            if len(force) > 2
            else 0
        )
        idx = [-1, steep]
        if side == "forw":
            idx.append(0)
        for i in idx:
            # Orthonormal Cartesian directions carry off-path information.
            # A molecular translation is removed before orthogonalization.
            v = rng.normal(size=(N_DIR, len(z), 3))
            v -= v.mean(axis=1, keepdims=True)
            directions = np.linalg.qr(v.reshape(N_DIR, -1).T, mode="reduced")[
                0
            ].T.reshape(-1, len(z), 3)
            for u in directions:
                pts.extend([(z, r[i] - step * u), (z, r[i] + step * u)])
                labels.append(np.sum(force[i] * u))
    banks = kernel.batch(pts)
    values = np.column_stack([a[:, cols] for a in banks])
    derivative = (values[::2] - values[1::2]) / (2 * step)
    tmp = dest.with_suffix(".pending")
    with tmp.open("wb") as f:
        np.savez(f, rows=np.column_stack((derivative, labels)).astype(np.float32))
    tmp.replace(dest)


def solve(output, energy_limit=None, curvature=False, refine=False):
    """Select a force weight on formula-grouped training folds and refit all training paths."""
    records = json.loads((SOURCE / "snapshot.json").read_text())
    train = [r for r in records if not r["heldout"]]
    n = len(train)
    # Identical training formula folds to the native energy benchmark.
    fold = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"kernel56/{r['formula']}".encode()).digest()[:4], "big"
            )
            % 3
            for r in train
        ]
    )
    dim = 2 * len(cols)
    ep = np.zeros((3, dim + 1, dim + 1))
    fp = np.zeros_like(ep)
    gap_mid = np.empty((n, 2, dim + 1))
    for h in range(3):
        ix = np.flatnonzero(fold == h)
        for start in range(0, len(ix), 48):
            e, f = [], []
            for j in ix[start : start + 48]:
                rx = train[j]["rxn"]
                with (
                    np.load(SOURCE / (rx + ".npz")) as a,
                    np.load(ROOT / (rx + "-forces.npz")) as b,
                ):
                    x = a["rows"].astype(float)
                    force = b["rows"].astype(float)
                gap_mid[j] = x[:2]
                e.append(x)
                f.append(force / np.sqrt(len(force)))
            x = np.vstack(e)
            force = np.vstack(f)
            ep[h] += x.T @ x
            fp[h] += force.T @ force

    def fitted(e, f, count, weight):
        g = e + weight * f
        lhs = g[:-1, :-1].copy()
        lhs.flat[:: dim + 1] += 1e-7 * count
        return cho_solve(cho_factor(lhs), g[:-1, -1])

    weights = (0.0, 0.03, 0.15, 0.75, 3.75, 18.75)
    signs = np.ones(n)
    if energy_limit is not None:
        # The two IRC files are integration branches. The reference endpoint
        # geometries identify which branch terminates at the physical reactant.
        for j, record in enumerate(train):
            ends = direction(record["rxn"])
            if ends is None:
                raise ValueError(f"Cannot assign training endpoints for {record['rxn']}")
            signs[j] = 1 if ends[0] == 0 else -1
        weights = sorted({*weights, 0.04, 0.06, 0.08, 0.10, 0.12})
    curvature_weights = (0.0,)
    if curvature:
        from experiments.full_public.audit_irc_curvature import signed_wavenumber
        from experiments.full_public.trial_irc_curvature import ROOT as curvature_cache
        with np.load(curvature_cache / "training-curvatures.npz") as archive:
            np.testing.assert_array_equal(archive["ids"], [r["rxn"] for r in train])
            curvature_rows, usable = archive["rows"], archive["use"]
        curvature_gram = np.zeros_like(ep)
        for h in range(3):
            block = curvature_rows[(fold == h) & usable]
            curvature_gram[h] = block.T @ block
        weights = (0.0, 0.03, 0.1)
        curvature_weights = (0.0, 0.0001, 0.001, 0.01, 0.1)
        if refine:
            # The coarse training folds bracketed the energy and force ceilings.
            # Retain the p66 selection as an eligible candidate in this bracket.
            weights = (0.04, 0.045, 0.05, 0.06, 0.075, 0.10)
            curvature_weights = (0.0001, 0.0008, 0.002, 0.004, 0.006, 0.008, 0.01)

    choices = []
    for weight, strength in ((w, c) for w in weights for c in curvature_weights):
        diff = np.empty((n, 2))
        curvature_errors = []
        force_square = 0.0
        for h in range(3):
            held = fold == h
            energy_terms = ep.sum(0) - ep[h]
            if curvature:
                energy_terms += strength * (curvature_gram.sum(0) - curvature_gram[h])
            w = fitted(energy_terms, fp.sum(0) - fp[h], int((~held).sum()), weight)
            if curvature:
                rows = curvature_rows[held & usable]
                curvature_errors.extend(abs(signed_wavenumber(rows[:, :-1] @ w)
                                            - signed_wavenumber(rows[:, -1])))
            a = gap_mid[held]
            diff[held] = a[:, :, :-1] @ w - a[:, :, -1]
            c = np.r_[w, -1.0]
            force_square += c @ fp[h] @ c
        quality = metrics(signs * diff[:, 0], diff[:, 1])
        rmse = np.sqrt(max(0.0, force_square) / n)
        # Balance the requested 5 kcal/mol scale with a preregistered
        # 0.20 eV/Å scale for a unit Cartesian directional force.
        selection = (quality["reaction"] + quality["barriers"]) / 10 + rmse / 0.20
        choices.append(
            {
                "weight": weight,
                "inner_reaction": quality["reaction"],
                "inner_barrier": quality["barriers"],
                "inner_force_rmse": rmse,
                "selection": selection,
            }
        )
        if energy_limit is not None:
            choices[-1].update(inner_forward=quality["forward"], inner_reverse=quality["reverse"])
        if curvature:
            choices[-1].update(curvature_weight=strength, inner_curvature_mae_cm1=float(np.mean(curvature_errors)))
    if energy_limit is None:
        best = min(choices, key=lambda v: v["selection"])
    else:
        # The held-out 626 reactions take no part in model selection. Each
        # directed barrier must independently satisfy the energy constraint.
        def excess(candidate):
            energy_errors = [candidate[k] for k in ("inner_reaction", "inner_forward", "inner_reverse")]
            return max(0.0, max(energy_errors) - energy_limit)

        if curvature:
            # First require all energy targets, then the fixed Cartesian force
            # budget. Among eligible candidates, minimize the signed frequency
            # error. Ordering fallback violations keeps the choice reproducible.
            best = min(choices, key=lambda v: (
                excess(v), max(0.0, v["inner_force_rmse"] - 0.60),
                v["inner_curvature_mae_cm1"],
            ))
        else:
            best = min(choices, key=lambda v: (excess(v), v["inner_force_rmse"]))
    energy_terms = ep.sum(0)
    if curvature:
        energy_terms += best["curvature_weight"] * curvature_gram.sum(0)
    w = fitted(energy_terms, fp.sum(0), n, best["weight"])
    # This stage exports a single geometry-only potential. Its weights do not
    # depend on a family annotation at the training or prediction geometry.
    np.savez_compressed(output / "native.npz", weights=w[:, None], groups=["all"])
    kt.write_json(
        output / "settings.json",
        {
            "training": n,
            "directions_per_path": 5 * N_DIR,
            "columns": dim,
            "choices": choices,
            "selected": best,
            "training_energy_ceiling_kcal_mol": energy_limit,
            **({"training_curvatures": int(usable.sum()), "training_force_ceiling_eV_A": 0.60} if curvature else {}),
        },
    )
    kt.write_json(output / "snapshot.json", records)
    print(
        "Selected isotropic force weight", best, "using training formulas", flush=True
    )


def run(energy_limited=False, curvature=False, refine=False):
    stage = 67 if refine else 66 if curvature else 64 if energy_limited else 63
    output = IRC.parent / f"multiscale-path{stage}"
    output.mkdir(exist_ok=True)
    ROOT.mkdir(exist_ok=True)
    records = json.loads((SOURCE / "snapshot.json").read_text())
    initialize()
    train = [r["rxn"] for r in records if not r["heldout"]]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(pool.map(encode, train, chunksize=4, buffersize=6), 1):
            if i % 800 == 0:
                print(
                    "Prepared independent force directions",
                    i,
                    "of",
                    len(train),
                    flush=True,
                )
    if curvature:
        from experiments.full_public.trial_irc_curvature import prepare
        prepare()
    solve(output, energy_limit=5.0 if energy_limited or curvature else None, curvature=curvature, refine=refine)
    from experiments.full_public.native_irc_pes import run as forces
    from experiments.full_public.trial_path_scales import audit

    audit(output, SOURCE)
    result = json.loads((output / "all-held-validation.json").read_text())[0]
    print("Full native reaction validation with independent forces", result, flush=True)
    forces(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-limited", action="store_true")
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--refine-curvature", action="store_true")
    args = parser.parse_args()
    run(args.energy_limited, args.curvature or args.refine_curvature, args.refine_curvature)
