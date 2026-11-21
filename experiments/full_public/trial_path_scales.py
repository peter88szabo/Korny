"""Learn the shape of doublet reaction paths at two local environment scales.

Each complete RDB19-Rad IRC pair supplies the endpoint gap, the transition
state's height above their midpoint, and energy and force information on both
flanks. The native fit has its own electronic reference. Transfer to Korny uses
only smooth molecular energies evaluated on one geometry at a time.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_periodic as kernel
from experiments.full_public.irc_learning_curve import ROOT as OLD
from experiments.full_public.irc_learning_curve import direction
from experiments.full_public.trial_irc_transfer import IRC, transfer, verified_pairs
from experiments.full_public.trial_measured_curvature import load_basis

SNAPSHOT = IRC.parent / "multiscale-path56"
ROOT = SNAPSHOT
OPTIONS = ((0.85, 3), (1.35, 6))
STRIDE = 3


def initialize():
    kernel.setup()
    kernel.basis, selected = load_basis()
    kernel.OPTIONS = OPTIONS
    return selected[::STRIDE]


def observation_rows(rx, features):
    """The force label is the measured work per Å along an interior IRC step."""
    positions, energies, slopes = [], [], []
    for side in ("forw", "rev"):
        with np.load(IRC / f"{rx}-{side}.npz") as archive:
            z = archive["z"]
            r, e, f = (archive[k] for k in ("r_A", "energy_eV", "force_eV_A"))
            if side == "forw":
                positions.append((z, r[0]))
                energies.append(e[0])
            middle = (len(e) - 1) // 2
            nxt = min(middle + 1, len(e) - 1)
            for idx in (-1, middle, nxt):
                positions.append((z, r[idx]))
                energies.append(e[idx])
            dr = r[nxt].astype(float) - r[middle]
            distance = np.linalg.norm(dr)
            work = -np.sum(0.5 * (f[nxt] + f[middle]) * dr)
            slopes.append((distance, work))
    a = np.column_stack((features(positions), energies))
    ts, left, left_mid, left_next, right, right_mid, right_next = a
    gap = right - left
    mid = ts - (left + right) / 2
    rows = [gap, mid, 0.5 * (left_mid - ts), 0.5 * (right_mid - ts)]
    for cur, nxt, (dist, work) in zip(
        (left_mid, right_mid), (left_next, right_next), slopes
    ):
        delta = nxt - cur
        delta[-1] = work
        rows.append(0.20 * delta / dist if dist > 1e-4 else np.zeros_like(delta))
    # Compute differences before downcasting to protect the small barrier
    # labels from cancellation of the much larger absolute electronic energies.
    return np.asarray(rows)


def encode(rx):
    if (ROOT / f"{rx}.npz").exists():
        return

    def features(positions):
        return np.column_stack([bank[:, columns] for bank in kernel.batch(positions)])

    rows = observation_rows(rx, features)
    temp = ROOT / f"{rx}.pending"
    with temp.open("wb") as stream:
        np.savez(stream, rows=np.asarray(rows, dtype=np.float32))
    temp.replace(ROOT / f"{rx}.npz")


def prepare(full=False):
    global columns
    ROOT.mkdir(exist_ok=True)
    manifest = ROOT / "snapshot.json"
    if not manifest.exists():
        if full:
            # The full p45 manifest is generated after archive MD5 verification
            # and complete extraction, including forward/reverse matching.
            records = json.loads((OLD / "manifest.json").read_text())
        else:
            records = verified_pairs(sorted(IRC.glob("*-forw.npz")))
        kt.write_json(manifest, records)
    records = json.loads(manifest.read_text())
    if full and STRIDE == 3:
        for row in records:
            file = row["rxn"] + ".npz"
            if (SNAPSHOT / file).is_file() and not (ROOT / file).exists():
                os.link(SNAPSHOT / file, ROOT / file)
    todo = [r["rxn"] for r in records if not (ROOT / (r["rxn"] + ".npz")).exists()]
    print(
        "Verified complete paired reactions",
        len(records),
        "remaining to encode",
        len(todo),
        flush=True,
    )
    columns = initialize()
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        for n, _ in enumerate(pool.map(encode, todo, chunksize=4, buffersize=8), 1):
            if n % 500 == 0:
                print(
                    "Encoded stationary and flank positions",
                    n,
                    "of",
                    len(todo),
                    flush=True,
                )


def metrics(gap, mid):
    """Arguments contain signed errors in eV in the physical forward convention."""
    errs = np.stack([gap, mid, mid + gap / 2, mid - gap / 2])
    values = np.abs(errs).mean(axis=1) * kt.KCAL
    return dict(
        zip(("reaction", "midpoint", "forward", "reverse"), map(float, values))
    ) | {"barriers": float(values[2:].mean())}


def fit_native(full=False):
    records = json.loads((ROOT / "snapshot.json").read_text())
    # Use exactly the previous finite-direction benchmark for both data sizes.
    frozen = json.loads((OLD / "learning-curve-snapshot-ids.json").read_text())
    old_ids = {r["rxn"] for r in frozen if not r["heldout"]}
    signs = {}
    for r in frozen:
        if r["heldout"] and (ends := direction(r["rxn"])) is not None:
            signs[r["rxn"]] = 1 if ends[0] == 0 else -1
    assert set(signs) <= {r["rxn"] for r in records if r["heldout"]}
    rows = []
    for record in records:
        with np.load(ROOT / (record["rxn"] + ".npz")) as a:
            rows.append(a["rows"])
    data = np.stack(rows)
    del rows
    train = np.array([not r["heldout"] for r in records])
    old = np.array([r["rxn"] in old_ids for r in records])
    held = np.array([r["rxn"] in signs for r in records])
    orient = np.array([signs[r["rxn"]] for r in records if r["rxn"] in signs])
    formulas = np.array([r["formula"] for r in records], dtype=np.uint64)
    # Tune width/degree and shrinkage on training formulas only. Keep every
    # reaction with the same composition together in each internal split.
    buckets = {
        int(f): int.from_bytes(
            hashlib.sha256(f"kernel56/{f}".encode()).digest()[:4], "big"
        )
        % 3
        for f in np.unique(formulas)
    }
    inner = np.array([buckets[int(f)] for f in formulas])
    width = (data.shape[2] - 1) // 2
    banks = {
        "cubic": np.arange(width),
        "sixth": np.arange(width, 2 * width),
        "both": np.arange(2 * width),
    }
    width2 = 2 * width
    ids = np.arange(width2 + 1)

    def gram(mask):
        block = np.zeros((len(ids), len(ids)))
        idx = np.flatnonzero(mask)
        for start in range(0, len(idx), 96):
            a = data[idx[start : start + 96]].reshape(-1, len(ids)).astype(float)
            block += a.T @ a
        return block

    def solve(g, sel, ridge, n):
        w = np.zeros(width2)
        lhs = g[np.ix_(sel, sel)]
        lhs.flat[:: len(lhs) + 1] += n * ridge
        w[sel] = cho_solve(cho_factor(lhs), g[sel, -1])
        return w

    def score(mask, w, sign=1):
        x = data[mask, :2].astype(float)
        err = x[:, :, :-1] @ w - x[:, :, -1]
        return metrics(err[:, 0] * sign, err[:, 1])

    fitted = []
    ridges = (0.0001, 0.001, 0.01, 0.1)
    if full:
        # The initial three training folds selected the edge of the first
        # ridge grid. Resolve the lower end on those same training folds.
        ridges = (1e-7, 1e-6, 1e-5, *ridges)
    training_sets = [("expanded", train)]
    if STRIDE == 3:
        training_sets.insert(0, ("previous", old & train))
    for label, use in training_sets:
        grams = [gram(use & (inner == k)) for k in range(3)]
        total = sum(grams)
        candidates = []
        for name, idx in banks.items():
            for reg in ridges:
                risk, count = 0.0, 0
                for k, g in enumerate(grams):
                    check = use & (inner == k)
                    if not check.any():
                        continue
                    w = solve(total - g, idx, reg, (use & ~check).sum())
                    val = score(check, w)
                    risk += (val["reaction"] + val["barriers"]) / 2 * check.sum()
                    count += check.sum()
                candidates.append((risk / count, name, reg))
        chosen = min(candidates)
        for name, idx in banks.items():
            risk, _, reg = min(c for c in candidates if c[1] == name)
            w = solve(total, idx, reg, use.sum())
            row = dict(
                training=label,
                paths=int(use.sum()),
                kernel=name,
                ridge=reg,
                inner_mae=risk,
                held=int(held.sum()),
                **score(held, w, orient),
            )
            print("RDB19 native fit", row, flush=True)
            fitted.append(row)
        if label == "expanded":
            risk, name, reg = chosen
            idx = banks[name]
            shared = solve(total, idx, reg, use.sum())
            family = np.array([r["family"] for r in records])
            groups = [
                "all",
                *sorted(k for k, n in Counter(family[train]).items() if n >= 12),
            ]
            coeff = [shared]
            for group in groups[1:]:
                mask = train & (family == group)
                g = gram(mask)
                g[:-1, -1] -= g[:-1, :-1] @ shared
                coeff.append(solve(g, idx, max(reg, 0.002), mask.sum()))
            np.savez_compressed(
                ROOT / "native.npz", groups=groups, weights=np.column_stack(coeff)
            )
            if full:
                # Score the frozen pilot and refit on all newly arrived held
                # IDs with identifiable educt/product, after both are frozen.
                initial = {
                    r["rxn"]
                    for r in json.loads((SNAPSHOT / "snapshot.json").read_text())
                }
                fresh = np.zeros(len(records), dtype=bool)
                sign = []
                for i, r in enumerate(records):
                    if r["heldout"] and r["rxn"] not in initial:
                        ends = direction(r["rxn"])
                        if ends is not None:
                            fresh[i] = True
                            sign.append(1 if ends[0] == 0 else -1)
                if not fresh.any():
                    raise ValueError(
                        "No independently arrived direction-assigned validation pairs"
                    )
                with np.load(SNAPSHOT / "native.npz") as f:
                    previous = f["weights"][:, 0]
                if STRIDE == 1:
                    # The smaller bank is an exact nested subset at both
                    # length scales. Embed the frozen readout without refitting.
                    embedded = np.zeros((2, width))
                    embedded[:, ::3] = previous.reshape(2, -1)
                    previous = embedded.ravel()
                fresh_metrics = {
                    label: score(fresh, coef, np.array(sign))
                    for label, coef in (("snapshot", previous), ("full", shared))
                }
                fresh_metrics.update(
                    ids=[r["rxn"] for i, r in enumerate(records) if fresh[i]],
                    n=int(fresh.sum()),
                    formulas=len(set(formulas[fresh])),
                )
                kt.write_json(ROOT / "later-holdout.json", fresh_metrics)
                print(
                    "Later-arriving radical holdout",
                    fresh.sum(),
                    fresh_metrics["snapshot"],
                    fresh_metrics["full"],
                    flush=True,
                )
    kt.write_json(
        ROOT / "metrics.json",
        {
            "fits": fitted,
            "selected_by_training_formulas": chosen[1],
            "ridge": chosen[2],
            "columns_per_scale": width,
            "train_reactions": int(train.sum()),
            "new_train_reactions": int((train & ~old).sum()),
            "native_held_reactions": int(held.sum()),
            "training_positions_used": int(train.sum() * 7),
            "snapshot_available_training_path_positions": sum(
                r["frames"] for r in records if not r["heldout"]
            ),
        },
    )


def fit_public(full=False):
    """Use the selected scales as doublet energy features on the p49 surface."""
    records = json.loads((ROOT / "snapshot.json").read_text())
    with np.load(ROOT / "native.npz") as archive:
        groups = archive["groups"].tolist()
        parts = np.split(archive["weights"], len(OPTIONS))
    banks = []
    for (length, degree), weights in zip(OPTIONS, parts, strict=True):
        if np.any(weights):
            basis, indices = load_basis()
            basis.length, basis.degree = length, degree
            banks.append((basis, indices[::STRIDE], weights))
    transfer(
        groups,
        banks,
        records,
        "p61-public" if STRIDE == 1 else ("p59-public" if full else "p56-public"),
        "p49-public",
        "p49-public",
    )


def audit(model_root=None, rows_root=None):
    """Score every direction-assigned held path and retain its two signed errors."""
    model_root = ROOT if model_root is None else model_root
    rows_root = model_root if rows_root is None else rows_root
    records = json.loads((model_root / "snapshot.json").read_text())
    previous = json.loads((OLD / "learning-curve-snapshot-ids.json").read_text())
    previous_ids = {r["rxn"] for r in previous if r["heldout"]}
    train_formulas = {r["formula"] for r in records if not r["heldout"]}
    with np.load(model_root / "native.npz") as f:
        groups = f["groups"].tolist()
        weights = f["weights"]
    rows = []
    for record in records:
        if not record["heldout"]:
            continue
        if record["formula"] in train_formulas:
            raise ValueError("Training and validation share a molecular formula")
        rx = record["rxn"]
        sides = direction(rx)
        if sides is None:
            continue
        with np.load(rows_root / (rx + ".npz")) as f:
            x = f["rows"][:2].astype(float)
        w = weights[:, 0]
        family = record["family"]
        w_family = w + weights[:, groups.index(family)] if family in groups[1:] else w
        err = x[:, :-1] @ np.column_stack((w, w_family)) - x[:, -1, None]
        err[0] *= 1 if sides[0] == 0 else -1
        rows.append(
            {
                "rxn": rx,
                "family": family,
                "formula": record["formula"],
                "earlier": rx in previous_ids,
                "error_eV": err.tolist(),
            }
        )
    if not rows:
        raise ValueError("No direction-assigned validation reactions")
    families = sorted({r["family"] for r in rows})
    summary = []
    for label, use in [
        ("all", rows),
        ("original comparison", [r for r in rows if r["earlier"]]),
        ("additional IDs", [r for r in rows if not r["earlier"]]),
        *((name, [r for r in rows if r["family"] == name]) for name in families),
    ]:
        if not use:
            continue
        errors = np.array([v["error_eV"] for v in use])
        entry = {
            "panel": label,
            "n": len(use),
            "formulas": len({r["formula"] for r in use}),
        }
        for i, key in enumerate(("shared", "family")):
            entry[key] = metrics(errors[:, 0, i], errors[:, 1, i])
        summary.append(entry)
    kt.write_json(model_root / "native-direction-errors.json", rows)
    kt.write_json(model_root / "all-held-validation.json", summary)
    print("All formula-held native reactions", summary[:3], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "native", "transfer", "audit"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the completed RDB19-Rad manifest and a fresh held-out panel",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Fit the full release with the denser nested bank of atomic environments",
    )
    args = parser.parse_args()
    if args.dense and not args.full:
        parser.error("--dense requires --full")
    if args.full:
        STRIDE = 1 if args.dense else 3
        ROOT = IRC.parent / ("multiscale-path61" if args.dense else "multiscale-path59")
    if args.step == "prepare":
        prepare(args.full)
    elif args.step == "native":
        fit_native(args.full)
        audit()
    elif args.step == "audit":
        audit()
    else:
        fit_public(args.full)
