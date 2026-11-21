"""Measure the effect of adding training radical IRCs with a fixed model and test set.

Each training reaction contributes its path-relative energies and projected
forces exactly as in the p45 auxiliary fit. Hold the single polynomial basis,
ridge per reaction, formula split and test reaction IDs fixed. A stable hash
yields nested training subsets. This diagnostic reports the native RDB19
electronic surface, independently of transfer into the final Korny readout.
"""

import argparse
import hashlib
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public.rdb19_irc import OUT
from experiments.full_public.validate_irc import rms

ROOT = OUT.parent / "path-kernel45"


def direction(rx):
    labels = []
    for name in ("IRC_educt", "IRC_product"):
        path = OUT / "source-endpoints" / f"{rx}-{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as d:
            labels.append((d["z"], d["r_A"]))
    ends = []
    for name in ("forw", "rev"):
        with np.load(OUT / f"{rx}-{name}.npz") as d:
            if any(not np.array_equal(d["z"], z) for z, r in labels):
                return None
            ends.append(d["r_A"][-1])
    diff = np.array([[rms(r, p) for p in ends] for _, r in labels])
    a = diff.argmin(1)
    if a[0] == a[1] or np.max(diff[np.arange(2), a]) > 0.002:
        return None
    return (0, 2) if a[0] == 0 else (2, 0)


def run(snapshot=False):
    if snapshot:
        from experiments.full_public.trial_irc_transfer import verified_pairs

        # Freeze only the completed design matrices. The full transfer fit
        # maintains its own manifest after the complete archive is verified.
        dest = ROOT / "learning-curve-snapshot-ids.json"
        if dest.is_file():
            rec = json.loads(dest.read_text())
        else:
            paths = [
                OUT / (p.stem + "-forw.npz")
                for p in sorted(ROOT.glob("*.npz"))
                if p.stem.isdecimal()
            ]
            rec = verified_pairs(paths)
            kt.write_json(dest, rec)
    else:
        rec = json.loads((ROOT / "manifest.json").read_text())
    train = sorted(
        (v for v in rec if not v["heldout"]),
        key=lambda v: hashlib.sha256(("fixed-irc-curve/" + v["rxn"]).encode()).digest(),
    )
    held = []
    for r in rec:
        if r["heldout"]:
            rx = r["rxn"]
            direction_ids = direction(rx)
            if direction_ids is None:
                continue
            with np.load(ROOT / (rx + ".npz")) as a:
                ends = a["ends"]
            i, j = direction_ids
            a, h, b = ends[[i, 1, j]]
            held.append([b - a, h - (a + b) / 2, h - a, h - b])
    if not held:
        raise ValueError("No direction-assigned held-out reactions")
    held = np.array(held)
    gram = np.zeros((held.shape[-1],) * 2)
    wanted = {min(n, len(train)) for n in (71, 512, 2048, len(train))}
    results = []
    frames = 0
    for i, r in enumerate(train, 1):
        with np.load(ROOT / (r["rxn"] + ".npz")) as a:
            gram += a["gram"]
        frames += r["frames"]
        if i not in wanted:
            continue
        lhs = gram[:-1, :-1].copy()
        lhs.flat[:: len(lhs) + 1] += 0.015 * i
        w = cho_solve(cho_factor(lhs), gram[:-1, -1])
        err = np.abs(held[:, :, :-1] @ w - held[:, :, -1]).mean(axis=0) * kt.KCAL
        row = {
            "training_reactions": i,
            "training_frames": frames,
            "test_reactions": len(held),
            "reaction": err[0],
            "midpoint": err[1],
            "forward": err[2],
            "reverse": err[3],
            "barriers": (err[2] + err[3]) / 2,
        }
        results.append(row)
        print("Fixed-basis radical data learning curve", row, flush=True)
    filename = (
        "snapshot-learning-curve.json"
        if snapshot
        else "fixed-basis-learning-curve.json"
    )
    kt.write_json(ROOT / filename, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    with threadpool_limits(1):
        run(parser.parse_args().snapshot)
