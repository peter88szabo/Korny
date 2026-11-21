"""Compare sparse radial kernels on identical atom environments and formula folds.

The landmark Gram supplies the Nyström metric in the ridge penalty. The
portable calculator stores equivalent coefficients in the original landmark
coordinates. Every feature sums local atomic contributions, and the isolated
atom's landmark response is subtracted to retain the dissociation convention.
"""

import argparse
import json

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_multiscale import baseline
from src.reactive_basis import ReactiveBasis, radial_profile

ROOT = kt.WORK / "profile-comparison69"
FAMILIES = ("gaussian", "matern52", "rational_quadratic", "exponential")
PARENT = "p68-public"


def basis_for(family, multiplier):
    with np.load(kt.WORK / "rich-basis.npz") as archive:
        old = ReactiveBasis.load(archive)
    # These centers were selected without reactions in either internal fold
    # or in the public validation partition. All profiles share the same 344.
    centers = {s: c[::3] for s, c in old.centers.items()}
    typical = np.median(np.concatenate([pdist(c) for c in centers.values()]))
    spec = [old.spec[k] for k in ("nr", "lmax", "ron", "rc")]
    return ReactiveBasis(
        old.species,
        spec,
        old.projections,
        centers,
        old.radial_dims,
        multiplier * typical,
        1.0,
        np.zeros(sum(map(len, centers.values()))),
        profile=family,
    )


def penalty(b):
    """Ridge on whitened Nyström features equals a landmark-Gram penalty."""
    pieces = []
    for c in b.centers.values():
        gram = radial_profile(b.profile, cdist(c, c, "sqeuclidean"), b.length)[0]
        gram.flat[:: len(c) + 1] += 1e-6
        pieces.append(gram)
    return block_diag(*pieces)


def molecular_features(b, side, start, stop):
    """Aggregate cached atomic descriptors in bounded batches with empty species allowed."""
    prefix = "rich-" if side == "main" else "rich-aux-"
    result = np.zeros((stop - start, b.size))
    column = 0
    for s, centers in b.centers.items():
        off = np.load(kt.WORK / f"{prefix}off-{s}.npy", mmap_mode="r")
        sites = np.load(kt.WORK / f"{prefix}sites-{s}.npy", mmap_mode="r")
        if stop > len(off) - 1:
            raise ValueError("Frame range exceeds the cached molecular offsets")
        operator = hasattr(b, "site_values")
        if not operator:
            ref = radial_profile(b.profile, np.sum(centers**2, axis=1), b.length)[0]
        for lo in range(start, stop, 512):
            hi = min(stop, lo + 512)
            offsets = off[lo : hi + 1]
            active = np.flatnonzero(np.diff(offsets))
            if len(active):
                chunk = sites[offsets[0] : offsets[-1]]
                val = (
                    b.site_values(s, chunk)
                    if operator
                    else radial_profile(
                        b.profile, cdist(chunk, centers, "sqeuclidean"), b.length
                    )[0]
                    - ref
                )
                result[lo - start + active, column : column + b.widths[s]] = (
                    np.add.reduceat(val, offsets[active] - offsets[0], axis=0)
                )
        column += b.widths[s]
    return result


def score_design(public, gram, data, parents, ridges=(0.3, 6.0, 60.0)):
    """Evaluate corrections on two formula folds excluded from both parent and fit."""
    results = []
    scores = {ridge: {} for ridge in ridges}
    for t, d in data.items():
        s, y, source, fold, kind, _ = d
        x = s @ public
        for h in (0, 1):
            target = y - s @ parents[t, h]
            train = (fold >= 0) & (fold != h)
            groups = kt.grouped_moments(d, x, target, train)
            eq = groups["other"] + 96 * groups["mid"] + 36 * groups["gap"]
            for ridge, by_source in scores.items():
                w = cho_solve(cho_factor(eq[:-1, :-1] + ridge * gram), eq[:-1, -1])
                err = (x @ w - target) * kt.KCAL
                for src in kt.BARRIERS:
                    use = (source == src) & (fold == h)
                    if not use.any():
                        continue
                    bucket = by_source.setdefault(src, {"reaction": [], "barriers": []})
                    for name, kinds in (
                        ("reaction", ["reaction"]),
                        ("barriers", ["forward", "reverse"]),
                    ):
                        bucket[name].extend(abs(err[use & np.isin(kind, kinds)]))
    for ridge, by_source in scores.items():
        summary = {
            src: {k: float(np.mean(v)) for k, v in row.items()}
            for src, row in by_source.items()
        }
        score = float(np.mean([v for row in summary.values() for v in row.values()]))
        results.append(dict(ridge=ridge, macro_mae=score, sources=summary))
    return results


def screen():
    n = json.loads((kt.WORK / "rich-ready.json").read_text())["molecular"]
    data = {t: kt.equation_data(t) for t in kt.TARGETS}
    # The p12 parent has genuine predictions from refits that omit each inner
    # fold. Later full-data parents cannot be used for hyperparameter selection.
    parents = {
        (t, h): baseline(t, h, "parent", "p12-public") for t in data for h in (0, 1)
    }
    results = []
    for family in FAMILIES:
        for width in (0.55, 1.10):
            b = basis_for(family, width)
            public = molecular_features(b, "main", 0, n)
            gram = penalty(b)
            for row in score_design(public, gram, data, parents):
                results.append(dict(family=family, width=width, **row))
            print("Completed training-formula comparison", family, width, flush=True)
    selected = min(
        (c for c in results if c["family"] != "exponential"),
        key=lambda c: c["macro_mae"],
    )
    kt.write_json(ROOT / "screen.json", {"choices": results, "selected": selected})
    print(
        "Selected twice-differentiable kernel on training formulas",
        selected,
        flush=True,
    )


def refit(balanced=False):
    """Change the radial profile, then vary singlet DFT supervision at fixed basis."""
    from experiments.full_public.trial_path_directions import fit_surfaces

    pick = json.loads((ROOT / "screen.json").read_text())["selected"]
    b = basis_for(pick["family"], pick["width"])
    main_meta = json.loads((kt.WORK / "rich-ready.json").read_text())
    aux_meta = json.loads((kt.WORK / "rich-aux-ready.json").read_text())
    main = molecular_features(
        b, "main", 0, main_meta["molecular"] + 3 * main_meta["hdf"]
    )
    start = aux_meta["private"]
    rad = molecular_features(b, "aux", start, start + 7 * aux_meta["radical"]).reshape(
        -1, 7, b.size
    )
    # Same public IDs, parent, ridge, derivative anchors and dispersion in both
    # fits. Only the second fit adds paired singlet DFT training labels.
    for stage, weight in (((71, 0.0),) if balanced else ((69, 0.0), (70, 0.25))):
        info = dict(
            profile=pick["family"],
            half_width=b.length,
            ridge=pick["ridge"],
            centers=b.size,
        )
        fit_surfaces(
            [(b, None)],
            main,
            rad,
            info,
            f"p{stage}-public",
            f"profile{stage}",
            parent_stage=PARENT,
            ridge_penalty=pick["ridge"] * penalty(b),
            shared_dft_weight=weight,
            balance_families=balanced,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("screen", "refit", "balanced"))
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.step == "screen":
        screen()
    else:
        refit(balanced=args.step == "balanced")
