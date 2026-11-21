"""Build and compare continuous and discrete eigenfunction kernels for Korny.

All modes act on the same atom-centered invariant descriptors. Operator
parameters and normalization use the training landmarks from the fixed rich
basis. Neither the public validation labels nor their geometries choose modes.
"""

import argparse
import json
from math import comb, factorial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from numpy.polynomial import chebyshev as ch, hermite, laguerre, legendre
from scipy.spatial.distance import cdist
from scipy.special import jacobi, j0, jn_zeros

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_kernel_profiles import (
    molecular_features,
    score_design,
)
from experiments.full_public.trial_multiscale import baseline
from src.operator_kernel import OperatorKernel
from src.reactive_basis import ReactiveBasis

ROOT = kt.WORK / "operator-kernels73"
FAMILIES = (
    "hermite",
    "laguerre",
    "legendre",
    "chebyshev",
    "jacobi",
    "dirichlet_sine",
    "neumann_cosine",
    "bessel_disk",
    "krawtchouk",
    "hahn",
    "graph_diffusion",
)
DISCRETE = ("krawtchouk", "hahn", "graph_diffusion")


def scalar_coefficients(name):
    if name in ("dirichlet_sine", "neumann_cosine"):
        return np.pi * np.arange(1, 6)
    if name == "graph_diffusion":
        return np.empty(0)
    if name == "bessel_disk":
        # The regular disk modes J0(j_0n r) vanish at the outer boundary.
        # Tabulate a machine-precision Chebyshev interpolant so the portable
        # calculator and its analytic derivative need only NumPy at runtime.
        x = ch.chebpts1(48)
        roots = jn_zeros(0, 5)
        return ch.chebfit(x, j0((x[:, None] + 1) * roots / 2), 40).T
    if name in ("krawtchouk", "hahn"):
        # Birth/death generators on the 13 discrete coordination bins.
        # Detailed balance symmetrizes the operator. Dividing its normalized
        # eigenvectors by sqrt(pi) gives the functions of the original chain.
        n = 12
        j = np.arange(n + 1)
        birth, death = n - j, j
        if name == "hahn":
            birth, death = (j + 1.5) * (n - j), j * (n + 3.0 - j)
        prob = np.r_[1.0, np.cumprod(birth[:-1] / death[1:])]
        prob /= prob.sum()
        edge = np.diag(np.sqrt(birth[:-1] * death[1:]), 1)
        _, v = np.linalg.eigh(np.diag(birth + death) - edge - edge.T)
        modes = v[:, 1:6] / np.sqrt(prob[:, None])
        x = ch.chebpts1(n + 1)
        p = (x[:, None] + 1) / 2
        weights = np.array([comb(n, int(k)) for k in j]) * p**j * (1 - p) ** (n - j)
        return ch.chebfit(x, weights @ modes, n).T
    coeff = np.zeros((5, 6))
    for n in range(1, 6):
        monic = np.eye(6)[n, : n + 1]
        if name == "hermite":
            c = hermite.herm2poly(monic) / np.sqrt(2**n * factorial(n))
        elif name == "legendre":
            c = legendre.leg2poly(monic)
        elif name == "chebyshev":
            c = ch.cheb2poly(monic)
        elif name == "laguerre":
            c = laguerre.lag2poly(monic)
        elif name == "jacobi":
            c = jacobi(n, 1.0, 2.0).c[::-1]
        else:
            raise ValueError(name)
        coeff[n - 1, : len(c)] = c
    return coeff


def build(name, width, base=None):
    if base is None:
        with np.load(kt.WORK / "rich-basis.npz") as f:
            base = ReactiveBasis.load(f)
    params = {}
    features_per_element = 200 if name == "tensor_legendre" else 80
    for s, c in base.centers.items():
        p = {
            "normalizer": np.ones(features_per_element),
            "zero": np.zeros(features_per_element),
        }
        if name == "graph_diffusion":
            d2 = cdist(c, c, "sqeuclidean")
            sigma = width * np.sqrt(np.median(np.partition(d2, 12, axis=1)[:, 12]))
            a = np.exp(-d2 / (2 * sigma**2))
            d = a.sum(axis=1)
            mu, v = np.linalg.eigh(a / np.sqrt(d[:, None] * d[None, :]))
            ids = np.argsort(mu)[::-1][1:81]
            # One Markov step supplies mu times each eigenfunction when a
            # landmark is queried. A second diffusion step suppresses noise.
            p["modes"] = v[:, ids] * np.sqrt(d.sum() / d)[:, None] * mu[ids]
            p["scale"] = np.array([sigma])
        else:
            rng = np.random.default_rng(73000 + int(s))
            dirs = np.linalg.qr(rng.normal(size=(c.shape[1], 16)))[0]
            coords = c @ dirs
            p.update(
                directions=dirs,
                mean=coords.mean(0),
                scale=width * np.maximum(0.05, coords.std(0)),
            )
        params[s] = p
    spec = [base.spec[k] for k in ("nr", "lmax", "ron", "rc")]
    b = OperatorKernel(
        base.species,
        spec,
        base.projections,
        base.centers,
        params,
        name,
        scalar_coefficients("legendre" if name == "tensor_legendre" else name),
        np.zeros(len(params) * features_per_element),
    )
    for s, c in b.centers.items():
        p = params[s]
        zero = b.raw_modes(s, np.zeros_like(c[:1]))[0][0]
        magnitude = np.sqrt(np.mean((b.raw_modes(s, c)[0] - zero) ** 2, axis=0))
        taper = (
            np.exp(-np.arange(80) / 40)
            if name == "graph_diffusion"
            else np.tile(1 / np.sqrt(1 + (np.arange(1, 6) / 3) ** 2), 16)
        )
        if name == "tensor_legendre":
            taper = np.r_[taper, np.full(120, 1 / np.sqrt(1 + (2 / 3) ** 2))]
        p.update(zero=zero, normalizer=taper / np.maximum(magnitude, 0.15))
    return b


def screen_family(family):
    n = json.loads((kt.WORK / "rich-ready.json").read_text())["molecular"]
    rows = []
    for width in (0.8, 1.6):
        b = build(family, width)
        mat = molecular_features(b, "main", 0, n)
        for row in score_design(
            mat, np.eye(b.size), datasets, parents, (0.3, 6.0, 60.0, 600.0)
        ):
            rows.append(dict(family=family, width=width, **row))
    kt.write_json(ROOT / f"{family}.json", rows)
    best = min(rows, key=lambda a: a["macro_mae"])
    print("Completed eigenfunction comparison", family, best["macro_mae"], flush=True)


def screen():
    global datasets, parents
    datasets = {t: kt.equation_data(t) for t in kt.TARGETS}
    parents = {
        (t, h): baseline(t, h, "parent", "p12-public") for t in datasets for h in (0, 1)
    }
    ROOT.mkdir(exist_ok=True)
    todo = [f for f in FAMILIES if not (ROOT / f"{f}.json").is_file()]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        list(pool.map(screen_family, todo, chunksize=1))
    best = [
        min(json.loads((ROOT / f"{f}.json").read_text()), key=lambda x: x["macro_mae"])
        for f in FAMILIES
    ]
    # Record the unconditional no-extension control on precisely these folds.
    zero = {}
    for t, (s, y, src, h, kind, case) in datasets.items():
        err = np.zeros(len(y))
        for f in (0, 1):
            m = h == f
            err[m] = abs(s[m] @ parents[t, f] - y[m]) * kt.KCAL
        for source in kt.BARRIERS:
            m = (src == source) & np.isin(h, (0, 1))
            if m.any():
                zero[source] = [
                    float(err[m & (kind == "reaction")].mean()),
                    float(err[m & np.isin(kind, ("forward", "reverse"))].mean()),
                ]
    kt.write_json(
        ROOT / "summary.json",
        dict(
            best=best, control=zero, control_macro=float(np.mean(list(zero.values())))
        ),
    )


def refit(pick, stage, relax=True, cache=None):
    from experiments.full_public.trial_path_directions import fit_surfaces

    b = build(pick["family"], pick["width"])
    meta = json.loads((kt.WORK / "rich-ready.json").read_text())
    aux = json.loads((kt.WORK / "rich-aux-ready.json").read_text())
    main = molecular_features(b, "main", 0, meta["molecular"] + 3 * meta["hdf"])
    rad = molecular_features(
        b, "aux", aux["private"], aux["private"] + 7 * aux["radical"]
    ).reshape(-1, 7, b.size)
    fit_surfaces(
        [(b, None)],
        main,
        rad,
        dict(
            operator=pick["family"],
            width=pick["width"],
            ridge=pick["ridge"],
            columns=b.size,
        ),
        f"p{stage}-public",
        f"operator{stage}",
        parent_stage="p68-public",
        ridge_penalty=pick["ridge"] * np.eye(b.size),
        parent_force_cache=cache,
        relax_spectra=relax,
    )


def full_candidates():
    """Rank continuous and discrete models independently before public scoring."""
    best = json.loads((ROOT / "summary.json").read_text())["best"]
    leaders = [
        min(
            (x for x in best if (x["family"] in DISCRETE) == flag),
            key=lambda x: x["macro_mae"],
        )
        for flag in (False, True)
    ]
    return dict(enumerate(leaders + [x for x in best if x not in leaders], 73))


def radial_controls():
    """Give the conventional profiles the same column and ridge budgets as the operator fits."""
    from experiments.full_public.trial_kernel_profiles import (
        FAMILIES as profiles,
        basis_for,
        penalty,
    )

    n = json.loads((kt.WORK / "rich-ready.json").read_text())["molecular"]
    data = {t: kt.equation_data(t) for t in kt.TARGETS}
    out_of_fold = {
        (t, h): baseline(t, h, "parent", "p12-public") for t in data for h in (0, 1)
    }
    rows = []
    for family in profiles:
        for width in (0.55, 1.10):
            b = basis_for(family, width)
            b.centers = {s: c[:80] for s, c in b.centers.items()}
            b.widths = dict.fromkeys(b.centers, 80)
            b.size = sum(b.widths.values())
            b.weights = np.zeros(b.size)
            matrix = molecular_features(b, "main", 0, n)
            for row in score_design(
                matrix, penalty(b), data, out_of_fold, (0.3, 6.0, 60.0, 600.0)
            ):
                rows.append(dict(family=family, width=width, **row))
    kt.write_json(ROOT / "radial-controls.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=(
            "screen",
            "controls",
            "tensor",
            "fit-family",
            "continuous",
            "discrete",
            "remaining-a",
            "remaining-b",
            "spectra-a",
            "spectra-b",
        ),
    )
    parser.add_argument("--family", choices=FAMILIES, help="Named operator for fit-family")
    args = parser.parse_args()
    if (args.step == "fit-family") != (args.family is not None):
        parser.error("fit-family requires --family; other steps select their own basis")
    if args.step == "screen":
        screen()
    elif args.step == "controls":
        radial_controls()
    elif args.step == "fit-family":
        stage, pick = next((s, c) for s, c in full_candidates().items()
                           if c["family"] == args.family)
        refit(pick, stage)
    elif args.step == "tensor":
        control = next(
            x for x in full_candidates().values() if x["family"] == "legendre"
        )
        refit(control | {"family": "tensor_legendre"}, 85)
    elif args.step.startswith("spectra-"):
        from experiments.full_public.trial_spectra import evaluate_trial

        for stage in range(75 if args.step == "spectra-a" else 76, 84, 2):
            name = f"p{stage}-public"
            if not (kt.WORK / f"{name}-spectra.json").exists():
                evaluate_trial(name)
                print("Completed full relaxed spectra", name, flush=True)
    else:
        candidates = full_candidates()
        if args.step in ("continuous", "discrete"):
            stage = 74 if args.step == "discrete" else 73
            refit(candidates[stage], stage)
        else:
            cache = {}
            offset = 75 if args.step == "remaining-a" else 76
            for stage in range(offset, 75 + len(FAMILIES) - 2, 2):
                if not (kt.WORK / f"p{stage}-public-diagnostics.json").exists():
                    refit(candidates[stage], stage, relax=False, cache=cache)
