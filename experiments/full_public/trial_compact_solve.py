"""Accumulate exact sparse-kernel normal equations in bounded Cartesian blocks.

A large design can have hundreds of millions of nonzeros when thousands of
partially overlapping local centers contribute to a molecule. Keep the design
on disk and accumulate a single symmetric Gram using BLAS. No operation copies
the complete HDF5 matrix, its three displacement slices, or their transposes.
"""

import pickle
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk, dsymv
from experiments.full_public import (
    kernel_trials as kt,
    trial_fit,
    trial_multiscale as ms,
)
from experiments.full_public.trial_compact import ROOT, PARENT, STAGE, isomers
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.tauri_validation import compare
from src.forcefield import ForceField


def add(gram, x, y=None, w=None):
    """Pack feature and reference columns in row-major order for a zero-copy BLAS transpose."""
    a = np.empty((len(x), len(gram)), dtype=float)
    a[:, :-1] = x
    a[:, -1] = 0.0 if y is None else y
    if w is not None:
        a *= w[:, None]
    out = dsyrk(1.0, a.T, beta=1.0, c=gram, lower=1, overwrite_c=1)
    if not np.shares_memory(out, gram):
        raise RuntimeError("BLAS unexpectedly copied the Gram")


def anchors(model, main, n):
    dest = ROOT / "anchor-gram.npy"
    if dest.exists():
        return np.load(dest, mmap_mode="r")
    dim = model.size
    g = np.zeros((dim + 1, dim + 1), order="F")
    sources = np.load(kt.WORK / "hdf-sources.npy")
    cnt = Counter(sources.tolist())
    weight = np.sqrt([4500 / cnt[s] for s in sources])
    probes = np.random.default_rng(297).normal(size=(dim, 2)) / np.sqrt(dim)
    exact = np.zeros(2)
    batch = 768
    for lo in range(0, len(sources), batch):
        hi = min(lo + batch, len(sources))
        block = main[n + 3 * lo : n + 3 * hi].toarray().reshape(hi - lo, 3, dim)
        center, plus, minus = block.transpose(1, 0, 2)
        plus -= center
        minus -= center
        w = weight[lo:hi, None]
        rows = np.vstack(
            (
                0.14 * w * center,
                (0.56 / 0.002) * w * (plus - minus),
                (0.055 / 0.001**2) * w * (plus + minus),
            )
        )
        del center, plus, minus, block
        add(g, rows)
        exact += ((rows @ probes) ** 2).sum(axis=0)
        if (lo // batch + 1) % 30 == 0:
            print("Accumulated exact HDF curvature/force anchors", hi, flush=True)
    parent = ForceField.load(kt.OUT / "joint-pbe0-reaction.npz", dispersion="d3")
    nr = 4 * nist_rows(lambda z, r: model.design(z, r, True)[1], parent)[:, :-1]
    add(g, nr)
    exact += ((nr @ probes) ** 2).sum(axis=0)
    check = []
    for v in probes.T:
        u = np.r_[v, 0.0]
        check.append(u @ dsymv(1.0, g, u, lower=1))
    np.testing.assert_allclose(check, exact, rtol=3.0e-10)
    tmp = dest.with_suffix(".pending")
    with tmp.open("wb") as f:
        np.save(f, g)
    tmp.replace(dest)
    print("Verified full anchor Gram on independent probe vectors", flush=True)
    return g


def sparse_rows(g, x, y, ids, w):
    for lo in range(0, len(ids), 640):
        sl = slice(lo, lo + 640)
        add(g, x[ids[sl]].toarray(), y[ids[sl]], w[sl])


def add_radical(g, rad, bm, meta, saddle, h):
    select = (meta[:, 0] >= 0) & (meta[:, 0] != h)
    for lo in range(0, len(meta), 192):
        hi = min(lo + 192, len(meta))
        keep = select[lo:hi]
        x = rad[7 * lo : 7 * hi].toarray().reshape(hi - lo, 7, -1)[keep]
        b = bm[lo:hi][keep]
        m = meta[lo:hi][keep]
        sd = saddle[lo:hi][keep]
        op = [
            lambda v: (v[:, 1] - v[:, 2]) / 0.008,
            lambda v: ((v[:, 3] - v[:, 0]) + (v[:, 4] - v[:, 0])) / 0.004**2,
            lambda v: ((v[:, 5] - v[:, 0]) + (v[:, 6] - v[:, 0])) / 0.004**2,
        ]
        for fn, target, weight in zip(
            op,
            [np.zeros(len(m)), m[:, 2], m[:, 3]],
            [2 * m[:, 1], 0.16 * np.where(sd, 3.0, 1.0), np.full(len(m), 0.16)],
        ):
            add(g, fn(x), target - fn(b), weight)


def run(model, main, rad):
    n = len(np.load(kt.WORK / "prior.npy", mmap_mode="r"))
    nnz = main.indptr[n]
    pub = sparse.csr_matrix(
        (main.data[:nnz], main.indices[:nnz], main.indptr[: n + 1]),
        shape=(n, model.size),
        copy=False,
    )
    anchor = anchors(model, main, n)
    meta = np.load(kt.WORK / "rad-meta.npy")
    with (kt.WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [row[1][0].split("/")[-1].startswith("ts") for row in pickle.load(f)]
        )
    settings = {}
    scores = {}
    for t in kt.TARGETS:
        d = kt.equation_data(t)
        s, y, src, fold, kind, case = d
        x = (s @ pub).tocsr()
        primary = trial_fit.selection_mask(d, t)
        dev = {}
        for h in (0, 1, -1):
            gram = np.array(anchor, order="F", copy=True)
            base = ms.baseline(t, h, "parent", PARENT)
            resid = y - s @ base
            ids = np.flatnonzero((fold >= 0) & (fold != h))
            cnt = Counter(src[ids].tolist())
            w = np.sqrt(
                [
                    min(1.0, 4500 / cnt[k]) * (36.0 if k in kt.BARRIERS else 1.0)
                    for k in src[ids]
                ]
            )
            sparse_rows(gram, x, resid, ids, w)
            iso, iy = isomers(t, h, pub, base)
            sparse_rows(gram, iso, iy, np.arange(len(iy)), np.ones(len(iy)))
            del iso, iy
            if t == "wb97xd3":
                bm = ms.baseline(t, h, "modes", PARENT)
                add_radical(gram, rad, bm, meta, saddle, h)
            chosen = None if h >= 0 else min(dev, key=lambda k: np.mean(dev[k]))
            for ridge in (1.0, 8.0):
                key = f"compact38-r{ridge:g}"
                if chosen is not None and key != chosen:
                    continue
                a = np.array(gram[:-1, :-1], order="F", copy=True)
                a.flat[:: len(a) + 1] += ridge
                coef = cho_solve(
                    cho_factor(a, lower=True, overwrite_a=True, check_finite=False),
                    gram[-1, :-1],
                    check_finite=False,
                )
                del a
                en = base + pub @ coef
                err = s @ en - y
                np.save(kt.WORK / f"{key}-{t}-parent-{h}.npy", en)
                if t == "wb97xd3":
                    modes = bm + (rad @ coef).reshape(-1, 7)
                    np.save(kt.WORK / f"{key}-{t}-modes-{h}.npy", modes)
                if h >= 0:
                    rx, fw, bw = ms.metrics(d, err, (fold == h) & primary)
                    score = 0.4 * rx + 0.3 * (fw + bw)
                    if t == "wb97xd3":
                        score += (
                            ms.saddle_metrics(modes, (meta[:, 0] == h) & saddle, meta)[
                                0
                            ]
                            / 350
                        )
                    dev.setdefault(key, []).append(float(score))
                else:
                    held = fold == -1
                    np.savez_compressed(
                        kt.WORK / f"{STAGE}-{t}-equations.npz",
                        source=src[held],
                        kind=kind[held],
                        case=case[held],
                        error_kcal=err[held] * kt.KCAL,
                    )
                    if t == "wb97xd3":
                        np.save(
                            kt.WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1]
                        )
                    model.weights = coef
                    ff = ForceField.load(kt.WORK / f"{PARENT}-{t}.npz", dispersion="d3")
                    ff.additional_bases.append(model)
                    path = kt.WORK / f"{STAGE}-{t}.npz"
                    ff.save_with_base(kt.original.BASE, path)
                    restored = ForceField.load(path, dispersion="d3")
                    z, r = trial_fit.sample()
                    energy, force = restored.predict(z, r)
                    np.testing.assert_allclose(
                        energy, en[trial_fit.sample.index], rtol=0, atol=2.0e-6
                    )
                    v = np.random.default_rng(91).normal(size=r.shape)
                    v /= np.linalg.norm(v)
                    fd = (
                        restored.energy(z, r - 2.0e-5 * v)
                        - restored.energy(z, r + 2.0e-5 * v)
                    ) / 4.0e-5
                    np.testing.assert_allclose((force * v).sum(), fd, atol=5.0e-5)
                    settings[t] = key
            del gram
            print("Completed bounded-memory compact fit", t, h, flush=True)
        scores[t] = dev
        del x
    kt.write_json(
        kt.WORK / f"{STAGE}-settings.json",
        dict(selected=STAGE, branches=settings, development=scores, centers=model.size),
    )
    compare(STAGE)
    validate(STAGE)
