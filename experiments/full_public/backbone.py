"""Fit a charge-aware sparse Gaussian PBE0 surface on the complete public split.

Independent workers form double-precision sufficient statistics on disjoint
training batches. Only columns active in a molecule's composition enter its
matrix product. Memory depends on the sparse basis and worker count.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations_with_replacement
import json
import multiprocessing
import sqlite3
from time import monotonic

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from experiments.full_public.data import DB, OUT
from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField
from src.local import Molecules, descriptor_array
from src.pair import PairBaseline


@lru_cache(maxsize=512)
def packed_channels(species, active, nr, lmax):
    """Map the compact descriptor of the elements present into the full basis."""
    positions = []
    offset = 0
    for s in species:
        if s in active:
            positions.extend(range(offset, offset + nr))
        offset += nr
    for s, t in combinations_with_replacement(species, 2):
        width = (nr * (nr + 1) // 2 if s == t else nr * nr) * lmax
        if s in active and t in active:
            positions.extend(range(offset, offset + width))
        offset += width
    return np.array(positions, dtype=int)


class Basis:
    def __init__(self, structures, species, states):
        self.species = tuple(species)
        self.local = Molecules(
            structures,
            dict(species=self.species, nr=2, lmax=2, ron=2.0, rc=4.2),
            M=64,
            ell=5.0,
            scale_floor=0.1,
        )
        self.pair = PairBaseline(
            species,
            np.r_[np.arange(0.7, 3.5, 0.18), 3.7, 4.2, 5.0, 6.0, 7.0],
            0.22,
            5.5,
            8.0,
        )
        self.ef = ElectronicFeatures(species)
        self.states = sorted(states)
        self.charges = sorted({q for q, _ in states if q})
        self.offsets = len(species) + len(self.charges)
        self.pairs = len(self.pair.pairs) * len(self.pair.centers)
        self.sites = sum(head.size for head in self.local.heads.values())
        self.reference = (0, 1)
        self.others = sorted(set(states) - {self.reference})
        # Coefficients of smooth coordination (six per atom type) give charges
        # and radicals a Cartesian force response as well as an energy offset.
        self.state_cols = np.r_[
            0 : 1 + len(species), self.ef.size - 6 * len(species) : self.ef.size
        ]
        self.dim = (
            self.offsets
            + self.pairs
            + self.sites
            + len(self.others) * len(self.state_cols)
        )

    @lru_cache(maxsize=1024)
    def compact(self, s, active):
        """Absent element channels are constant; cache their Gaussian exponent."""
        loc = self.local
        indices = packed_channels(
            self.species, active, loc.spec["nr"], loc.spec["lmax"]
        )
        mean, inv = map(np.concatenate, loc.transform[s])
        h = loc.heads[s]
        delta = -mean * inv - h.centers
        absent = np.ones(len(mean), dtype=bool)
        absent[indices] = False
        c = delta[:, absent]
        return (
            mean[indices],
            inv[indices],
            h.centers[:, indices].copy(),
            np.einsum("ij,ij->i", c, c),
        )

    def design(self, z, r, q, m, deriv=True):
        """Return energy and conservative force rows in the saved model's basis."""
        pair, forces = self.pair.design(z, r)
        nrows = 1 + r.size if deriv else 1
        block = np.zeros((nrows, self.dim))
        block[0, : self.offsets] = [np.sum(z == s) for s in self.species] + [
            q == k for k in self.charges
        ]
        block[0, self.offsets : self.offsets + self.pairs] = pair
        if deriv:
            block[1:, self.offsets : self.offsets + self.pairs] = forces
        offset = self.offsets + self.pairs
        distances = cdist(r, r)
        for s in self.species:
            head = self.local.heads[s]
            end = offset + head.size
            for i in np.flatnonzero(z == s):
                ids = np.flatnonzero(distances[i] < self.local.spec["rc"])
                a = tuple(sorted(set(map(int, z[ids]))))
                mean, inv, centers, absent = self.compact(s, a)
                spec = dict(self.local.spec, species=a)
                raw, jac = descriptor_array(
                    r[ids], z[ids], int(np.searchsorted(ids, i)), gradient=deriv, **spec
                )
                delta = (raw - mean) * inv - centers
                ker = np.exp(-0.5 * (absent + np.einsum("ij,ij->i", delta, delta)))
                block[0, offset:end] += ker @ head.W - self.local.empty[s]
                if deriv:
                    df = jac.reshape(len(raw), -1).T @ (
                        (inv[:, None] * delta.T) @ (ker[:, None] * head.W)
                    )
                    block[
                        1 + (3 * ids[:, None] + np.arange(3)).ravel(), offset:end
                    ] += df
            offset = end
        if (q, m) != self.reference:
            st = self.others.index((q, m))
            cols = offset + st * len(self.state_cols) + np.arange(len(self.state_cols))
            v, f = self.ef.design(z, r)
            block[0, cols] = v[self.state_cols]
            if deriv:
                block[1:, cols] = f[:, self.state_cols]
        return block

    def model(self, w):
        local = self.local
        self.pair.weights = w[self.offsets : self.offsets + self.pairs]
        offset = self.offsets + self.pairs
        local.w = w[offset : offset + self.sites]
        local.freeze()
        charges = {
            0: 0.0,
            **dict(zip(self.charges, w[len(self.species) : self.offsets], strict=True)),
        }
        model = ForceField(self.pair, local, w[: len(self.species)], charges, "PBE0")
        table = np.zeros((len(self.states), self.ef.size))
        for j, state in enumerate(self.others):
            table[self.states.index(state), self.state_cols] = w[
                offset + self.sites + j * len(self.state_cols) : offset
                + self.sites
                + (j + 1) * len(self.state_cols)
            ]
        model.electronic = ElectronicCorrection(
            self.species, self.states, table, "PBE0"
        )
        return model


def sample(row):
    z, r, q, m, e, f = row
    return (
        np.frombuffer(z, "u1"),
        np.frombuffer(r, "<f8").reshape(-1, 3),
        q,
        m,
        e,
        np.frombuffer(f, "<f8").reshape(-1, 3),
    )


def worker_init():
    global connection
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def process(task):
    """Accumulate only nonzero design columns; scores use the same force rows."""
    mode, source, keys = task
    gram = np.zeros((basis.dim + 1, basis.dim + 1)) if mode == "train" else None
    metrics = np.zeros(6)
    which = ",".join("?" for _ in keys)
    records = connection.execute(
        f"SELECT z,r,q,m,energy,force FROM samples WHERE source=? AND name IN ({which})",
        (source, *keys),
    )
    groups = defaultdict(list)
    for row in records:
        z, r, q, m, e, f = sample(row)
        b = basis.design(z, r, q, m)
        y = np.r_[e, f.ravel()]
        if mode == "train":
            b = np.column_stack((b, y))
            b[1:] *= 4.0 / np.sqrt(r.size)
            # Sufficient statistics are accumulated in batches per composition.
            active = np.flatnonzero(np.any(b != 0.0, axis=0))
            groups[active.tobytes()].append(b[:, active])
        else:
            err = b @ weights - y
            metrics += [
                1,
                abs(err[0]),
                err[0] ** 2,
                r.size,
                np.abs(err[1:]).sum(),
                err[1:] @ err[1:],
            ]
    for raw, blocks in groups.items():
        cols = np.frombuffer(raw, dtype=int)
        v = np.vstack(blocks)
        gram[np.ix_(cols, cols)] += v.T @ v
    if gram is not None:
        active = np.flatnonzero(gram.diagonal())
        gram = (active, gram[np.ix_(active, active)])
    return source, len(keys), gram, metrics


def run(workers):
    global basis, weights
    started = monotonic()
    out = OUT / "backbone.json"
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    splits = defaultdict(lambda: defaultdict(list))
    states = set()
    species = set()
    seeds = []
    seen = set()
    for split, src, name, fp, q, m, z in conn.execute(
        "SELECT split,source,name,fp,q,m,z FROM samples WHERE split IN ('train','dev','test') ORDER BY split,source,name"
    ):
        # Keep distinct electronic states even at identical nuclear coordinates.
        key = split, fp, q, m
        if key in seen:
            continue
        seen.add(key)
        splits[split][src].append(name)
        if split == "train":
            states.add((q, m))
            species.update(z)
    rng = np.random.default_rng(723035)
    covered = set()
    for src, names in splits["train"].items():
        chosen = list(rng.choice(names, min(72, len(names)), replace=False))
        for name in chosen:
            z, r = conn.execute(
                "SELECT z,r FROM samples WHERE source=? AND name=?", (src, str(name))
            ).fetchone()
            seeds.append(
                (np.frombuffer(z, "u1"), np.frombuffer(r, "<f8").reshape(-1, 3))
            )
            covered.update(z)
    for s in species - covered:
        z, r = next(
            (z, r)
            for z, r in conn.execute("SELECT z,r FROM samples WHERE split='train'")
            if s in z
        )
        seeds.append((np.frombuffer(z, "u1"), np.frombuffer(r, "<f8").reshape(-1, 3)))
    conn.close()
    basis = Basis(seeds, sorted(species), states)
    print("sparse basis", basis.dim, "states", states, "elements", species, flush=True)
    # Independent check of the compressed descriptor and every sign against the
    # ordinary ForceField prediction and the uncompressed training Jacobian.
    z, r = min((p for p in seeds if len(p[0]) > 2), key=lambda p: len(p[0]))
    b = basis.design(z, r, 0, 1)
    v, j = basis.local._first_order(z, r, frozen=False)
    left = basis.offsets + basis.pairs
    np.testing.assert_allclose(
        b[:, left : left + basis.sites], np.vstack([v, -j]), atol=1e-8
    )
    allstats = {}
    timings = {}
    for mode in ("train", "dev", "test"):
        before = monotonic()
        tasks = [
            (mode, src, names[i : i + 64])
            for src, names in splits[mode].items()
            for i in range(0, len(names), 64)
        ]
        counts = Counter()
        stats = defaultdict(lambda: np.zeros(6))
        if mode == "train":
            gram = np.zeros((basis.dim + 1, basis.dim + 1))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=worker_init,
        ) as pool:
            for src, n, g, metrics in pool.map(process, tasks, buffersize=workers * 2):
                counts[src] += n
                if mode == "train":
                    active, partial = g
                    gram[np.ix_(active, active)] += partial
                else:
                    stats[src] += metrics
                if counts[src] % 2048 < n or counts[src] == len(splits[mode][src]):
                    print(
                        mode,
                        src,
                        counts[src],
                        len(splits[mode][src]),
                        round(monotonic() - before),
                        "seconds",
                        flush=True,
                    )
        if mode == "train":
            d = np.r_[
                np.full(basis.offsets, 1e-6),
                np.full(basis.pairs, 0.05),
                np.full(basis.sites, 0.005),
                np.full(len(basis.others) * len(basis.state_cols), 0.25),
            ]
            normal = gram[:-1, :-1].copy()
            normal.flat[:: len(normal) + 1] += d
            weights = cho_solve(cho_factor(normal), gram[:-1, -1])
            model = basis.model(weights)
            checkpoint = OUT / "pbe0-full.npz"
            model.save(checkpoint)
            # Validate the production serialization on the same molecule and
            # geometry as the training-design implementation.
            z, r = next((a, p) for a, p in seeds if (a.sum() % 2) == 0)
            check = basis.design(z, r, 0, 1) @ weights
            e, f = ForceField.load(checkpoint).predict(z, r, 0, 1)
            np.testing.assert_allclose(check, np.r_[e, f.ravel()], atol=1e-6)
        else:
            for name, (n, ae, se, nf, af, sf) in stats.items():
                allstats.setdefault(mode, {})[name] = dict(
                    n=int(n),
                    energy_mae_eV=ae / n,
                    energy_rmse_eV=np.sqrt(se / n),
                    force_mae_eV_A=af / nf,
                    force_rmse_eV_A=np.sqrt(sf / nf),
                )
        timings[mode] = monotonic() - before
        out.write_text(
            json.dumps(
                dict(
                    counts={
                        k: {s: len(n) for s, n in d.items()} for k, d in splits.items()
                    },
                    metrics=allstats,
                    seconds=timings,
                    elapsed=monotonic() - started,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=5)
    run(p.parse_args().workers)
