"""Compress every requested HDF5 training energy and force into the final basis.

The matrix product and right-hand side are sufficient statistics of the exact
linear loss. Sparse center selection limits the basis size only. Every distinct
valid geometry/state and every Cartesian force component enter the regression.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sqlite3
from time import monotonic

import numpy as np

from experiments.full_public.specified import DB, OUT
from src.forcefield import ForceField


def init():
    global conn, model
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    model = ForceField.load(OUT / "pbe0-reaction.npz")


def compress(task):
    source, names = task
    dim = len(model.weights)
    gram = np.zeros((dim + 1, dim + 1))
    blocks = defaultdict(list)
    nforce = 0

    def flush():
        for packed, parts in blocks.items():
            idx = np.frombuffer(packed, dtype=np.int32)
            a = np.vstack(parts)
            gram[np.ix_(idx, idx)] += a.T @ a
        blocks.clear()

    for i, name in enumerate(names):
        bz, br, q, m, e, bf = conn.execute(
            "SELECT z,r,q,m,energy,force FROM samples WHERE source=? AND name=?",
            (source, name),
        ).fetchone()
        z = np.frombuffer(bz, "u1").astype(int)
        r = np.frombuffer(br, "<f8").reshape(-1, 3)
        f = np.frombuffer(bf, "<f8").reshape(-1, 3)
        v, j = model.design(z, r, q, m, True)
        base, df = model.prior(z, r, q, m, True)
        obs = np.empty((r.size + 1, dim + 1))
        obs[0, :-1] = v
        obs[1:, :-1] = j
        obs[:, -1] = np.r_[e - base, (f - df).ravel()]
        obs[1:] *= 4.0 / np.sqrt(r.size)
        ids = np.flatnonzero(np.any(obs != 0.0, axis=0)).astype(np.int32)
        blocks[ids.tobytes()].append(obs[:, ids])
        nforce += r.size
        if (i + 1) % 48 == 0:
            flush()
    flush()
    ix = np.flatnonzero(gram.diagonal())
    return source, len(names), nforce, ix, gram[np.ix_(ix, ix)]


def run():
    t = monotonic()
    conn = sqlite3.connect(DB)
    by = defaultdict(list)
    seen = set()
    for s, k, fp, q, m in conn.execute(
        "SELECT source,name,fp,q,m FROM samples WHERE split='train' ORDER BY source,name"
    ):
        if (fp, q, m) in seen:
            continue
        seen.add((fp, q, m))
        by[s].append(k)
    conn.close()
    dim = len(ForceField.load(OUT / "pbe0-reaction.npz").weights)
    done = Counter()
    ncomp = 0
    gram = np.zeros((dim + 1, dim + 1))
    tasks = [
        (s, keys[i : i + 256])
        for s, keys in by.items()
        for i in range(0, len(keys), 256)
    ]
    with ProcessPoolExecutor(
        max_workers=7, mp_context=mp.get_context("fork"), initializer=init
    ) as pool:
        for s, n, nf, idx, g in pool.map(compress, tasks, buffersize=14):
            # Equalize total weight across the four sources while retaining
            # equal per-geometry weight within a source.
            gram[np.ix_(idx, idx)] += g * (4500.0 / len(by[s]))
            done[s] += n
            ncomp += nf
            if done[s] % 4096 < n or done[s] == len(by[s]):
                print(
                    s,
                    done[s],
                    len(by[s]),
                    "elapsed",
                    round(monotonic() - t),
                    "s",
                    flush=True,
                )
    np.save(OUT / "joint-gram.npy", gram)
    print(
        "all HDF5 rows",
        dict(done),
        "force components",
        ncomp,
        "seconds",
        monotonic() - t,
        flush=True,
    )


if __name__ == "__main__":
    run()
