"""Fit local first and second Gaussian derivatives along reacting environments.

The radical IRC coordinates preserve atom correspondence on both branches.
For singlets, a signed eigenvector of the archived RDB7 Hessian supplies the
local direction of motion at a saddle. The deployed potential evaluates the
resulting kernel centers from a single structure.
"""

import json
import sqlite3
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_sites
from experiments.full_public.trial_adaptive_spectrum import (
    geometry_atoms,
    json_count,
    tables,
)
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.trial_path_directions import fit_surfaces
from experiments.full_public.trial_path_scales import ROOT as NATIVE
from experiments.reactivity.stationary_fit import internal_transform
from src.path_hermite import PathHermiteKernel
from src.reactive_basis import ReactiveBasis

ROOT = kt.WORK / "path-hermite58"
SPEC = (3, 3, 2.0, 4.2)


def learn():
    trial_sites.setup(SPEC)
    with np.load(kt.WORK / "rich-basis.npz") as f:
        base = ReactiveBasis.load(f)
    pools = defaultdict(list)
    counts = defaultdict(int)

    def observe(z, r1, r2, source):
        # Same atom ordering is required within one pair only. Center choice
        # never depends on an energy residual or a validation reaction.
        raw1, raw2 = (
            trial_sites.raw(z, np.asarray(r, dtype=float), SPEC) for r in (r1, r2)
        )
        candidates = []
        for element, projection in base.projections.items():
            use = z == element
            if not np.any(use):
                continue
            start, end = raw1[use] @ projection, raw2[use] @ projection
            change = end - start
            norms = np.linalg.norm(change, axis=1)
            idx = np.argmax(norms)
            if norms[idx] <= 1e-6:
                continue
            tangent = change[idx] / norms[idx]
            # Fix the arbitrary sign of a normal mode before comparing centers.
            tangent *= np.sign(tangent[np.argmax(abs(tangent))])
            candidates.append(
                (norms[idx], element, np.r_[(start[idx] + end[idx]) / 2, tangent])
            )
        for _, element, row in sorted(candidates, key=lambda v: v[0], reverse=True)[:2]:
            pools[source, element].append(row)
        counts[source] += 1

    records = json.loads((NATIVE / "snapshot.json").read_text())
    for record in records:
        if record["heldout"]:
            continue
        for side in ("forw", "rev"):
            with np.load(IRC / f"{record['rxn']}-{side}.npz") as f:
                r = f["r_A"]
                observe(f["z"], r[0], r[-1], "radical_branches")
    excluded = set()
    for target in kt.TARGETS:
        s, _, _, fold, _, _ = kt.equation_data(target)
        excluded.update(s[fold < 0].indices)
    masses = {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915}
    with sqlite3.connect(
        f"file:{kt.WORK / 'rdb-dft-frames.sqlite'}?mode=ro", uri=True
    ) as db:
        seen = set()
        for idx, zbuf, rbuf, hbuf in db.execute(
            """SELECT coord_id,z,r,hessian FROM stationary
               WHERE split='train' AND m=1 AND converged=1 AND name LIKE '%/ts%'
                 AND hessian IS NOT NULL ORDER BY coord_id,name"""
        ):
            if idx in excluded or idx in seen:
                continue
            seen.add(idx)
            z = np.frombuffer(zbuf, np.uint8)
            r = np.frombuffer(rbuf, float).reshape(-1, 3)
            hessian = np.frombuffer(hbuf, float).reshape(r.size, r.size)
            transform = internal_transform(np.array([masses[int(k)] for k in z]), r)
            eig, modes = np.linalg.eigh(transform.T @ hessian @ transform)
            if not len(eig) or eig[0] >= -1e-4:
                continue
            direction = (transform @ modes[:, 0]).reshape(r.shape)
            direction *= 0.10 / np.linalg.norm(direction)
            observe(z, r - direction, r + direction, "singlet_saddles")
    rng = np.random.default_rng(5814)
    found = defaultdict(list)
    for (_, element), entries in sorted(pools.items()):
        data = np.stack(entries)
        d = data.shape[1] // 2
        # Balance singlet and doublet centers. Spread them over both the
        # environment and the direction in which it changes.
        inputs = data.copy()
        inputs[:, d:] *= 0.45
        dist = np.square(inputs - inputs.mean(axis=0)).sum(axis=1)
        for _ in range(min(32, len(data))):
            if dist.sum() < 1e-10:
                break
            j = rng.choice(len(data), p=dist / dist.sum())
            found[element].append(data[j])
            dist = np.minimum(dist, np.square(inputs - inputs[j]).sum(axis=1))
    centers = {}
    for element, rows in sorted(found.items()):
        x = np.stack(rows)
        coords = x[:, : x.shape[1] // 2]
        neighbors = cKDTree(coords).query(coords, k=min(7, len(coords)))[0]
        sigma = np.clip(np.atleast_2d(neighbors)[:, -1], 0.35, 1.75)
        centers[element] = np.vstack(
            [np.column_stack([x, sigma, np.full(len(x), parity)]) for parity in (0, 1)]
        )
    count = sum(map(len, centers.values()))
    model = PathHermiteKernel(
        base.species,
        SPEC,
        base.projections,
        centers,
        base.radial_dims,
        1.0,
        1.0,
        np.zeros(count),
    )
    np.savez_compressed(ROOT / "basis.npz", **model.fields())
    info = dict(counts) | {
        "centers_by_element": {int(k): len(v) // 2 for k, v in centers.items()},
        "features": count,
    }
    kt.write_json(ROOT / "basis.json", info)
    print("Local directions fitted", info, flush=True)
    return model


def encode(model, auxiliary):
    maps = tables(auxiliary)
    skip = json_count() if auxiliary else 0
    n = len(next(iter(maps.values()))[0]) - 1 - skip
    path = ROOT / ("rad.npy" if auxiliary else "main.npy")
    if path.is_file():
        return np.load(path, mmap_mode="r")
    tmp = path.with_suffix(".pending")
    output = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=float, shape=(n, model.size)
    )
    for lo in range(0, n, 256):
        ids = np.arange(lo + skip, min(lo + 256, n) + skip)
        begin = 0
        block = np.zeros((len(ids), model.size))
        for element, centers in model.centers.items():
            end = begin + len(centers)
            offsets, sites = maps[element]
            idx, _, present, starts = geometry_atoms(offsets, ids)
            if len(idx):
                local, _ = model.values(element, sites[idx])
                block[present, begin:end] = np.add.reduceat(local, starts, axis=0)
            begin = end
        output[lo : lo + len(ids)] = block
    output.flush()
    del output
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


def run():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "basis.json").exists():
        with np.load(ROOT / "basis.npz") as f:
            basis = PathHermiteKernel.load(f)
    else:
        basis = learn()
    info = json.loads((ROOT / "basis.json").read_text())
    main = encode(basis, False)
    rad = encode(basis, True).reshape(-1, 7, basis.size)
    fit_surfaces(
        [(basis, np.eye(basis.size))], main, rad, info, "p58-public", "hermite58"
    )


if __name__ == "__main__":
    run()
