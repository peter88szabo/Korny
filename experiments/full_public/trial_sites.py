"""Stream projected atomic environments for kernel trials under the shared RAM limit.

For every HDF5 training structure store a center and a symmetric displacement
along a reproducible isotropic direction. Their difference constrains the force
change of the new correction at every training geometry without storing millions
of full feature Jacobians. The production predictor uses analytic derivatives.
"""

import ctypes as ct
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from experiments.full_public.data import DB, OUT
from experiments.full_public.kernel_trials import WORK, inputs
from src.forcefield import ForceField
from src.local import descriptor_array, descriptor_layout
from src.reactive_basis import ReactiveBasis

ACTIVE = (1, 6, 7, 8)
STEP = 0.001


def setup(spec):
    global lib, species, types, indices, width
    parent = ForceField.load(OUT / "joint-high-reaction.npz")
    species = parent.kernel.species
    types = np.full(128, -1, np.int32)
    types[list(species)] = np.arange(len(species))
    nr, lmax = spec[:2]
    widths, groups = descriptor_layout(len(species), nr, lmax)
    width = sum(widths)
    ch = nr * len(species)
    indices = np.full((lmax, ch, ch), -1, np.int32)
    for l, (pos, a, b) in enumerate(groups):
        indices[l, a, b] = pos
    so = WORK / "trial_raw.so"
    source = Path(__file__).with_name("trial_descriptors.cpp")
    if not so.exists() or source.stat().st_mtime > so.stat().st_mtime:
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                str(source),
                "-o",
                str(so),
            ],
            check=True,
        )
    lib = ct.CDLL(str(so)).raw_sites
    lib.argtypes = [ct.c_int] * 5 + [ct.c_double] * 2 + [ct.c_void_p] * 4
    return parent


def raw(z, r, spec):
    nr, ell, ron, rc = spec
    t = np.ascontiguousarray(types[z])
    rr = np.ascontiguousarray(r)
    out = np.zeros((len(z), width))
    lib(
        len(z),
        nr * len(species),
        nr,
        ell,
        width,
        ron,
        rc,
        t.ctypes.data,
        rr.ctypes.data,
        indices.ctypes.data,
        out.ctypes.data,
    )
    return out


def candidates(gs, rows, classes):
    """Collect training-only endpoint and saddle environments by reaction family."""
    pools = defaultdict(set)
    for target, src, role, name, terms, y, kind in rows:
        if role != "train" or kind != "forward":
            continue
        ts = terms[-1][0]
        comp = tuple(sorted(map(int, gs[ts].z)))
        h = (
            int.from_bytes(
                hashlib.sha256(("kt1/" + str(comp)).encode()).digest()[:4], "big"
            )
            % 8
        )
        if h in (0, 1):
            continue
        family = classes.get(name, src)
        for k, c in terms:
            pools[src, family].add(k)
    return sorted(
        {
            k
            for ks in pools.values()
            for k in sorted(ks, key=lambda x: hashlib.sha256(x.encode()).digest())[:128]
        }
    )


def make_basis(spec, gs, rows, classes):
    parent = setup(spec)
    from experiments.reactivity.reaction_classes import lookup

    classes.update(lookup())
    keys = candidates(gs, rows, classes)
    pool = {s: [] for s in ACTIVE}
    for k in keys:
        g = gs[k]
        a = raw(g.z, g.r, spec)
        for s in ACTIVE:
            if np.any(g.z == s):
                pool[s].append(a[g.z == s])
    pool = {s: np.vstack(v) for s, v in pool.items()}
    projections = {}
    rad = {}
    centers = {}
    same = spec == (2, 2, 1.9, 3.2)
    for s, arr in pool.items():
        if same:
            p = parent.kernel.projections[s]
            rd = parent.kernel.radial_dims[s]
        else:
            blocks = []
            for v in (
                arr[:, : len(species) * spec[0]],
                arr[:, len(species) * spec[0] :],
            ):
                vals, vecs = np.linalg.eigh(np.cov(v, rowvar=False))
                keep = np.flatnonzero(vals > 1e-7)[-12:]
                blocks.append(vecs[:, keep] / np.sqrt((vals[keep] + 0.02) * len(keep)))
            p = block_diag(*blocks)
            rd = blocks[0].shape[1]
        projections[s] = p
        rad[s] = rd
        x = arr @ p
        # Start with distance to the deployed centers if the projection is identical.
        # Then each addition fills the largest remaining hole in the candidate pool.
        if same:
            dist = cdist(x, parent.kernel.centers[s], "sqeuclidean").min(axis=1)
        else:
            dist = np.square(x - x.mean(axis=0)).sum(axis=1)
        chosen = []
        for _ in range(192 if same else 256):
            pick = int(dist.argmax())
            chosen.append(x[pick])
            dist = np.minimum(dist, np.sum((x - x[pick]) ** 2, axis=1))
        centers[s] = (
            np.vstack((parent.kernel.centers[s], chosen)) if same else np.array(chosen)
        )
    return ReactiveBasis(
        species,
        spec,
        projections,
        centers,
        rad,
        1.0,
        1.0,
        np.zeros(sum(len(c) for c in centers.values())),
    )


def hdf(include_state=False):
    records = []
    seen = set()
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        for src, fp, q, m, bz, br in db.execute(
            "SELECT source,fp,q,m,z,r FROM samples WHERE split='train' ORDER BY source,name"
        ):
            uid = fp, q, m
            if uid in seen:
                continue
            seen.add(uid)
            z = np.frombuffer(bz, "u1")
            r = np.frombuffer(br, "<f8").reshape(-1, 3)
            rng = np.random.default_rng(int.from_bytes(fp[:8], "little"))
            direction = rng.choice([-1.0, 1.0], size=r.shape) / np.sqrt(r.size)
            row = (src, z, r, direction)
            records.append((*row, q, m) if include_state else row)
    assert len(records) == 91156
    return records


def encode_chunk(bounds):
    start, stop = bounds
    for i in range(start, stop):
        z, r = frames[i]
        a = raw(z, r, descriptor_spec)
        for s in ACTIVE:
            lo, hi = offsets[s][i : i + 2]
            if hi > lo:
                sites[s][lo:hi] = a[z == s] @ basis.projections[s]
    return stop - start


def prepare(spec, tag):
    global frames, offsets, sites, descriptor_spec, basis
    descriptor_spec = spec
    file = WORK / f"{tag}-ready.json"
    if file.exists():
        return
    gs, rows, classes = inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        ids, names = pickle.load(f)
    basis = make_basis(spec, gs, rows, classes)
    g = next(v for v in gs.values() if len(v.z) > 8 and set(v.z) <= set(species))
    ref = descriptor_array(g.r, g.z, 0, **basis.spec)[0]
    np.testing.assert_allclose(raw(g.z, g.r, spec)[0], ref, atol=2e-12)
    np.savez(WORK / f"{tag}-basis.npz", **basis.fields())
    frames = [(gs[k].z, gs[k].r) for k in names]
    db = hdf()
    np.save(WORK / "hdf-sources.npy", np.array([s for s, *_ in db]))
    # Use all train geometries. The three HDF rows in each group are indexed
    # center, plus, minus, so caches serve every bandwidth without rereading HDF.
    for _, z, r, direction in db:
        frames.extend([(z, r), (z, r + STEP * direction), (z, r - STEP * direction)])
    counts = np.array([[np.count_nonzero(z == s) for s in ACTIVE] for z, r in frames])
    offsets = {}
    sites = {}
    for j, s in enumerate(ACTIVE):
        offsets[s] = np.r_[0, np.cumsum(counts[:, j])]
        np.save(WORK / f"{tag}-off-{s}.npy", offsets[s])
        sites[s] = np.lib.format.open_memmap(
            WORK / f"{tag}-sites-{s}.npy",
            mode="w+",
            dtype=float,
            shape=(int(offsets[s][-1]), basis.projections[s].shape[1]),
        )
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        tasks = [(i, min(i + 256, len(frames))) for i in range(0, len(frames), 256)]
        done = 0
        for n in pool.map(encode_chunk, tasks, buffersize=14):
            done += n
    for a in sites.values():
        a.flush()
    from experiments.full_public.kernel_trials import write_json

    write_json(file, dict(molecular=len(names), hdf=len(db), settings=spec))
    print(
        "Cached",
        tag,
        len(names),
        "reaction geometries and",
        len(db),
        "PBE0 anchor geometries",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    tag = sys.argv[1]
    prepare((2, 2, 1.9, 3.2) if tag == "base" else (3, 3, 2.0, 4.2), tag)
