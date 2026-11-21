"""Encode private energy structures and all archived radical Hessian directions."""

import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from experiments.full_public import trial_sites as ts
from experiments.full_public.kernel_trials import WORK, OUT
from src.reactive_basis import ReactiveBasis
from src.forcefield import ForceField
from src.dispersion.d3 import d3_energy_forces


def read_frames():
    coords = []
    private_file = WORK / "leuven-data.pkl"
    if private_file.exists():
        with private_file.open("rb") as f:
            coords = pickle.load(f)[0]
    with (WORK / "radical-modes.pkl").open("rb") as f:
        rad = pickle.load(f)
    frames = [(z, r) for z, r, q, m in coords]
    meta = []
    for fold, (name, z, r, st, stationary, modes) in rad:
        import hashlib

        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        )
        d = rng.choice([-1.0, 1.0], size=r.shape) / np.sqrt(r.size)
        h = 0.004
        frames.extend([(z, r), (z, r + h * d), (z, r - h * d)])
        for v, kref, mass in modes:
            frames.extend([(z, r + h * v), (z, r - h * v)])
        meta.append(
            (fold, stationary, modes[0][1], modes[1][1], modes[0][2], modes[1][2])
        )
    return frames, len(coords), np.array(meta)


def prior(lo):
    out = []
    for z, r in radframes[lo : lo + 64]:
        x = parent.design(z, r, 0, 2)[0]
        e = parent.prior(z, r, 0, 2)[0] + d3_energy_forces(z, r)[0]
        out.append(e + coeff @ x)
    return lo, np.array(out)


def run(tag):
    global radframes, parent, coeff
    spec = (2, 2, 1.9, 3.2) if tag == "base" else (3, 3, 2.0, 4.2)
    ts.setup(spec)
    ts.descriptor_spec = spec
    with np.load(WORK / f"{tag}-basis.npz") as ar:
        ts.basis = ReactiveBasis.load(ar)
    ts.frames, npvt, meta = read_frames()
    if not (WORK / f"{tag}-aux-ready.json").exists():
        counts = np.array(
            [[np.count_nonzero(z == s) for s in ts.ACTIVE] for z, r in ts.frames]
        )
        ts.offsets = {}
        ts.sites = {}
        for j, s in enumerate(ts.ACTIVE):
            off = np.r_[0, np.cumsum(counts[:, j])]
            ts.offsets[s] = off
            np.save(WORK / f"{tag}-aux-off-{s}.npy", off)
            ts.sites[s] = np.lib.format.open_memmap(
                WORK / f"{tag}-aux-sites-{s}.npy",
                mode="w+",
                dtype=float,
                shape=(int(off[-1]), ts.basis.projections[s].shape[1]),
            )
        with ProcessPoolExecutor(
            max_workers=5, mp_context=mp.get_context("fork")
        ) as pool:
            tasks = [
                (i, min(i + 256, len(ts.frames))) for i in range(0, len(ts.frames), 256)
            ]
            for _ in pool.map(ts.encode_chunk, tasks, buffersize=12):
                pass
        for arr in ts.sites.values():
            arr.flush()
        (WORK / f"{tag}-aux-ready.json").write_text(
            json.dumps(dict(private=npvt, radical=len(meta)))
        )
    if not (WORK / "rad-prior.npy").exists():
        parent = ForceField.load(OUT / "joint-pbe0-reaction.npz")
        a, g, m = json.loads((WORK / "spin-choices.json").read_text())["wb97xd3"][
            "best"
        ]
        coeff = np.stack(
            [
                parent.weights + np.load(WORK / f"spin-wb97xd3-{h}-{a}-{g}-{m}.npy")
                for h in (0, 1)
            ]
            + [ForceField.load(WORK / "spin-wb97xd3.npz").weights]
        )
        radframes = ts.frames[npvt:]
        out = np.empty((len(radframes), 3))
        with ProcessPoolExecutor(
            max_workers=5, mp_context=mp.get_context("fork")
        ) as pool:
            for lo, v in pool.map(prior, range(0, len(radframes), 64), buffersize=12):
                out[lo : lo + len(v)] = v
        np.save(WORK / "rad-prior.npy", out.reshape(-1, 7, 3))
        np.save(WORK / "rad-meta.npy", meta)
    print(
        "Prepared",
        tag,
        npvt,
        "private structures",
        len(meta),
        "public radical Hessians",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
