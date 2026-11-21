"""Prepare and verify the bond-environment design on the original split."""

import ctypes
import multiprocessing as mp
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from experiments.full_public import trial_sites as sites
from experiments.full_public.kernel_trials import WORK, inputs
from experiments.full_public.trial_aux import read_frames
from src.bond_environment import BondBasis


def values(z, r):
    z = np.ascontiguousarray(z, dtype=np.int32)
    r = np.ascontiguousarray(r, dtype=float)
    out = np.zeros(basis.size)
    native(
        len(z),
        basis.width,
        basis.frequencies.shape[-1],
        z.ctypes.data,
        r.ctypes.data,
        basis.frequencies.ctypes.data,
        basis.phases.ctypes.data,
        out.ctypes.data,
    )
    return out


def process(lo):
    for i in range(lo, min(lo + 192, len(frames))):
        output[i] = values(*frames[i])


def encode(name, data):
    global frames, output
    frames = data
    dest = (
        WORK
        / f"bond-{basis.width}-{length:g}{'-angle' if basis.angular else ''}-{name}.npy"
    )
    tmp = dest.with_suffix(".part.npy")
    output = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=float, shape=(len(data), basis.size)
    )
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(process, range(0, len(data), 192), buffersize=14):
            pass
    output.flush()
    tmp.replace(dest)
    print("Completed", name, len(data), "geometries", flush=True)


def run(width, scale, angular=False):
    global length, basis, native
    length = scale
    basis = BondBasis.generate(width, length, angular)
    so = WORK / "trial_bonds.so"
    source = Path(__file__).with_name("trial_bonds.cpp")
    if not so.exists() or so.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            [
                "c++",
                "-O3",
                "-ffast-math",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(source),
                "-o",
                str(so),
            ],
            check=True,
        )
    native = ctypes.CDLL(str(so)).bond_values
    native.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5
    native.restype = None
    gs, _, _ = inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    data = [(gs[k].z, gs[k].r) for k in names]
    rng = np.random.default_rng(403)
    for i in rng.choice(len(data), 24, replace=False):
        z, r = data[i]
        v, f = basis.design(z, r, True)
        np.testing.assert_allclose(values(z, r), v, atol=1e-10)
        dx = rng.normal(size=r.shape)
        dx /= np.linalg.norm(dx)
        fd = (values(z, r - 2e-5 * dx) - values(z, r + 2e-5 * dx)) / 4e-5
        np.testing.assert_allclose(dx.ravel() @ f, fd, atol=5e-7)
        ix = rng.permutation(len(z))
        np.testing.assert_allclose(values(z[ix], r[ix] + 1), v, atol=1e-11)
    print(
        "Passed analytic force, invariance and training/deployment checks on 24 structures",
        flush=True,
    )
    for _, z, r, d in sites.hdf():
        data.extend(((z, r), (z, r + sites.STEP * d), (z, r - sites.STEP * d)))
    encode("main", data)
    del data
    encode("aux", read_frames()[0])


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), "--angle" in sys.argv[3:])
