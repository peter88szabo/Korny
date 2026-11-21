"""Build four-body trial features in the same order as the public and private kernels."""

import ctypes
import multiprocessing as mp
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.full_public import trial_sites as sites
from experiments.full_public.trial_aux import read_frames
from experiments.full_public.kernel_trials import WORK, inputs
from src.four_body import four_body


def values(z, r):
    z = np.ascontiguousarray(z, dtype=np.int32)
    r = np.ascontiguousarray(r, dtype=float)
    result = np.zeros(12)
    native(len(z), z.ctypes.data, r.ctypes.data, result.ctypes.data)
    return result


def process(start):
    for i in range(start, min(start + 256, len(frames))):
        destination[i] = values(*frames[i])


def encode(tag, data):
    global destination, frames
    frames = data
    path = WORK / f"four-{tag}.npy"
    tmp = path.with_suffix(".part.npy")
    destination = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=float, shape=(len(data), 12)
    )
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(process, range(0, len(data), 256), buffersize=12):
            pass
    destination.flush()
    tmp.replace(path)
    print("Four-body features", tag, len(data), flush=True)


def run():
    global native
    src = Path(__file__).with_name("trial_four.cpp")
    so = WORK / "trial_four.so"
    subprocess.run(
        ["c++", "-std=c++17", "-O3", "-fPIC", "-shared", str(src), "-o", str(so)],
        check=True,
    )
    native = ctypes.CDLL(str(so)).four_values
    native.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 3
    native.restype = None
    gs, _, _ = inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    data = [(gs[k].z, gs[k].r) for k in names]
    # Check both symmetry contractions, including exchanged atoms and the force sign.
    rng = np.random.default_rng(42)
    for i in rng.choice(len(data), 24, replace=False):
        z, r = data[i]
        v, f = four_body(z, r, True)
        np.testing.assert_allclose(values(z, r), v, atol=1e-11)
        dx = rng.normal(size=r.shape)
        dx /= np.linalg.norm(dx)
        fd = (values(z, r - 1e-5 * dx) - values(z, r + 1e-5 * dx)) / 2e-5
        np.testing.assert_allclose(dx.ravel() @ f, fd, atol=1e-7)
        ix = rng.permutation(len(z))
        np.testing.assert_allclose(values(z[ix], r[ix] + 1), v, atol=1e-11)
    for _, z, r, d in sites.hdf():
        data.extend(((z, r), (z, r + sites.STEP * d), (z, r - sites.STEP * d)))
    encode("main", data)
    del data
    encode("aux", read_frames()[0])


if __name__ == "__main__":
    run()
