"""Compile and verify the compact per-atom descriptors for TS volume fitting."""

import ctypes as ct
import subprocess
from pathlib import Path
import numpy as np
from rdkit import Chem
from experiments.des15k.run import SPECIES
from experiments.full_public.data import OUT
from src.local import descriptor_array

SPEC = dict(nr=2, lmax=1, ron=2.0, rc=3.8, species=SPECIES)
WIDTH = 152


def accelerator():
    source = Path(__file__).with_name("volume_features.cpp")
    target = OUT / "volume_features.so"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(target),
        ],
        check=True,
    )
    fun = ct.CDLL(str(target)).features
    fun.argtypes = [ct.c_int] + [ct.c_void_p] * 6
    radii = np.array(
        [0.0] + [Chem.GetPeriodicTable().GetRcovalent(s) for s in range(1, 119)]
    )
    types = np.full(128, -1, dtype=np.int32)
    for i, s in enumerate(SPECIES):
        types[s] = i

    def compute(z, r, features):
        out = np.empty((len(z), WIDTH)) if features else None
        h = np.empty(3, dtype=np.uint64)
        status = fun(
            len(z),
            z.ctypes.data,
            r.ctypes.data,
            radii.ctypes.data,
            types.ctypes.data,
            h.ctypes.data,
            out.ctypes.data if features else None,
        )
        if status:
            raise ValueError(f"Invalid native volume geometry, status {status}")
        return int(h[0]), h[1:].tobytes(), out

    z = np.array([6, 1, 8, 7, 1], dtype="u1")
    r = np.random.default_rng(521).normal(size=(len(z), 3))
    _, _, v = compute(z, r, True)
    np.testing.assert_allclose(
        v, [descriptor_array(r, z, i, **SPEC)[0] for i in range(len(z))], atol=1e-12
    )
    return compute
