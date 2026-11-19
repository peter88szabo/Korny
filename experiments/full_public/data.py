"""Locations and rotation-invariant geometry IDs for the requested public fit."""

import hashlib
import numpy as np
from scipy.spatial.distance import pdist
from experiments.des15k.run import ROOT

DATA = ROOT / "dataset_for_training"
OUT = DATA / "requested_public"
DB = OUT / "pbe0.sqlite"
H5 = {
    "QM7X": DATA / "QM7_X/8000_starting_set/8000.hdf5",
    "DES15K": DATA / "DES15K/DES15k.hdf5",
    "SPICE": DATA / "DES15K/SPICE_dipeptides.hdf5",
    "TorsionNet": DATA / "DES15K/TorsionNet500.hdf5",
}


def fingerprint(z, r):
    """Rotation-, translation- and atom-order-independent geometry signature."""
    i, j = np.triu_indices(len(z), 1)
    pairs = 128 * np.minimum(z[i].astype(int), z[j]) + np.maximum(z[i], z[j])
    d = np.rint(pdist(r) * 10000).astype(np.int32)
    order = np.lexsort((d, pairs))
    digest = hashlib.sha256(np.sort(z).astype("u1").tobytes())
    digest.update(pairs[order].astype("<u2").tobytes())
    digest.update(d[order].astype("<i4").tobytes())
    return digest.digest()
