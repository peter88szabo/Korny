"""Stream the original AIMNet2 closed-shell ωB97M archive in its deposited units.

Each group has one fixed atom count. Batches preserve molecular charge, the
closed-shell singlet state, atomic Hirshfeld charges and dispersion-free forces
and energies. A stoichiometry stays in one partition across charge states,
atom order and conformers. Reserve every formula present in existing Korny
reaction, paper and PBE0 geometry validation before fitting an auxiliary model.
"""

import json
import pickle
import sqlite3

import h5py
import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import DATA, DB

ROOT = DATA / "AIMNet2_NSE" / "closed_shell_parent"
ARCHIVE = ROOT / "aimnet2_wb97m.h5"
FIELDS = (
    "numbers",
    "coord",
    "charge",
    "energy",
    "forces",
    "charges",
    "dipole",
    "quadrupole",
)
# Deterministic additive hashing makes the partition invariant to atom order.
# The sums fit in uint64 for the largest (76-atom) systems in this archive.
SALT = np.random.default_rng(72154).integers(1, 2**52, 119, dtype=np.uint64)


def formula_keys(z):
    return SALT[z].sum(axis=-1, dtype=np.uint64)


def validation_formulas():
    """Prevent transfer learning on stoichiometries held out in the legacy tests."""
    # Existing auxiliary fits share this frozen chemical-formula split. A
    # retired benchmark can still reserve a formula without contributing a
    # single energy, gradient or barrier to an active fit or score.
    with (kt.OUT / "reaction-data.pkl").open("rb") as stream:
        g, rows, _ = pickle.load(stream)
    names = {
        n
        for _, _, split, _, terms, _, _ in rows
        if split == "validation"
        for n, _ in terms
    }
    keys = {formula_keys(g[n].z) for n in names}
    del g, rows
    for file, table, column in [
        (DB, "samples", "z"),
        (DATA / "paper_references/electronic.sqlite", "geometries", "z"),
    ]:
        with sqlite3.connect(f"file:{file}?mode=ro", uri=True) as conn:
            for (raw,) in conn.execute(
                f"SELECT DISTINCT {column} FROM {table} WHERE split='validation'"
            ):
                keys.add(formula_keys(np.frombuffer(raw, np.uint8)))
    return np.asarray(sorted(keys), np.uint64)


def blocks(handle, fields=FIELDS, groups=None, rows=2048):
    """Return (source group, first row, field arrays) with stable original IDs."""
    for name in sorted(handle) if groups is None else groups:
        g = handle[name]
        n = len(g["energy"])
        if any(len(g[k]) != n for k in fields) or g["numbers"].shape != (n, int(name)):
            raise ValueError(f"Inconsistent sample/atom axes in group {name}")
        for lo in range(0, n, rows):
            yield name, lo, {k: g[k][lo : lo + rows] for k in fields}


def heldout(z, protected):
    key = formula_keys(z)
    return (key % 19 == 0) | np.isin(key, protected)


def verify_schema():
    """Return exact geometry and atom counts from the publisher's complete file."""
    expected = json.loads((ROOT / "schema.json").read_text())
    with h5py.File(ARCHIVE, "r") as h:
        current = [
            [k, {name: [list(d.shape), str(d.dtype)] for name, d in h[k].items()}]
            for k in sorted(h)
        ]
        if current != expected:
            raise ValueError(
                "The downloaded archive differs from the recorded source schema"
            )
        return sum(len(h[k]["energy"]) for k in h), sum(
            int(k) * len(h[k]["energy"]) for k in h
        )
