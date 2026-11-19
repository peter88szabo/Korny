"""Fit the bare DES15k PBE0 surface with bounded-memory least squares."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from io import BytesIO
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.linalg import solve_triangular

from src.electronic import ElectronicCorrection, ElectronicFeatures, state_key, validate_electrons
from src.forcefield import ForceField
from src.local import Molecules
from src.pair import PairBaseline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset_for_training/DES15K"
OUT = Path(__file__).resolve().parent
# eV, from the release's hdf5_to_extxyz.py, used also by the official test set.
ATOMIC = {1: -13.6414041617373, 6: -1027.60791501338, 7: -1484.27481908749,
          8: -2039.75030551381, 9: -2710.54734320680, 15: -9283.01120605055,
          16: -10828.7228945312, 17: -12516.4600457922}
SPECIES = tuple(ATOMIC)
SEED = 15107


@dataclass
class Sample:
    name: str
    parent: str
    z: np.ndarray
    r: np.ndarray
    charge: int
    energy: float
    force: np.ndarray
    energy_weight: float = 1.
    multiplicity: int | None = 1


def electronic_labels(group):
    """Prefer explicit molecular state metadata in labelled HDF5 records.

    The original DES, QM7-X and dipeptide releases used by this experiment
    supply Hirshfeld charges and lack explicit spin metadata. Total partial charge
    is used only when an explicit total charge is absent. A new radical record
    must carry its multiplicity, since electron parity alone cannot identify it.
    """
    def scalar(name):
        item = group[name][()] if name in group else group.attrs.get(name)
        if item is None:
            return None
        if np.size(item) != 1:
            raise ValueError(f"Expected a scalar {name} at {group.name}")
        return np.asarray(item).item()

    charge = scalar("charge")
    if charge is None:
        total = float(np.sum(group["hCHG"][:]))
        charge = round(total)
        if not np.isclose(total, charge, atol=.05, rtol=0):
            raise ValueError(f"Ambiguous charge at {group.name}: {total}")
    multiplicity = scalar("multiplicity")
    if multiplicity is None:
        multiplicity = scalar("spin_multiplicity")
    numbers = group["atNUM"][:]
    state = state_key(charge, 1 if multiplicity is None else multiplicity)
    if multiplicity is None and (int(numbers.sum()) - state[0]) % 2:
        return state[0], None
    validate_electrons(numbers, *state)
    return state


def read_des(path):
    """Read bare electronic energies and forces, in eV and eV/Å.

    Fixed isolated-atom energies shift the absolute electronic energy zero;
    all atom-balanced reaction differences are invariant to this shift.
    """
    rows = []
    with h5py.File(path) as f:
        for parent in f.values():
            for key, g in parent.items():
                z = g["atNUM"][:]
                raw = float(g["ePBE0"][0])
                forces = g["pbe0FOR"][:]
                charge, multiplicity = electronic_labels(g)
                rows.append(Sample(key, key.rsplit("-", 1)[0], z, g["atXYZ"][:],
                                   charge, raw - sum(ATOMIC[a] for a in z), forces,
                                   multiplicity=multiplicity))
    return rows


def split(rows, rng):
    """Keep every displacement of a DES parent in the same partition."""
    parents = sorted({r.parent for r in rows})
    rng.shuffle(parents)
    test = set(parents[:round(len(parents)*0.2)])
    # Partition all source IDs first to reproduce the original holdout, then mask
    # odd-electron records whose multiplicity is missing in the released file.
    labelled = [r for r in rows if r.multiplicity is not None]
    return [r for r in labelled if r.parent not in test], [r for r in labelled if r.parent in test]


def design(sample, pair, local, states=(), electronic=None, charge_columns=(-1, 1)):
    """Build one energy row and 3N negative energy-derivative rows.

    Force weighting is per geometry, so a large dimer does not automatically
    outweigh a small one. The last column contains the observed PBE0 values.
    """
    z, r = sample.z, sample.r
    p, pf = pair.design(z, r)
    g, dg = local._first_order(z, r, frozen=False)
    composition = [np.count_nonzero(z == s) for s in SPECIES]
    key = state_key(sample.charge, sample.multiplicity)
    validate_electrons(z, *key)
    offsets = [sample.charge == q for q in charge_columns]
    erow = np.r_[composition, offsets, p, g]
    force_rows = np.column_stack((np.zeros((r.size, len(composition) + len(offsets))), pf, -dg))
    if states:
        # Independent charge/spin interactions apply to both E and -dE/dR.
        # The shared backbone supplies the reference surface, usually (0,1).
        ev = np.zeros((len(states), electronic.size))
        ef = np.zeros((r.size, len(states), electronic.size))
        if key in states:
            j = states.index(key)
            ev[j], ef[:, j] = electronic.design(z, r)
        erow = np.r_[erow, ev.ravel()]
        force_rows = np.column_stack((force_rows, ef.reshape(r.size, -1)))
    scale = 4. / np.sqrt(r.size)
    block = np.vstack((sample.energy_weight*np.r_[erow, sample.energy],
                       scale*np.column_stack((force_rows, sample.force.ravel()))))
    if not np.isfinite(block).all():
        raise ValueError(f"Non-finite PBE0 training data for {sample.name}")
    return block


def fit(train, rng, out, *, center_count=48):
    """Jointly regress charge/composition offsets, radial pairs and Gaussian sites.

    Train-only environments set the centers. Block QR keeps memory independent
    of the number of geometries. The same coefficients predict E and -dE/dR.
    """
    started = perf_counter()
    # Select centers only among training geometries; include every central species.
    candidates = list(rng.choice(len(train), min(350, len(train)), replace=False))
    covered = set(np.concatenate([train[i].z for i in candidates]))
    for s in set(SPECIES) - covered:
        candidates.append(next(i for i, row in enumerate(train) if s in row.z))
    local = Molecules([(train[i].z, train[i].r) for i in candidates],
                      dict(species=SPECIES, nr=2, lmax=2, ron=2., rc=4.2),
                      M=center_count, ell=5., scale_floor=.1)
    pair = PairBaseline(SPECIES, np.r_[np.arange(.7, 3.5, .18), 3.7, 4.2, 5., 6., 7.],
                        .22, 5.5, 8.)
    # QR compresses the augmented [design | PBE0 target] without squaring its condition number.
    n_pair = len(pair.pairs)*len(pair.centers)
    observed = sorted({state_key(s.charge, s.multiplicity) for s in train})
    reference = (0, 1) if (0, 1) in observed else observed[0]
    states = tuple(k for k in observed if k != reference)
    charge_columns = sorted({q for q, _ in observed if q != 0})
    electronic = ElectronicFeatures(SPECIES)
    n_offset = len(SPECIES) + len(charge_columns)
    n_local = sum(h.size for h in local.heads.values())
    diagonal = np.sqrt(np.r_[np.full(n_offset, 1e-7), np.full(n_pair, .05),
                             np.full(n_local, .005),
                             np.full(len(states)*electronic.size, .25)])
    reduced = np.column_stack((np.diag(diagonal), np.zeros(len(diagonal))))
    batch, nrows = [], 0
    for idx, sample in enumerate(train, 1):
        block = design(sample, pair, local, states, electronic, charge_columns)
        batch.append(block)
        nrows += len(block)
        if nrows >= 8192 or idx == len(train):
            reduced = np.linalg.qr(np.vstack([reduced, *batch]), mode="r")
            batch.clear()
            nrows = 0
        if idx % 1000 == 0:
            print(f"fit {idx}/{len(train)} in {perf_counter()-started:.1f}s", flush=True)
    n = len(diagonal)
    coeff = solve_triangular(reduced[:n, :n], reduced[:n, n])
    fit_seconds = perf_counter()-started
    pair.weights = coeff[n_offset:n_offset+n_pair]
    local.w = coeff[n_offset+n_pair:n_offset+n_pair+n_local]
    local.freeze()
    charge_offsets = {q: 0. for q, _ in observed}
    charge_offsets.update(zip(charge_columns, coeff[len(SPECIES):n_offset], strict=True))
    bundle = ForceField(pair, local, coeff[:len(SPECIES)], charge_offsets, "PBE0")
    weights = {reference: np.zeros(electronic.size)}
    weights.update(zip(states, coeff[n_offset+n_pair+n_local:].reshape(len(states), electronic.size),
                       strict=True))
    bundle.electronic = ElectronicCorrection(SPECIES, observed, [weights[k] for k in observed], "PBE0")
    # An immutable filename also identifies the model to Smite's Hessian cache.
    buf = BytesIO()
    bundle.save(buf)
    content = buf.getvalue()
    name = "pbe0-" + hashlib.sha256(content).hexdigest()[:16] + ".npz"
    (out/name).write_bytes(content)
    return name, fit_seconds, n


def metrics(energy_errors, forces_abs, forces_sq, ncomp, natoms):
    e = np.asarray(energy_errors)
    return dict(geometries=len(e), force_components=ncomp,
                energy_mae_eV=float(np.mean(np.abs(e))), energy_rmse_eV=float(np.sqrt(np.mean(e**2))),
                energy_per_atom_mae_eV=float(np.mean(np.abs(e)/natoms)),
                force_mae_eV_A=float(forces_abs/ncomp), force_rmse_eV_A=float(np.sqrt(forces_sq/ncomp)))


def evaluate(rows, model):
    e, fa, fs, ncomp, na = [], 0., 0., 0, []
    for row in rows:
        energy, force = model.predict(row.z, row.r, row.charge, row.multiplicity)
        e.append(energy-row.energy)
        df = force-row.force
        fa += np.abs(df).sum()
        fs += np.square(df).sum()
        ncomp += df.size
        na.append(len(row.z))
    return metrics(e, fa, fs, ncomp, na)


def run(data, out):
    out.mkdir(exist_ok=True, parents=True)
    rows = read_des(data)
    rng = np.random.default_rng(SEED)
    train, test = split(rows, rng)
    def counts(rows):
        return dict(structures=len(rows), parents=len({r.parent for r in rows}),
                    charges=dict(sorted(Counter(r.charge for r in rows).items())))
    print("DES split", counts(train), counts(test), flush=True)
    name, seconds, dimension = fit(train, rng, out)
    report = dict(seed=SEED, unlabelled_spin_excluded=[r.name for r in rows if r.multiplicity is None],
                  training=counts(train), test=counts(test), fit_seconds=seconds,
                  coefficients_per_surface=dimension, checkpoint=name, energy_target="ePBE0",
                  force_target="pbe0FOR", dispersion="off",
                  metrics=evaluate(test, ForceField.load(out/name)))
    (out/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"test_parents.txt").write_text("".join(p+"\n" for p in sorted({r.parent for r in test})))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA/"DES15k.hdf5")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.data, args.out)
