"""Transfer measured radical path shapes through differentiable energy features.

In RDB19-Rad each transition state has two Gaussian IRC branches. Fit the
local polynomial kernel to path-relative energies and forces projected along
each short Cartesian step. One shared readout and reaction-family readouts
encode different radical chemistry. Inputs to the deployed surface are only
elements, coordinates, molecular charge and multiplicity. Family labels
supervise the features during training and are unnecessary at inference.
"""

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import pickle
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import (
    kernel_trials as kt,
)
from experiments.full_public import (
    trial_fit,
)
from experiments.full_public import (
    trial_multiscale as ms,
)
from experiments.full_public import (
    trial_periodic as kernel,
)
from experiments.full_public.aimnet_parent import formula_keys, validation_formulas
from experiments.full_public.rdb19_irc import OUT as IRC
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

WORK = kt.WORK
ROOT = IRC.parent / "path-kernel44"
STAGE = "p44-public"
PARENT = "p43-public"
TARGET = "wb97xd3"


def read_manifest():
    """Freeze the available complete pairs before any model predictions or validation."""
    dest = ROOT / "manifest.json"
    if dest.exists():
        return json.loads(dest.read_text())
    records = verified_pairs(sorted(IRC.glob("*-forw.npz")))
    kt.write_json(dest, records)
    return records


def verified_pairs(forward_files):
    """Verify paired paths and assign the established formula split, independently of cache size."""
    labels = json.loads((IRC / "reaction-identities.json").read_text())
    protected = validation_formulas()
    records = []
    for f in forward_files:
        rx = f.stem.split("-")[0]
        other = IRC / f"{rx}-rev.npz"
        if not other.exists():
            continue
        with np.load(f) as a, np.load(other) as b:
            if not np.array_equal(a["z"], b["z"]) or not np.allclose(
                a["r_A"][0], b["r_A"][0], atol=0.0001, rtol=0
            ):
                continue
            if (
                abs(a["energy_eV"][0] - b["energy_eV"][0]) > 0.0001
                or int(a["multiplicity"]) != 2
                or int(b["multiplicity"]) != 2
            ):
                continue
            # "forw"/"rev" names can differ between Gaussian jobs. The
            # electronic Hamiltonian and total charge must agree across the
            # entire source before pooling these electronic energies.
            for branch in (a, b):
                route = "".join(str(branch["route"]).lower().split())
                required = (
                    "bv5lyp/def2tzvp",
                    "iop(3/76=1000002000)",
                    "iop(3/77=0720008000)",
                    "iop(3/78=0810010000)",
                    "empiricaldispersion=gd3bj",
                )
                if (
                    int(branch["charge"]) != 0
                    or str(branch["scf"]) != "UB+HF-V5LYP"
                    or not all(x in route for x in required)
                ):
                    raise ValueError(f"Different electronic reference in IRC {rx}")
            z = a["z"]
            frames = len(a["point"]) + len(b["point"]) - 1
        label = labels[rx]
        fk = int(formula_keys(z))
        held = (
            fk in protected
            or int.from_bytes(hashlib.sha256(str(fk).encode()).digest()[:4], "big") % 6
            == 0
        )
        records.append(
            {
                "rxn": rx,
                "family": label["reaction_type"],
                "formula": fk,
                "heldout": bool(held),
                "frames": frames,
            }
        )
    if not any(r["heldout"] for r in records) or not any(
        not r["heldout"] for r in records
    ):
        raise ValueError("Both formula partitions are required")
    return records


def init():
    global basis, columns
    kernel.setup()
    basis, cols = load_basis()
    # Retain a regularly spaced subset of the p42 features in every chemical
    # group. This gives enough independent parameters for the available IRC families.
    columns = cols[::3]
    kernel.basis = basis
    kernel.OPTIONS = ((0.85, 3),)


def cache(record):
    rx = record["rxn"]
    dest = ROOT / (rx + ".npz")
    if dest.exists():
        return
    coords = []
    forces = []
    energies = []
    for direction in ("forw", "rev"):
        with np.load(IRC / f"{rx}-{direction}.npz") as a:
            z = a["z"]
            coords.append(a["r_A"].astype(float))
            forces.append(a["force_eV_A"].astype(float))
            energies.append(a["energy_eV"])
    # Zigzag the two source branches through their common TS once. This is a
    # geometric path coordinate only, unrelated to the tabulated forward sign.
    coords = np.concatenate((coords[0][::-1], coords[1][1:]))
    forces = np.concatenate((forces[0][::-1], forces[1][1:]))
    energy = np.r_[energies[0][::-1], energies[1][1:]]
    ts = len(energies[0]) - 1
    x = np.vstack(
        [
            kernel.batch([(z, r) for r in coords[i : i + 24]])[0][:, columns]
            for i in range(0, len(coords), 24)
        ]
    )
    dr = np.diff(coords, axis=0)
    norm = np.linalg.norm(dr.reshape(len(dr), -1), axis=1)
    work = -np.einsum("iab,iab->i", 0.5 * (forces[:-1] + forces[1:]), dr)
    # Drop zero displacements from the derivative target. Archive SCF precision
    # precludes stable secants at an infinitesimal displacement.
    keep = norm > 0.0001
    dx = np.diff(x, axis=0)[keep] / norm[keep, None]
    slope = work[keep] / norm[keep]
    eblock = np.column_stack((x - x[ts], energy - energy[ts]))
    fblock = np.column_stack((dx, slope))
    # Each reaction has the same weight despite differing integration step counts.
    gram = eblock.T @ eblock / len(eblock) + 0.20 * (fblock.T @ fblock) / len(fblock)
    cols = np.column_stack((x[[0, ts, -1]], energy[[0, ts, -1]]))
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f, gram=gram, ends=cols)
    dest.with_suffix(".pending").replace(dest)


def teachers(records):
    """Each auxiliary surface has its own energy origin and uses only training formulas."""
    counts = Counter(r["family"] for r in records if not r["heldout"])
    groups = ["all", *sorted(k for k, v in counts.items() if v >= 4)]
    accum = {k: np.zeros((len(columns) + 1,) * 2) for k in groups}
    for row in records:
        if row["heldout"]:
            continue
        with np.load(ROOT / (row["rxn"] + ".npz")) as a:
            accum["all"] += a["gram"]
            if row["family"] in accum:
                accum[row["family"]] += a["gram"]
    coeff = []
    for k, g in accum.items():
        lhs = g[:-1, :-1].copy()
        # Common fit explains the broad chemistry. Class-specific coefficients
        # are departures from it, ridge-regularized at the same physical scale.
        prior = np.zeros(len(lhs)) if k == "all" else coeff[0]
        ridge = 0.015 * (sum(counts.values()) if k == "all" else counts[k])
        lhs.flat[:: len(lhs) + 1] += ridge
        w = cho_solve(cho_factor(lhs), g[:-1, -1] + ridge * prior)
        coeff.append(w if k == "all" else w - prior)
    learned = np.column_stack(coeff)
    # Only update diagnostics here. Held-out paths cannot change the feature fit.
    diagnostics = []
    for row in records:
        if not row["heldout"]:
            continue
        with np.load(ROOT / (row["rxn"] + ".npz")) as a:
            end = a["ends"]
            fun = learned[:, 0].copy()
            if row["family"] in groups:
                fun += learned[:, groups.index(row["family"])]
            err = (end[:, :-1] @ fun - end[:, -1]) * kt.KCAL
            diagnostics.append(
                [
                    row["rxn"],
                    row["family"],
                    float(err[1] - err[0]),
                    float(err[1] - err[2]),
                    float(err[2] - err[0]),
                ]
            )
    kt.write_json(ROOT / "auxiliary-path-errors.json", diagnostics)
    return groups, learned


def compress(matrix, mask, learned, selected=None):
    """Read only doublet entries, apply their learned projection in bounded blocks."""
    if selected is None:
        selected = columns
    out = np.zeros((len(mask), learned.shape[1]))
    active = np.flatnonzero(mask)
    for lo in range(0, len(active), 256):
        sel = active[lo : lo + 256]
        out[sel] = matrix[np.ix_(sel, selected)] @ learned
    return out


def prepare_records(records):
    """Cache each trajectory independently, before any cross-reaction fit."""
    # A streaming preparation and a full-archive fit can meet when the
    # download finishes. Reserve the cache writer here and recheck its files.
    with (ROOT / "encode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        init()
        pending = [r for r in records if not (ROOT / (r["rxn"] + ".npz")).exists()]
        with ProcessPoolExecutor(
            max_workers=3, mp_context=mp.get_context("fork")
        ) as pool:
            for i, _ in enumerate(pool.map(cache, pending, buffersize=8), 1):
                if i % 20 == 0:
                    print(
                        "Encoded radical IRC reactions",
                        i,
                        "of",
                        len(pending),
                        flush=True,
                    )


def trial():
    ROOT.mkdir(exist_ok=True)
    records = read_manifest()
    prepare_records(records)
    groups, learned = teachers(records)
    b, _ = load_basis()
    transfer(groups, [(b, columns, learned)], records, STAGE, PARENT, "p41-public")


def transfer(groups, banks, records, stage, parent_stage, high_stage):
    """Calibrate native IRC readouts on the target Hamiltonian and verify deployment.

    Each bank contributes local features at its own length and degree. The
    shared/family readouts from the native fit are frozen. The target fit
    changes only their mixing coefficients for doublets on the radical surface.
    """
    s, y, src, fold, kind, case = d = kt.equation_data(TARGET)
    geometry, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    state = np.array([geometry[n].state[1] for n in names])

    def project(name, mask):
        result = np.zeros((len(mask), len(groups)))
        for basis, selected, native in banks:
            matrix = np.load(
                WORK / f"periodic-{basis.length:g}-{name}.npy", mmap_mode="r"
            )
            result += compress(matrix, mask, native, selected)
        return result

    x = project("main", state == 2)
    base = ms.baseline(TARGET, -1, "parent", parent_stage)
    A = s @ x
    g = kt.grouped_moments(d, A, y - s @ base, fold >= 0)
    eq = g["other"] + 96 * g["mid"] + 36 * g["gap"]
    with (WORK / "radical-modes.pkl").open("rb") as f:
        rads = pickle.load(f)
    mask = np.array([r[1][3][1] == 2 for r in rads])
    saddle = np.array([r[1][0].split("/")[-1].startswith("ts") for r in rads])
    rad = project("rad", np.repeat(mask, 7)).reshape(-1, 7, len(groups))
    meta = np.load(WORK / "rad-meta.npy")
    bm = ms.baseline(TARGET, -1, "modes", parent_stage)
    a, ref = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
    eq += trial_fit.reduce_gram(a, ref, np.repeat(meta[:, 0] >= 0, 3))
    # All 91,156 unique HDF5 backbone training states and NIST minima are
    # singlets. Their forces and spectra are independent of this doublet term.
    lhs = eq[:-1, :-1].copy()
    lhs.flat[:: len(lhs) + 1] += 3.0
    coeff = cho_solve(cho_factor(lhs), eq[:-1, -1])
    pred = base + x @ coeff
    key = "irc" + stage.split("-")[0][1:]
    np.save(WORK / f"{key}-{TARGET}-parent--1.npy", pred)
    modes = bm + rad @ coeff
    np.save(WORK / f"{key}-{TARGET}-modes--1.npy", modes)
    np.save(WORK / f"{stage}-held-modes.npy", modes[meta[:, 0] == -1])
    parent = ForceField.load(WORK / f"{parent_stage}-{TARGET}.npz", dispersion="d3")
    old, extra = parent.spin_readouts.get(2, (parent.weights, {}))
    extra = extra.copy()
    for b, selected, native in banks:
        b.weights = np.zeros(b.size)
        index = len(parent.additional_bases)
        parent.additional_bases.append(b)
        w = b.weights.copy()
        w[selected] = native @ coeff
        extra[index] = w
    parent.spin_readouts[2] = old, extra
    path = WORK / f"{stage}-{TARGET}.npz"
    parent.save_with_base(kt.original.BASE, path)
    deployed = ForceField.load(path, dispersion="d3")
    for m in (1, 2):
        i = next(
            i
            for i, n in enumerate(names)
            if state[i] == m and 4 <= len(geometry[n].z) <= 12
        )
        g0 = geometry[names[i]]
        z, r = g0.z, g0.r
        e, f = deployed.predict(z, r, *g0.state)
        np.testing.assert_allclose(e, pred[i], atol=3e-6, rtol=0)
        v = np.random.default_rng(i).normal(size=r.shape)
        v /= np.linalg.norm(v)
        fd = (
            deployed.energy(z, r - 2e-5 * v, *g0.state)
            - deployed.energy(z, r + 2e-5 * v, *g0.state)
        ) / 4e-5
        np.testing.assert_allclose(fd, (f * v).sum(), atol=5e-5, rtol=0)
    h = fold == -1
    np.savez_compressed(
        WORK / f"{stage}-{TARGET}-equations.npz",
        source=src[h],
        case=case[h],
        kind=kind[h],
        error_kcal=(s @ pred - y)[h] * kt.KCAL,
    )
    # The high-level PES and its singlet Hessians come from the indicated checkpoint.
    for suffix in ("high.npz", "high-equations.npz", "spectra.json"):
        shutil.copyfile(WORK / f"{high_stage}-{suffix}", WORK / f"{stage}-{suffix}")
    high_branch = json.loads((WORK / f"{high_stage}-settings.json").read_text())[
        "branches"
    ]["high"]
    cfg = {
        "selected": stage,
        "parent": parent_stage,
        "high_parent": high_stage,
        "branches": {"high": high_branch, TARGET: key},
        "training_records": sum(not r["heldout"] for r in records),
        "validation_records": sum(r["heldout"] for r in records),
        "training_frames": sum(r["frames"] for r in records if not r["heldout"]),
        "validation_frames": sum(r["frames"] for r in records if r["heldout"]),
        "families": groups,
        "readout": coeff.tolist(),
    }
    kt.write_json(WORK / f"{stage}-settings.json", cfg)
    compare(stage)
    validate(stage)
    print("Completed radical path transfer", cfg, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Fit the checksum-verified full RDB19-Rad release",
    )
    p.add_argument(
        "--prepare-available",
        action="store_true",
        help="Encode complete downloaded trajectory pairs for the full fit, without training or freezing its split",
    )
    args = p.parse_args()
    if args.full or args.prepare_available:
        if args.full and not (IRC.parent / "Reactions.tar.bz2").is_file():
            p.error("The completed RDB19-Rad archive is required")
        ROOT = IRC.parent / "path-kernel45"
        STAGE = "p45-public"
    with threadpool_limits(1):
        if args.prepare_available:
            ROOT.mkdir(exist_ok=True)
            complete = [
                {"rxn": f.name.split("-")[0]}
                for f in sorted(IRC.glob("*-forw.npz"))
                if f.with_name(f.name.replace("-forw.", "-rev.")).is_file()
                and not (ROOT / (f.name.split("-")[0] + ".npz")).is_file()
            ]
            prepare_records(complete)
            print("Encoded available complete IRC pairs", len(complete), flush=True)
        else:
            trial()
