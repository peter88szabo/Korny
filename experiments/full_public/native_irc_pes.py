"""Export and independently evaluate the geometry-only RDB19-Rad energy readout.

This auxiliary surface predicts relative electronic energies in eV and
Cartesian forces in eV/Å for neutral doublets. Its labels include the source
B3LYP-D3(BJ) dispersion and its additive energy zero is arbitrary. Korny's
separate switchable-dispersion production checkpoints have their own readouts.
"""

import argparse
import csv
import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_path_directions import compact_banks
from experiments.full_public.trial_path_scales import metrics
from src.electronic import validate_electrons
from src.local import descriptor_array
from src.periodic_kernel import PeriodicKernel


class NativeRadicalPotential:
    def __init__(self, banks, elements, hydrogen_transfer=None):
        self.banks = banks
        self.elements = np.asarray(elements, int)
        self.hydrogen_transfer = hydrogen_transfer

    def save(self, path):
        fields = {
            "elements": self.elements,
            "banks": len(self.banks),
            "source": "RDB19-Rad electronic B3LYP-D3(BJ)",
        }
        for i, bank in enumerate(self.banks):
            fields.update({f"bank{i}_{k}": v for k, v in bank.fields().items()})
        if self.hydrogen_transfer is not None:
            fields.update(self.hydrogen_transfer.fields())
        np.savez_compressed(path, **fields)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            banks = []
            for i in range(int(f["banks"])):
                prefix = f"bank{i}_"
                banks.append(
                    PeriodicKernel.load(
                        {k[len(prefix) :]: f[k] for k in f if k.startswith(prefix)}
                    )
                )
            hydrogen_transfer = None
            if "htransfer_family" in f:
                from src.hydrogen_transfer import HydrogenTransferKernel
                hydrogen_transfer = HydrogenTransferKernel.load(f)
            return cls(banks, f["elements"], hydrogen_transfer)

    def _evaluate(self, z, r, charge, multiplicity, gradient):
        z, r = np.asarray(z, int), np.asarray(r, float)
        if charge != 0 or multiplicity != 2 or not np.isin(z, self.elements).all():
            raise ValueError(
                "The native IRC readout requires a neutral doublet with source elements"
            )
        if r.shape != (len(z), 3) or not np.isfinite(r).all():
            raise ValueError("Invalid Cartesian coordinates")
        validate_electrons(z, charge, multiplicity)
        first = self.banks[0]
        types = first.groups[z]
        e = 0.0
        forces = np.zeros_like(r) if gradient else None
        lo = 0
        for key, centers in first.centers.items():
            sl = slice(lo, lo + len(centers))
            lo = sl.stop
            proj = first.projections[key]
            for i in np.flatnonzero(types == key):
                desc, j = descriptor_array(
                    r,
                    types,
                    i,
                    nr=first.shells.shape[1],
                    lmax=2,
                    ron=first.cutoff[0],
                    rc=first.cutoff[1],
                    species=tuple(range(9)),
                    gaussian_shells=first.shells,
                    atom_radii=first.radii[z],
                    gradient=gradient,
                )
                u = (desc @ proj)[None, :]
                dv = np.zeros(u.shape[1]) if gradient else None
                for b in self.banks:
                    val, der = b.values(u, b.centers[key], gradient)
                    e += val[0] @ b.weights[sl]
                    if gradient:
                        dv += b.weights[sl] @ der[0]
                if gradient:
                    forces -= (dv @ proj.T @ j.reshape(len(desc), -1)).reshape(r.shape)
        if self.hydrogen_transfer is not None:
            # This optional contribution shares the archived electronic energy
            # convention and the same Cartesian scalar surface as the banks.
            h = self.hydrogen_transfer
            features, jac = h.design(z, r, gradient)
            e += features @ h.weights
            if gradient:
                forces += (jac @ h.weights).reshape(r.shape)
        return float(e), forces

    def energy(self, numbers, positions, charge=0, multiplicity=2):
        return self._evaluate(numbers, positions, charge, multiplicity, False)[0]

    def predict(self, numbers, positions, charge=0, multiplicity=2):
        return self._evaluate(numbers, positions, charge, multiplicity, True)


def force_case(rx):
    """Use full reference vectors on both held flanks and their common saddle."""
    out = np.zeros((2, 5))
    for side in ("forw", "rev"):
        with np.load(IRC / f"{rx}-{side}.npz") as f:
            z, r, ref = f["z"], f["r_A"].astype(float), f["force_eV_A"]
        for j, i in ((0, 0), (1, (len(r) - 1) // 2)):
            if j == 0 and side == "rev":
                continue
            _, force = native.predict(z, r[i])
            err = force - ref[i]
            out[j] += [
                err.size,
                abs(err).sum(),
                np.square(err).sum(),
                np.abs(ref[i]).sum(),
                1,
            ]
    return out


def run(stage):
    global native
    root = IRC.parent / f"multiscale-path{stage}"
    rows_root = IRC.parent / "multiscale-path61" if stage in (63, 64, 66, 67) else root
    records = json.loads((root / "snapshot.json").read_text())
    with np.load(root / "native.npz") as f:
        w = f["weights"][:, 0, None]
    stride = 3 if stage == 59 else 1
    pairs, _ = compact_banks(w, stride)
    banks = []
    for b, coef in pairs:
        b.weights = coef[:, 0]
        banks.append(b)
    # Every source formula contributes one representative to the element inventory.
    seen = set()
    elements = set()
    for r in records:
        if not r["heldout"] and r["formula"] not in seen:
            seen.add(r["formula"])
            with np.load(IRC / (r["rxn"] + "-forw.npz")) as f:
                elements.update(map(int, f["z"]))
    path = root / "native-radical-pes.npz"
    NativeRadicalPotential(banks, sorted(elements)).save(path)
    native = NativeRadicalPotential.load(path)
    held = [r["rxn"] for r in records if r["heldout"]]
    for rx in held[:3]:
        with np.load(IRC / f"{rx}-forw.npz") as f:
            z, r = f["z"], f["r_A"].astype(float)
            e, force = native.predict(z, r[0])
            np.testing.assert_allclose(e, native.energy(z, r[0]), atol=1e-8, rtol=0)
            v = np.random.default_rng(int(rx)).normal(size=r[0].shape)
            v /= np.linalg.norm(v)
            fd = (
                native.energy(z, r[0] - 2e-5 * v) - native.energy(z, r[0] + 2e-5 * v)
            ) / 4e-5
            np.testing.assert_allclose(fd, (force * v).sum(), atol=2e-5, rtol=0)
        with np.load(rows_root / f"{rx}.npz") as f:
            row = f["rows"][0].astype(float)
        with np.load(IRC / f"{rx}-rev.npz") as f:
            other = f["r_A"][-1].astype(float)
        gap = native.energy(z, other) - native.energy(z, r[-1])
        np.testing.assert_allclose(gap, row[:-1] @ w[:, 0], atol=5e-5, rtol=0)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        totals = sum(pool.map(force_case, held, chunksize=4, buffersize=8))
    scores = {}
    for name, (n, absolute, square, zero, frames) in zip(
        ("saddle", "flanks"), totals, strict=True
    ):
        scores[name] = {
            "frames": int(frames),
            "components": int(n),
            "mae_eV_A": absolute / n,
            "rmse_eV_A": np.sqrt(square / n),
            "zero_force_mae_eV_A": zero / n,
        }
    kt.write_json(root / "native-forces.json", scores)
    print("Native full Cartesian force validation", stage, scores, flush=True)


def cross_archive(stage):
    """Apply the frozen native PES directly to supported common Grambow triples."""
    root = IRC.parent / f"multiscale-path{stage}"
    native = NativeRadicalPotential.load(root / "native-radical-pes.npz")
    geometry, _, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    ok = np.array(
        [
            tuple(geometry[n].state) == (0, 2)
            and np.isin(geometry[n].z, native.elements).all()
            for n in names
        ]
    )
    e = np.zeros(len(names))
    _, indices = load_basis()
    if stage == 59:
        indices = indices[::3]
    for b in native.banks:
        x = np.load(kt.WORK / f"periodic-{b.length:g}-main.npy", mmap_mode="r")
        for lo in range(0, len(e), 256):
            sl = slice(lo, min(lo + 256, len(e)))
            e[sl] += x[sl][:, indices] @ b.weights
    for i in np.flatnonzero(ok)[:3]:
        g = geometry[names[i]]
        np.testing.assert_allclose(
            e[i], native.energy(g.z, g.r, *g.state), atol=5e-6, rtol=0
        )
    s, y, src, fold, kind, case = kt.equation_data("wb97xd3")
    available = (abs(s) @ (~ok).astype(float)) == 0
    errs = s @ e - y
    with (kt.OUT / "so3lr-tauri-matched/p49-public-comparison.csv").open() as f:
        scored = {}
        for row in csv.DictReader(f):
            values = [
                row[n + " error_kcal"]
                for n in ("Korny", "Original SO3LR", "Tauri-M-2606", "Tauri-L-2606")
            ]
            if (
                row["source"] == "Grambow"
                and all(values)
                and np.isfinite(np.asarray(values, float)).all()
            ):
                scored.setdefault(row["id"], set()).add(row["observable"])
    common = {
        name
        for name, observables in scored.items()
        if observables == {"reaction", "forward", "reverse"}
    }
    if len(common) != kt.COMMON_REACTIONS["Grambow"]:
        raise ValueError("The common Grambow panel changed")
    joined = {}
    for i in np.flatnonzero((fold < 0) & (src == "Grambow") & available):
        if case[i] in common:
            joined.setdefault(str(case[i]), {})[str(kind[i])] = errs[i]
    paired = {
        k: v for k, v in joined.items() if set(v) == {"forward", "reverse", "reaction"}
    }
    gap, forward, reverse = np.array(
        [[r[k] for k in ("reaction", "forward", "reverse")] for r in paired.values()]
    ).T
    result = metrics(gap, (forward + reverse) / 2)
    # Keep the exact rounded reference barriers even if the independent
    # reference reaction energy differs slightly from their difference.
    result.update(
        forward=float(abs(forward).mean() * kt.KCAL),
        reverse=float(abs(reverse).mean() * kt.KCAL),
        barriers=float((abs(forward).mean() + abs(reverse).mean()) * kt.KCAL / 2),
    )
    kt.write_json(
        root / "direct-grambow.json",
        {
            "eligible": len(paired),
            "total": len(common),
            "scores": result,
            "ids": sorted(paired),
        },
    )
    print(
        "Direct frozen RDB19 readout on Grambow", stage, len(paired), result, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=int, choices=(59, 61, 63, 64, 66, 67))
    parser.add_argument("--cross-archive", action="store_true")
    args = parser.parse_args()
    if args.cross_archive:
        cross_archive(args.stage)
    else:
        run(args.stage)
