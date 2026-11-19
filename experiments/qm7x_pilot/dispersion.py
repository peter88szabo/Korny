"""Audit PBE0 TS, MBD@rsSCS and D3(BJ)+ATM corrections on the pilot test split."""
import argparse
import csv
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from pymbd.fortran import MBDGeom

from src.jets import Jet
from src.local import Molecules
from src.dispersion.d3 import d3_energy_gradient_bohr
from src.dispersion.mbd import mbd_energy_bohr
from src.dispersion.ts import ts_energy, ts_energy_forces
from src.dispersion.volumes import HirshfeldVolumes
from src.units import BOHR_A, HARTREE_EV
from .run import DATA, OUT, SPEC, PairBaseline, predict

SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}


def audit(data, out, probe):
    with (out/"predictions.csv").open() as file:
        rows = [row for row in csv.DictReader(file) if row["model"] == "gaussian"]
    paths = [r["molecule"]+"/"+r["geometry"] for r in rows if r["split"] == "test"]
    train = [r["molecule"]+"/"+r["geometry"] for r in rows if r["split"] == "train"]
    if {p.split("/")[0] for p in paths} & {p.split("/")[0] for p in train}:
        raise ValueError("Train/test molecule overlap")
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Expected one Gaussian prediction per held-out geometry")
    local = Molecules.load_local(out/"pair_gaussian.npz")
    pair = PairBaseline.load(out/"pair.npz")
    offsets = np.array(json.loads((out/"results.json").read_text())["config"]["composition_offsets_eV"])
    samples, lines = [], []
    with h5py.File(data, "r") as h:
        def volume_records():
            for path in train:
                g = h[path]
                yield g["atNUM"][()], g["atXYZ"][()], g["hRAT"][()]
        volumes = HirshfeldVolumes.fit(volume_records(), SPEC)
        volumes.save(out/"volumes.npz")
        volumes = HirshfeldVolumes.load(out/"volumes.npz")
        for path in paths:
            g = h[path]
            z = g["atNUM"][()]
            xyz = g["atXYZ"][()]/BOHR_A
            v = g["hRAT"][()].ravel()
            samples.append((z, xyz, v, g["eTS"][()].item(), g["eMBD"][()].item(),
                            g["vdwFOR"][()], -g["eAT"][()].item(), g["totFOR"][()]))
            atoms = ((SYMBOL[n], n, *coord, ratio) for n, coord, ratio in zip(z, xyz, v))
            lines.append(" ".join(str(x) for atom in atoms for x in atom))
    result = subprocess.run([str(probe.resolve())], input="\n".join(lines)+"\n",
                            text=True, capture_output=True, check=True)
    errors = {}
    def record(name, delta):
        errors.setdefault(name, []).append(np.asarray(delta).ravel())

    rng = np.random.default_rng(71)
    for sample, line in zip(samples, result.stdout.splitlines(), strict=True):
        z, xyz, v, ets, embd, fmbd, atomization, total_force = sample
        native = np.fromstring(line, sep=" ")
        ets_native, ed3_native = native[:2]
        grad_ts, grad_d3 = native[2:].reshape(2, len(z), 3)
        with MBDGeom(xyz) as geom:
            ets_lib, grad_lib = geom.ts_energy_species([SYMBOL[n] for n in z], v, .96, force=True)
        # hRAT supplies oracle values. These TS gradients hold hRAT fixed.
        direction = rng.normal(size=xyz.shape)
        direction /= np.linalg.norm(direction)
        korny = ts_energy(Jet(xyz, direction), z, np.log(v))
        mbd = mbd_energy_bohr(z, xyz, np.log(v))
        ed3, gd3 = d3_energy_gradient_bohr(z, xyz, "pbe0", atm=True)
        r = xyz*BOHR_A
        log_pred, _ = volumes.evaluate(z, r)
        epred, fpred = ts_energy_forces(z, r, volumes)
        record("predicted_minus_reference_volume_ratio", np.exp(log_pred)-v)
        record("geometry_only_TS_minus_QM7X_eTS_eV", epred-ets)
        step = 1e-5
        def displaced_energy(delta):
            rr = r + delta*direction
            lv, _ = volumes.evaluate(z, rr)
            return ts_energy(rr/BOHR_A, z, lv).x*HARTREE_EV
        fd = (displaced_energy(step)-displaced_energy(-step))/(2*step)
        record("geometry_only_TS_directional_force_FD_eV_A", fd+np.sum(fpred*direction))
        record("libmbd_TS_minus_QM7X_eTS_eV", ets_lib*HARTREE_EV-ets)
        record("libmbd_rsSCS_minus_QM7X_eMBD_eV", mbd*HARTREE_EV-embd)
        record("behemoth_TS_minus_libmbd_energy_eV", (ets_native-ets_lib)*HARTREE_EV)
        record("behemoth_TS_minus_libmbd_fixed_volume_gradient_Ha_bohr", grad_ts-grad_lib)
        record("korny_TS_minus_libmbd_energy_eV", (korny.x-ets_lib)*HARTREE_EV)
        record("korny_TS_minus_libmbd_directional_gradient_Ha_bohr",
               korny.u-np.sum(direction*grad_lib))
        record("behemoth_D3_minus_dftd3_energy_eV", (ed3_native-ed3)*HARTREE_EV)
        record("behemoth_D3_minus_dftd3_gradient_Ha_bohr", grad_d3-gd3)
        record("D3_minus_QM7X_MBD_energy_eV", ed3_native*HARTREE_EV-embd)
        force_d3 = -grad_d3*(HARTREE_EV/BOHR_A)
        record("D3_minus_QM7X_MBD_force_eV_A", force_d3-fmbd)
        # Geometry-only prediction against the atom-referenced PBE0+MBD surface.
        e0, f0 = predict(z, xyz*BOHR_A, offsets, local, pair)
        record("pair_gaussian_plus_D3_minus_QM7X_total_energy_eV",
               e0+ed3_native*HARTREE_EV-(atomization+embd))
        record("pair_gaussian_plus_D3_minus_QM7X_total_force_eV_A", f0+force_d3-total_force)
    report = {"pymbd": version("pymbd"), "dftd3": version("dftd3"),
              "geometries": len(paths), "molecule_ids": len({p.split("/")[0] for p in paths}),
              "TS": {"sR": .96, "d": 20., "volumes": "reference hRAT"},
              "MBD_rsSCS": {"beta": .85, "volumes": "reference hRAT"},
              "D3": {"functional": "pbe0", "damping": "BJ", "ATM": True}}
    for key, chunks in errors.items():
        residual = np.concatenate(chunks)
        if not np.isfinite(residual).all():
            raise ValueError(f"Non-finite values in {key}")
        report[key] = {"mean_abs": float(abs(residual).mean()),
                       "max_abs": float(abs(residual).max())}
    if report["geometry_only_TS_directional_force_FD_eV_A"]["max_abs"] > 1e-6:
        raise ValueError("Geometry-only TS forces failed the energy finite-difference check")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("behemoth_probe", type=Path)
    parser.add_argument("--data", default=DATA, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    args = parser.parse_args()
    stats = audit(args.data, args.out, args.behemoth_probe)
    (args.out/"dispersion.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats, indent=2))
