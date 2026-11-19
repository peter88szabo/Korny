"""Score a DES-trained Korny model on the released SO3LR QM7-X test geometries."""
import argparse
import io
import json
from pathlib import Path
import shlex
import tarfile
from time import perf_counter

import numpy as np

import h5py

from experiments.des15k.run import ATOMIC, DATA, OUT, metrics
from src.dispersion.d3 import d3_energy_forces
from src.dispersion.ts import ts_from_log_ratios
from src.dispersion.volumes import HirshfeldVolumes
from src.forcefield import ForceField

MEMBER = "so3lr_test/qm7x_10k_testset.extxyz"
ELEMENTS = dict(H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17)


def frames(stream):
    for line in stream:
        natoms = int(line)
        fields = dict(item.split("=", 1) for item in shlex.split(next(stream)))
        if fields["Properties"] != "species:S:1:pos:R:3:forces:R:3:hirsh_ratios:R:1":
            raise ValueError("Unexpected columns in the official QM7-X file")
        data = [next(stream).split() for _ in range(natoms)]
        z = np.array([ELEMENTS[row[0]] for row in data])
        values = np.array([row[1:8] for row in data], dtype=float)
        yield z, values[:, :3], values[:, 3:6], float(fields["energy"]), int(fields.get("charge", 0)), values[:, 6]


def des_frames(path, heldout):
    """Read the dispersion-inclusive validation labels; training never calls this reader."""
    with h5py.File(path) as f:
        for group in f.values():
            for name, g in group.items():
                if name.rsplit("-", 1)[0] in heldout:
                    z = g["atNUM"][:]
                    e = float(g["ePBE0+MBD"][0]) - sum(ATOMIC[a] for a in z)
                    yield z, g["atXYZ"][:], g["totFOR"][:], e, round(g["hCHG"][:].sum()), g["hRAT"][:]


def assess(records, checkpoint, volume_file):
    """Evaluate each geometry once, then add either correction to exactly the same base E,F."""
    model = ForceField.load(checkpoint)
    volumes = HirshfeldVolumes.load(volume_file)
    names = ("off", "ts", "d3")
    errors = {k: [] for k in names}
    abs_f, sq_f = dict.fromkeys(names, 0.), dict.fromkeys(names, 0.)
    natoms, ratio_errors = [], []
    started = perf_counter()
    for z, r, fref, eref, charge, href in records:
        e, f = model.predict(z, r, charge)
        logs, jac = volumes.evaluate(z, r)
        ratio_errors.extend(np.exp(logs)-href)
        corrections = ((0., 0.), ts_from_log_ratios(z, r, logs, jac), d3_energy_forces(z, r))
        for name, (de, df) in zip(names, corrections, strict=True):
            errors[name].append(e + de - eref)
            diff = f + df - fref
            abs_f[name] += np.abs(diff).sum()
            sq_f[name] += np.square(diff).sum()
        natoms.append(len(z))
    stats = {k: metrics(errors[k], abs_f[k], sq_f[k], 3*sum(natoms), natoms) for k in names}
    ratios = np.array(ratio_errors)
    if not np.isfinite(ratios).all() or not all(np.isfinite(list(v.values())).all() for v in stats.values()):
        raise ValueError("Non-finite dispersion validation result")
    return dict(checkpoint=checkpoint.name, volumes=volume_file.name, reference="PBE0+MBD",
                elapsed_seconds=perf_counter()-started, volume_ratio_mae=float(np.abs(ratios).mean()),
                volume_ratio_rmse=float(np.sqrt(np.mean(ratios**2))), models=stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["des", "qm7x"])
    parser.add_argument("--checkpoint", type=Path, help="Bare PBE0 checkpoint to evaluate")
    parser.add_argument("--out", type=Path, help="JSON destination")
    args = parser.parse_args()
    model = args.checkpoint or OUT/json.loads((OUT/"results.json").read_text())["checkpoint"]
    volumes = OUT/json.loads((OUT/"volumes.json").read_text())["checkpoint"]
    if args.dataset == "des":
        heldout = set((OUT/"test_parents.txt").read_text().splitlines())
        report = assess(des_frames(DATA/"DES15k.hdf5", heldout), model, volumes)
        expected = json.loads((OUT/"results.json").read_text())["test"]["structures"]
    else:
        with tarfile.open(DATA/"so3lr_test.tar.gz", "r:gz") as tar, io.TextIOWrapper(tar.extractfile(MEMBER)) as stream:
            report = assess(frames(stream), model, volumes)
        expected = 10000
    if report["models"]["off"]["geometries"] != expected:
        raise ValueError("Incorrect validation structure count")
    (args.out or OUT/(args.dataset+"_validation.json")).write_text(json.dumps(report, indent=2)+"\n")
