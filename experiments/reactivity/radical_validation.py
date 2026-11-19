"""Test a fitted model on an archived isoprene peroxy radical H-shift path.

Read existing ORCA endpoints and a frequency-confirmed first-order saddle. All
coordinates and reference energies come from the same neutral doublet job. The
reported barriers use its refined TS, not the highest discrete NEB image.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.augmented import OUT
from experiments.reactivity.frequency_validation import vibrations
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.forcefield import ForceField
from src.units import BOHR_A, HARTREE_EV

DEFAULT = (Path("/home/peter/Dropbox/Research_Leuven/LIM1_reevol/Ready_calc")
           / "16H_Shift_Case2/NEB_Frontal_ready_Case2/neb_conformer_pair_10_27-11")


def frames(path):
    with path.open() as f:
        for line in f:
            n = int(line)
            comment = next(f)
            atoms = [next(f).split() for _ in range(n)]
            yield (np.array([ELEMENTS[a[0]] for a in atoms]),
                   np.array([a[1:4] for a in atoms], float), comment)


def validate(model, folder):
    inp, = folder.glob("*.inp")
    out = inp.with_suffix(".out").read_text()
    q, m = map(int, re.search(r"^\*\s*xyzfile\s+(-?\d+)\s+(\d+)", inp.read_text(),
                               re.I | re.M).groups())
    modes = re.findall(r"([-\d.]+) cm\*\*-1\s+\*\*\*imaginary mode", out)
    if (len(modes) != 1 or "ORCA TERMINATED NORMALLY" not in out
            or "THE OPTIMIZATION HAS CONVERGED" not in out):
        raise ValueError("Need a completed first-order saddle")
    path = list(frames(next(folder.glob("*_MEP_trj.xyz"))))
    z, ts, _ = next(frames(next(folder.glob("*_NEB-TS_converged.xyz"))))
    if any(not np.array_equal(atoms, z) for atoms, _, _ in path):
        raise ValueError("Atom order differs along the isoprene path")
    ref = [float(r[-1].split()[-1]) * HARTREE_EV for r in path]
    pred = [model.energy(z, xyz, q, m) for _, xyz, _ in path]
    ref_ts = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)[-1]) * HARTREE_EV
    model_ts, force = model.predict(z, ts, q, m)
    # Curvature at the *reference* saddle is diagnostic if the ML force is large.
    mass_source = "elemental isotope masses"
    masses = np.array([{1: 1.007825, 6:12., 8:15.994915}[a] for a in z])
    hess=inp.with_suffix(".hess")
    if hess.exists():
        atoms=hess.read_text().split("$atoms\n",1)[1].strip().splitlines()
        if int(atoms[0]) != len(z):
            raise ValueError("Hessian and transition-state atom counts disagree")
        rows=[l.split() for l in atoms[1:1+len(z)]]
        np.testing.assert_array_equal([ELEMENTS[a[0]] for a in rows],z)
        np.testing.assert_allclose(np.array([a[2:5] for a in rows],float)*BOHR_A,ts,atol=4e-5)
        masses=np.array([a[1] for a in rows],float)
        mass_source="ORCA Hessian atom masses"
    freq, _ = vibrations(model, z, ts, masses, .001, charge=q, multiplicity=m)
    spectrum=out.rsplit("VIBRATIONAL FREQUENCIES",1)[-1]
    printed=re.findall(r"^\s*\d+:\s+([-\d.]+) cm\*\*-1",spectrum,re.M)
    if len(printed)!=ts.size:
        raise ValueError("Incomplete ORCA vibrational spectrum")
    scale=re.search(r"Scaling factor for frequencies\s*=\s*([\d.]+)",spectrum)
    factor=float(scale[1]) if scale else 1.
    # Compare harmonic curvatures before the empirical spectroscopic scaling.
    reference_freq=np.sort(np.array(printed[6:],float)/factor)
    def differences(values, saddle):
        return {k:float(x*EV_TO_KCAL) for k,x in
                (("forward_barrier", saddle-values[0]), ("reverse_barrier", saddle-values[-1]),
                 ("reaction", values[-1]-values[0]))}
    reference = differences(ref, ref_ts)
    prediction = differences(pred, model_ts)
    return dict(folder=str(folder), state=[q, m], n_atoms=len(z), reference_method=re.search(r"^!.*$", inp.read_text(), re.M)[0],
                reference_kcal_mol=reference, predicted_kcal_mol=prediction,
                error_kcal_mol={k:prediction[k]-v for k,v in reference.items()},
                reference_imaginary_cm1=float(modes[0]), model_negative_modes_cm1=freq[freq < -1].tolist(),
                reference_printed_frequency_scale=factor, frequency_masses=mass_source,
                reference_unscaled_cm1=reference_freq.tolist(),model_cm1=freq.tolist(),
                frequency_MAE_at_reference_TS_cm1=float(np.abs(freq-reference_freq).mean()),
                model_rms_force_at_ref_TS_eV_A=float(np.sqrt(np.mean(force**2))),
                reference_path_relative_kcal_mol=((np.array(ref)-ref[0])*EV_TO_KCAL).tolist(),
                model_path_relative_kcal_mol=((np.array(pred)-pred[0])*EV_TO_KCAL).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--folder", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path, default=OUT/"radical_validation.json")
    a = p.parse_args()
    result = validate(ForceField.load(a.model, dispersion="d3"), a.folder)
    result.update(checkpoint=a.model.name, dispersion="d3")
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    print(result["reference_kcal_mol"], result["predicted_kcal_mol"])
