"""Compare a saved force field with completed Behemoth PBE0 harmonic references.

Each model starts at the same reference geometry, relaxes without symmetry
constraints, and is differentiated at its own stationary point. Only the saved
Behemoth output is read. All forces used by the optimizer and Hessian are ML.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.constants import angstrom, atomic_mass, c, electron_volt
from scipy.linalg import null_space
from scipy.optimize import minimize

from experiments.reactivity.behemoth_frequencies import RUN
from src.forcefield import ForceField
from src.units import BOHR_A

HERE = Path(__file__).resolve().parent
ELEMENTS = {"H": (1, 1.008), "N": (7, 14.007), "O": (8, 15.999)}
# sqrt(eV / (angstrom^2 u)) is angular frequency. Divide by 2 pi c in cm/s.
TO_CM1 = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (2 * np.pi * c * 100)


def reference(case_dir):
    """Select a completed reference and read only its internal vibrational modes."""
    for run in sorted(case_dir.glob("attempt-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        status = run / "status.json"
        if status.exists() and json.loads(status.read_text()).get("completed_key"):
            break
    else:
        raise ValueError(f"No completed frequency calculation in {case_dir}")
    lines = (run / "optimized_geometry.xyz").read_text().splitlines()
    atoms = [line.split() for line in lines[2:] if line.strip()]
    xyz = np.array([row[1:4] for row in atoms], dtype=float)
    numbers, masses = np.array([ELEMENTS[row[0]] for row in atoms]).T
    data = (run / "vibrational_freq_frequency.dat").read_text().splitlines()
    values = [float(words[2]) for line in data if (words := line.split())
              and words[0].isdigit() and int(words[1]) >= 0]
    if len(atoms) != int(lines[0]) or len(values) != xyz.size - 6:
        raise ValueError(f"Incomplete nonlinear-molecule reference in {run}")
    return run, numbers.astype(int), masses, xyz, np.asarray(values)


def vibrations(model, numbers, xyz, mass, step):
    """Diagonalize the internal mass-weighted Hessian and retain imaginary modes.

    The orthogonal complement of the rigid mass-weighted displacements removes
    translations and rotations by geometry, so a negative bending eigenvalue
    cannot be accidentally removed as one of the six external modes.
    """
    hessian = []
    for direction in np.eye(xyz.size).reshape(-1, *xyz.shape):
        fplus = model.predict(numbers, xyz + step * direction)[1]
        fminus = model.predict(numbers, xyz - step * direction)[1]
        hessian.append(((fminus - fplus) / (2 * step)).ravel())
    hessian = np.array(hessian)
    hessian = (hessian + hessian.T) / 2
    centered = xyz - np.average(xyz, axis=0, weights=mass)
    rootmass = np.sqrt(mass)[:, None]
    external = []
    for axis in np.eye(3):
        external.extend([(rootmass * axis).ravel(), (rootmass * np.cross(centered, axis)).ravel()])
    internal = null_space(np.array(external))
    weights = np.repeat(rootmass, 3)
    dynamical = hessian / np.outer(weights, weights)
    values, vectors = np.linalg.eigh(internal.T @ dynamical @ internal)
    modes = (internal @ vectors / weights[:, None]).T.reshape(-1, *xyz.shape)
    return np.sign(values) * np.sqrt(np.abs(values)) * TO_CM1, modes


def differences(values, ref):
    """Compare in frequency order, counting both partners of a degenerate pair."""
    delta = values - ref
    return dict(error_cm1=delta.tolist(), mae_cm1=float(np.abs(delta).mean()),
                rmsd_cm1=float(np.sqrt(np.mean(delta**2))))


def geometry_errors(numbers, reference_xyz, xyz):
    """Compare the same bonds and angles on both geometries and remove rigid motion.

    Build the list of bonded pairs once from the reference. A bond broken by
    the model remains in the score. The fixed atom order follows the reference.
    """
    radii = {1:.31, 6:.76, 7:.71, 8:.66, 9:.57, 15:1.07, 16:1.05, 17:1.02}
    pairs = [(i,j) for i,j in combinations(range(len(numbers)),2)
             if np.linalg.norm(reference_xyz[i]-reference_xyz[j]) <
             1.22*(radii[numbers[i]]+radii[numbers[j]])]
    triples = []
    for i in range(len(numbers)):
        neighbors = [b if a == i else a for a,b in pairs if i in (a,b)]
        triples.extend((j,i,k) for j,k in combinations(neighbors,2))

    def internals(x):
        bonds = [np.linalg.norm(x[i]-x[j]) for i,j in pairs]
        angles = []
        for j,i,k in triples:
            a,b = x[j]-x[i], x[k]-x[i]
            angles.append(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
        return np.array(bonds),np.array(angles)

    ref_bonds,ref_angles = internals(reference_xyz)
    bonds,angles = internals(xyz)
    x,y = xyz-xyz.mean(axis=0),reference_xyz-reference_xyz.mean(axis=0)
    u,_,vt = np.linalg.svd(x.T@y)
    u[:,-1] *= np.linalg.det(u@vt)
    aligned = x@u@vt-y
    return dict(bond_lengths_A=bonds.tolist(), angles_deg=angles.tolist(),
                reference_bonds_A=ref_bonds.tolist(),reference_angles_deg=ref_angles.tolist(),
                bond_mae_A=float(np.mean(np.abs(bonds-ref_bonds))),
                angle_mae_deg=float(np.mean(np.abs(angles-ref_angles))),
                aligned_rmsd_A=float(np.sqrt(np.mean(np.sum(aligned**2,axis=1)))))


def measure(model, numbers, mass, start, ref):
    """Locate a nearby ML minimum and verify its curvature with two step sizes."""
    def energy_gradient(flat):
        energy, forces = model.predict(numbers, flat.reshape(start.shape))
        return energy, -forces.ravel()

    def relax(x):
        opt = minimize(energy_gradient, x.ravel(), jac=True, method="BFGS",
                       options={"gtol": 8e-7, "maxiter": 400})
        if np.abs(opt.jac).max() > 8e-6:
            raise RuntimeError(f"Relaxation failed: {opt.message}, |g|max={np.abs(opt.jac).max()}")
        return opt.x.reshape(start.shape)

    step = 0.005 * BOHR_A  # Match the reference central-difference displacement.
    xyz = relax(start)
    for attempt in range(5):
        freq, modes = vibrations(model, numbers, xyz, mass, step)
        if freq.min() >= -1:
            break
        # A symmetry-preserving start can lead to an unstable stationary point.
        # Follow both directions of its softest mode and keep the lower minimum.
        shift = 0.04 * modes[0] / np.linalg.norm(modes[0], axis=1).max()
        xyz = min((relax(xyz + shift), relax(xyz - shift)),
                  key=lambda x: model.energy(numbers, x))
    else:
        raise RuntimeError("Unable to resolve a negative internal mode")
    checked, _ = vibrations(model, numbers, xyz, mass, step / 2)
    return dict(predicted_cm1=freq.tolist(), **differences(freq, ref),
                max_step_change_cm1=float(np.abs(freq-checked).max()),
                max_force_eV_A=float(np.abs(model.predict(numbers, xyz)[1]).max()),
                **geometry_errors(numbers,start,xyz),
                xyz_A=xyz.tolist(), saddle_escapes=attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE/json.loads((HERE/"results.json").read_text())["checkpoint"])
    parser.add_argument("--references", type=Path, default=RUN)
    parser.add_argument("--volumes", type=Path, default=HERE.parent/"des15k"/json.loads(
        (HERE.parent/"des15k/volumes.json").read_text())["checkpoint"])
    parser.add_argument("--out", type=Path, default=HERE/"frequency_validation.json")
    args = parser.parse_args()
    result = {"checkpoint": args.model.name, "volumes": args.volumes.name,
              "reference_method": "Behemoth bare PBE0/cc-pVTZ harmonic", "cases": {}, "pooled": {}}
    for name in ("water", "ammonia"):
        folder, numbers, masses, xyz, ref = reference(args.references/name)
        case = {"reference_path": str(folder), "reference_cm1": ref.tolist(), "models": {}}
        for switch in ("off", "ts", "d3"):
            model = ForceField.load(args.model, dispersion=switch, volumes=args.volumes)
            sample = measure(model, numbers, masses, xyz, ref)
            case["models"][switch] = sample
            print(name, switch, np.round(sample["predicted_cm1"], 1),
                  "MAE", round(sample["mae_cm1"], 1), flush=True)
        result["cases"][name] = case
    for switch in ("off", "ts", "d3"):
        errors = np.concatenate([v["models"][switch]["error_cm1"] for v in result["cases"].values()])
        result["pooled"][switch] = dict(n_modes=len(errors), mae_cm1=float(np.abs(errors).mean()),
                                        rmsd_cm1=float(np.sqrt(np.mean(errors**2))))
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
