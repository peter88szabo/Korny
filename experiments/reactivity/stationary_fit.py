"""Fit bare-PBE0 residual forces at equilibria, harmonic spectra, and force changes.

NIST supplies unscaled harmonic wavenumbers and PBE0/cc-pVTZ Cartesian minima.
A reference minimum constrains equilibrium geometry by requiring zero forces
there. Its internal mass-weighted Hessian constrains the spectrum. The kernel
backbone and its existing dispersion switches are retained. Training never
reads the Behemoth water/ammonia references or the fixed reaction panel.
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

from experiments.reactivity.augmented import OUT, QM
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField
from src.valence import Valence

MASSES = {"H": (1, 1.007825), "C": (6, 12.), "N": (7, 14.003074), "O": (8, 15.994915)}
POINTS = OUT.parents[1]/"dataset_for_training/NIST/pbe0_cc_pvtz_stationary_points.json"


def internal_transform(masses, r):
    """Map an unweighted Cartesian Hessian to the vibrational mass-weighted basis."""
    root = np.sqrt(masses)[:, None]
    centered = r - np.average(r, axis=0, weights=masses)
    rigid = []
    for axis in np.eye(3):
        rigid.extend([(axis*root).ravel(), (np.cross(centered, axis)*root).ravel()])
    return null_space(np.array(rigid)) / np.repeat(root, 3)[:, None]


def stationary_rows(base, head, point):
    """Cache the gradient and internal Hessian for each coefficient at one minimum.

    A single set of finite differences is enough because the energy is linear
    in the new coefficients. The last column carries the frozen backbone.
    """
    z, mass = np.array([MASSES[s] for s in point["atoms"]]).T
    z = z.astype(int)
    r = np.array(point["positions_A"])
    ref = np.array(point["frequencies_cm1"])
    def gradients(x):
        return np.column_stack((head.design(z, x)[1], -base.predict(z, x)[1].ravel()))
    step = .001
    g = gradients(r)
    h = np.array([(gradients(r+step*u)-gradients(r-step*u))/(2*step)
                  for u in np.eye(r.size).reshape(-1, *r.shape)])
    h = (h + h.transpose(1, 0, 2))/2
    internal = internal_transform(mass, r)
    dyn = np.einsum('ia,ijw,jb->abw', internal, h, internal, optimize=True)
    if dyn.shape[0] != len(ref) or not np.all(ref > 0):
        raise ValueError("Expected a complete, stable spectrum")
    return g, dyn, ref


def mass_aligned(r, r0):
    """Rotate a distorted geometry and its force vectors back to the reference frame."""
    x = r-r.mean(axis=0)
    y = r0-r0.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vt)
    rotation = u @ vt
    return x @ rotation - y, rotation


def curvature_rows(base, head):
    """Read bare force secants only from training molecule IDs in QM7-X.

    Use nearby distortions and their parent geometry to measure a directional
    change in force. Dividing by displacement gives a secant stiffness in eV/Å².
    This protects curvature learned from forces away from the NIST minima.
    """
    split = json.loads((OUT/"results.json").read_text())
    heldout = set(split["heldout_QM7X_ids"])
    rng = np.random.default_rng(131)
    rows, provenance = [], []
    with h5py.File(QM) as h:
        for mol in rng.choice(sorted(set(h)-heldout), 32, replace=False):
            group = h[mol]
            eq = next(key for key in sorted(group) if key.endswith("-opt"))
            parent = eq.rsplit("-", 1)[0]
            point = group[eq]
            z, r = point["atNUM"][:], point["atXYZ"][:]
            d0 = head.design(z, r)[1]
            df0 = base.predict(z, r)[1]-point["pbe0FOR"][:]
            candidates = [key for key in group if key.startswith(parent+"-d")]
            nearby = []
            for name in rng.choice(sorted(candidates), min(32,len(candidates)), replace=False):
                target = group[name]
                x = target["atXYZ"][:]
                delta, rotation = mass_aligned(x, r)
                nearby.append((np.linalg.norm(delta),name,delta,rotation,x))
            for norm,name,delta,rotation,x in sorted(nearby, key=lambda t:t[0])[:2]:
                if norm < 1e-5:
                    continue
                # Transform the displacement into each frame to contract with
                # the Cartesian gradient of every coefficient in that frame.
                direction = delta / norm
                da = (direction @ rotation.T).ravel() @ head.design(z, x)[1]
                db = direction.ravel() @ d0
                df1 = base.predict(z, x)[1] - group[name]["pbe0FOR"][:]
                # Predicted force = base force minus the valence gradient.
                target = np.sum(direction*(df1 @ rotation - df0))
                rows.append(np.r_[(da-db)/norm, target/norm])
                provenance.append([str(mol),eq,str(name)])
    return np.array(rows), provenance


def residuals(w, gradients, hessians, curv, prior):
    """Evaluate joint residuals and their exact parameter Jacobian.

    The eigenvalue derivative is each mode's projection of the Hessian
    derivative. Squared wavenumbers retain the sign of unstable eigenvalues.
    Division by twice the reference frequency makes a small frequency error
    contribute nearly equally across the spectrum, including the bending band.
    """
    resid, jac = [], []
    for g in gradients:
        scale = .12*np.sqrt(len(g))
        resid.append((g[:, :-1]@w + g[:, -1])/scale)
        jac.append(g[:, :-1]/scale)
    for d, ref in hessians:
        h = d[:, :, :-1] @ w + d[:, :, -1]
        vals, v = np.linalg.eigh(h)
        scale = TO_CM1**2/(2*np.maximum(350.,ref)*120*np.sqrt(len(ref)))
        resid.append((vals - (ref/TO_CM1)**2)*scale)
        deriv = np.einsum('im,ijw,jm->mw',v,d[:, :, :-1],v,optimize=True)
        jac.append(deriv*scale[:, None])
    # One RMS block for changes in QM7-X stiffness and one for the ridge prior.
    scale = 2.*np.sqrt(len(curv))
    resid.extend(((curv[:, :-1]@w-curv[:, -1])/scale, w/prior))
    jac.extend((curv[:, :-1]/scale, np.diag(1/prior)))
    return np.concatenate(resid), np.vstack(jac)


def summarize(points, w):
    """Compute the force RMS and raw spectral errors at each reference geometry."""
    result = {}
    for name,(g,d,ref) in points.items():
        grad = g[:, :-1]@w + g[:, -1]
        vals = np.linalg.eigvalsh(d[:, :, :-1]@w+d[:, :, -1])
        freq = np.sign(vals)*np.sqrt(np.abs(vals))*TO_CM1
        result[name] = dict(force_rms_eV_A=float(np.sqrt(np.mean(grad**2))),
                            spectrum_mae_cm1=float(np.mean(np.abs(freq-ref))),
                            imaginary_modes=int(np.sum(freq < -1)))
    return result


def main():
    started = monotonic()
    backbone = OUT/json.loads((OUT/"results.json").read_text())["checkpoint"]
    base, head = ForceField.load(backbone), Valence()
    labels = json.loads(POINTS.read_text())
    data = {name:stationary_rows(base, head, p) for name,p in labels.items()}
    train = [data[name] for name,p in labels.items() if p["split"]=="train"]
    g, h = [item[0] for item in train], [item[1:] for item in train]
    curv, sources = curvature_rows(base, head)
    # Independent weak Gaussian priors on bond slopes, bond curvature, angular
    # slopes, bending stiffness, substitution and collective angular stiffness.
    prior = np.r_[np.tile([1.5,8.],len(head.pairs)),np.tile([2.,6.,2.,3.],len(head.centers))]
    zero = head.weights.copy()
    def fun(w):
        return residuals(w,g,h,curv,prior)[0]
    def jac(w):
        return residuals(w,g,h,curv,prior)[1]
    result = least_squares(fun, zero, jac=jac, ftol=1e-7, max_nfev=180)
    if not result.success:
        raise RuntimeError(result.message)
    head.weights = result.x
    base.valence = head
    stream = BytesIO()
    base.save(stream)
    content = stream.getvalue()
    name = "pbe0-stationary-" + hashlib.sha256(content).hexdigest()[:16] + ".npz"
    (OUT/name).write_bytes(content)
    scores = dict(checkpoint=name, backbone=backbone.name, fit_seconds=monotonic()-started,
                  training_molecules=[k for k,p in labels.items() if p["split"]=="train"],
                  development=[k for k,p in labels.items() if p["split"]=="development"],
                  bare_force_secants=sources, evaluations=result.nfev,
                  training_cost_before=float(.5*np.sum(fun(zero)**2)),training_cost_after=float(result.cost),
                  reference_geometry_scores_before=summarize(data,zero),
                  reference_geometry_scores_after=summarize(data,result.x))
    (OUT/"stationary_fit.json").write_text(json.dumps(scores,indent=2)+"\n")
    print(name, "time",round(scores["fit_seconds"],1),"seconds",flush=True)
    print("Independent ethanol, at reference geometry",scores["reference_geometry_scores_before"].get("ethanol"),
          scores["reference_geometry_scores_after"].get("ethanol"))


if __name__ == "__main__":
    main()
