"""Molecular Tkatchenko–Scheffler pair dispersion with PBE0 defaults."""
import numpy as np

from ..jets import Jet, jet
from ..units import BOHR_A, HARTREE_EV

# Neutral-atom (polarizability, C6, vdW radius), in atomic units.
# libMBD src/mbd_vdw_param.f90, mbd_damping.F90 and mbd_defaults.f90.
TS_FREE_ATOMS = {1: (4.5, 6.5, 3.10), 6: (12.0, 46.6, 3.59),
                 7: (7.4, 24.2, 3.34), 8: (5.4, 15.6, 3.19),
                 9: (3.8, 9.52, 3.04), 15: (25.0, 185.0, 4.01),
                 16: (19.6, 134.0, 3.86), 17: (15.0, 94.6, 3.71)}


def _atomic_data(Z, free):
    table = TS_FREE_ATOMS if free is None else free
    try:
        params = np.array([table[z] for z in Z], dtype=float).reshape(len(Z), 3)
    except KeyError as exc:
        raise ValueError(f"Missing TS free-atom data for atomic number {exc.args[0]}") from exc
    if not np.isfinite(params).all() or np.any(params <= 0):
        raise ValueError("TS free-atom parameters must be finite and positive")
    return params


def _pair_energy(r, log_i, log_j, atom_i, atom_j, sR, damping):
    if not 0 < sR < np.inf or not 0 < damping < np.inf:
        raise ValueError("TS range and damping parameters must be finite and positive")
    ai, ci, ri = np.moveaxis(atom_i, -1, 0)
    aj, cj, rj = np.moveaxis(atom_j, -1, 0)
    # TS mixing gives C6_ij = volume_i * volume_j * C6_ij(free).
    mixed = 2*ci*cj / (ci*aj/ai + cj*ai/aj)
    radius = (log_i*(1./3.)).exp()*ri + (log_j*(1./3.)).exp()*rj
    exponent = -damping*(r*(radius*sR).power(-1.) - 1.)
    damp = (1. + exponent.exp()).power(-1.)
    return -(log_i+log_j).exp()*mixed*damp*r.power(-6.)


def ts_energy(R, Z, log_volumes, free=None, sR=0.96, damping=20.):
    """Energy in Hartree. Bohr coordinates and log volume ratios may be Jets."""
    R, log_volumes = jet(R), jet(log_volumes)
    if log_volumes.x.shape == (len(Z), 1):
        log_volumes = log_volumes[:, 0]
    if log_volumes.x.shape != (len(Z),) or R.x.shape != (len(Z), 3):
        raise ValueError("Expected one log volume and one coordinate vector per atom")
    if not np.isfinite(log_volumes.x).all() or not np.isfinite(R.x).all():
        raise ValueError("Non-finite TS coordinates or log volumes")
    params = _atomic_data(Z, free)
    i, j = np.triu_indices(len(Z), 1)
    delta = R[i] - R[j]
    norm2 = (delta*delta).linear(lambda a: a.sum(axis=1))
    if np.any(norm2.x == 0.):
        raise ValueError("Coincident nuclei")
    pairs = _pair_energy(norm2.power(.5), log_volumes[i], log_volumes[j],
                         params[i], params[j], sR, damping)
    return pairs.linear(np.sum)


def ts_energy_forces(Z, R, volume_model, *, free=None, sR=0.96, damping=20.):
    """Geometry-only energy (eV) and forces (eV/Å), including volume response.

    volume_model.evaluate(Z, R) returns log ratios and their Cartesian Jacobian
    in inverse Å. Each pair needs only three partials: distance and two log ratios.
    """
    R = np.asarray(R, dtype=float)
    log_volume, jac = volume_model.evaluate(Z, R)
    return ts_from_log_ratios(Z, R, log_volume, jac, free=free, sR=sR, damping=damping)


def ts_from_log_ratios(Z, R, log_volume, jac, *, free=None, sR=0.96, damping=20.):
    """Use predicted log ratios and their Cartesian derivatives (per Å)."""
    i, j = np.triu_indices(len(Z), 1)
    delta = (R[i]-R[j])/BOHR_A
    r = np.linalg.norm(delta, axis=1)
    if np.any(r == 0.):
        raise ValueError("Coincident nuclei")
    partials = np.eye(3)
    def variable(values, direction):
        return Jet(np.broadcast_to(values[:, None], (len(r), 3)), partials[direction])
    params = _atomic_data(Z, free)
    pair = _pair_energy(variable(r, 0), variable(log_volume[i], 1), variable(log_volume[j], 2),
                        params[i, None], params[j, None], sR, damping)
    radial_gradient = pair.u[:, :1] * delta / (r[:, None]*BOHR_A)
    grad = np.zeros_like(R)
    np.add.at(grad, i, radial_gradient)
    np.add.at(grad, j, -radial_gradient)
    volume_gradient = np.zeros(len(Z))
    np.add.at(volume_gradient, i, pair.u[:, 1])
    np.add.at(volume_gradient, j, pair.u[:, 2])
    grad += np.einsum("i,ijk->jk", volume_gradient, jac)
    return float(pair.x[:, 0].sum()*HARTREE_EV), -grad*HARTREE_EV
