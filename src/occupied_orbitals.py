"""A smooth orbital-population representation for spin-resolved reaction energies.

A small geometry-dependent matrix spans one s function on H/He and s,p on
heavier atoms. Occupation counts are fixed by molecular charge and spin.
Finite-temperature occupations resolve near-degenerate levels smoothly. Atomic
traces of its spectral functions distinguish occupied, unpaired and frontier
environments. This is an empirical descriptor without self-consistent iterations.
"""

import numpy as np
from scipy.special import expit
from .four_body import switch

TEMPERATURE = 0.16
WIDTH = 16


def population(energy, count):
    """Return occupations and fixed-count chemical-potential response weights."""
    if count < 0 or count > len(energy):
        raise ValueError("Spin occupations exceed valence space")
    if count == 0 or count == len(energy):
        return np.full(len(energy), float(count != 0)), np.zeros_like(energy)
    lo, hi = energy[0] - 6.0, energy[-1] + 6.0
    mu = 0.5 * (energy[count - 1] + energy[count])
    for _ in range(50):
        p = expit((mu - energy) / TEMPERATURE)
        err = np.sum(p) - count
        slope = p * (1 - p) / TEMPERATURE
        if abs(err) < 1.0e-13:
            break
        if err > 0:
            hi = mu
        else:
            lo = mu
        guess = mu - err / max(slope.sum(), 1.0e-14)
        mu = guess if lo < guess < hi else (lo + hi) * 0.5
    return p, slope


def matrix(z, r, groups, radii, derivatives):
    """Build an orthogonal s,p matrix and analytic derivatives of its bond blocks."""
    n = len(z)
    typ = groups[z]
    index = np.arange(4 * n).reshape(n, 4)
    active = np.ravel(
        np.column_stack((np.ones(n, bool), np.repeat((z > 2)[:, None], 3, axis=1)))
    )
    onsite_s = np.where(z <= 2, -1.68, -1.16 - 0.195 * typ)
    onsite_p = -0.48 - 0.19 * typ
    diag = np.column_stack((onsite_s, onsite_p, onsite_p, onsite_p))
    h = np.diag(diag.ravel())
    a, b = np.triu_indices(n, 1)
    delta = r[b] - r[a]
    d = np.linalg.norm(delta, axis=1)
    if np.any(d < 1.0e-10):
        raise ValueError("Coincident atoms in orbital population kernel")
    scale = radii[z[a]] + radii[z[b]]
    take = d < 2.35 * scale
    a, b, d, delta, scale = (val[take] for val in (a, b, d, delta, scale))
    x = d / scale
    u = delta / d[:, None]
    eye = np.eye(3)
    uu = u[:, :, None] * u[:, None, :]
    window, dwindow = switch(x, 1.48, 2.35)
    exp = np.exp(-2.65 * (x - 1.0))
    rho = 0.72 * window * exp
    drho = 0.72 * exp * (dwindow - 2.65 * window) / scale
    block = np.empty((len(a), 4, 4))
    block[:, 0, 0] = -0.91
    block[:, 0, 1:] = 1.12 * u
    block[:, 1:, 0] = -1.12 * u
    block[:, 1:, 1:] = -0.29 * eye + 1.35 * uu
    gmat = None
    if derivatives:
        du = (eye - uu) / d[:, None, None]
        ds = np.zeros((len(a), 4, 4, 3))
        ds[:, 0, 1:] = 1.12 * du
        ds[:, 1:, 0] = -1.12 * du
        ds[:, 1:, 1:] = 1.35 * (
            du[:, :, None, :] * u[:, None, :, None]
            + u[:, :, None, None] * du[:, None, :, :]
        )
        gmat = (
            rho[:, None, None, None] * ds
            + block[:, :, :, None] * (drho[:, None] * u)[:, None, None, :]
        )
    block *= rho[:, None, None]
    hb = h.reshape(n, 4, n, 4)
    hb[a, :, b, :] = block
    hb[b, :, a, :] = block.transpose(0, 2, 1)
    return h[np.ix_(active, active)], active, (a, b, gmat)


def environments(z, r, q, m, groups, radii, derivatives=False):
    """Evaluate site spectra and differentiate near degeneracies by divided differences."""
    n = len(z)
    h, active, (a, b, grad) = matrix(z, r, groups, radii, derivatives)
    energy, orbit = np.linalg.eigh(h)
    electrons = int(np.sum(np.where(z == 1, 1, np.where(z == 2, 2, groups[z])))) - int(
        q
    )
    if (electrons + m - 1) % 2:
        raise ValueError("Inconsistent charge and multiplicity")
    up = (electrons + m - 1) // 2
    alpha, sa = population(energy, up)
    beta, sb = population(energy, electrons - up)
    e = energy + 1.5
    contrast = alpha - beta
    total = alpha + beta
    front = alpha * (1 - alpha) + beta * (1 - beta)
    # Spectral filters are smooth functions. Polynomial moments of *occupied*
    # levels respond to local bond formation and charge redistribution.
    f = np.stack(
        [
            total,
            contrast,
            e * total,
            e * contrast,
            e * e * total,
            4 * front,
            np.tanh(e) * total,
            np.tanh(e / 0.50) * contrast,
        ]
    )
    df_da = np.stack(
        [
            np.ones_like(e),
            np.ones_like(e),
            e,
            e,
            e * e,
            4 * (1 - 2 * alpha),
            np.tanh(e),
            np.tanh(e / 0.50),
        ]
    )
    df_db = np.stack(
        [
            np.ones_like(e),
            -np.ones_like(e),
            e,
            -e,
            e * e,
            4 * (1 - 2 * beta),
            np.tanh(e),
            -np.tanh(e / 0.50),
        ]
    )
    direct = np.stack(
        [
            np.zeros_like(e),
            np.zeros_like(e),
            total,
            contrast,
            2 * e * total,
            np.zeros_like(e),
            (1 - np.tanh(e) ** 2) * total,
            2 * (1 - np.tanh(e / 0.50) ** 2) * contrast,
        ]
    )
    fp = direct - df_da * sa - df_db * sb
    owner = np.nonzero(active)[0] // 4
    p = (np.nonzero(active)[0] % 4) != 0
    group = 2 * owner + p
    diag = orbit * orbit
    weights = np.zeros((2 * n, len(energy)))
    np.add.at(weights, group, diag)
    val = (weights @ f.T).reshape(n, WIDTH)
    if not derivatives:
        return val, None
    gap = energy[:, None] - energy
    close = abs(gap) < 1.0e-7
    divided = np.divide(
        f[:, :, None] - f[:, None, :],
        gap,
        out=np.zeros((len(f), len(energy), len(energy))),
        where=~close,
    )
    divided = np.where(close[None], 0.5 * (fp[:, :, None] + fp[:, None, :]), divided)
    # The total occupation remains fixed under each coordinate displacement.
    va = weights @ (df_da * sa).T
    vb = weights @ (df_db * sb).T
    jac = np.zeros((n, WIDTH, r.size))
    full = np.zeros((4 * n, 4 * n, 3))
    for atom in range(n):
        full.fill(0.0)
        blocks = full.reshape(n, 4, n, 4, 3)
        for ids, sgn in (
            (np.flatnonzero(b == atom), 1.0),
            (np.flatnonzero(a == atom), -1.0),
        ):
            blocks[a[ids], :, b[ids], :, :] = sgn * grad[ids]
            blocks[b[ids], :, a[ids], :, :] = sgn * grad[ids].transpose(0, 2, 1, 3)
        dh = full[active][:, active].transpose(2, 0, 1)
        transformed = orbit.T @ dh @ orbit
        de = np.diagonal(transformed, axis1=1, axis2=2)
        chemical = va[:, :, None] * ((de @ sa) / max(sa.sum(), 1.0e-14)) + vb[
            :, :, None
        ] * ((de @ sb) / max(sb.sum(), 1.0e-14))
        # Only diagonal entries of Q df(H) Q^T are needed. Contract in
        # Cartesian batches so memory is bounded by a few orbital matrices.
        for axis in range(3):
            diagonal = np.einsum(
                "ij,fjk,ik->if",
                orbit,
                divided * transformed[axis],
                orbit,
                optimize=True,
            )
            atom_der = chemical[:, :, axis].copy()
            np.add.at(atom_der, group, diagonal)
            jac[:, :, 3 * atom + axis] = atom_der.reshape(n, WIDTH)
    return val, jac


class OccupationKernel:
    """Element-resolved Gaussian kernels on occupied orbital populations."""

    def __init__(
        self,
        groups,
        radii,
        centers,
        projections,
        length,
        weights,
        bond_populations=False,
    ):
        self.bond_populations = bool(bond_populations)
        if self.bond_populations:
            from .orbital_bonds import environments as bond_environments

            self.environments = bond_environments
        else:
            self.environments = environments
        self.groups = np.asarray(groups)
        self.radii = np.asarray(radii)
        self.centers = centers
        self.projections = projections
        self.length = float(length)
        self.weights = np.asarray(weights)
        self.size = sum(map(len, centers.values()))
        if self.length <= 0 or self.weights.shape != (self.size,):
            raise ValueError("Invalid occupation kernel")

    def design_state(self, z, r, charge, multiplicity, forces=False):
        x, der = self.environments(
            z, r, charge, multiplicity, self.groups, self.radii, forces
        )
        types = self.groups[z]
        out = np.zeros(self.size)
        jac = np.zeros((r.size, self.size)) if forces else None
        off = 0
        for g, c in self.centers.items():
            sl = slice(off, off + len(c))
            off = sl.stop
            take = types == g
            if not np.any(take):
                continue
            p = self.projections[g]
            diff = (x[take] @ p)[:, None, :] - c
            k = np.exp(-0.5 * np.einsum("akd,akd->ak", diff, diff) / self.length**2)
            out[sl] = k.sum(axis=0)
            if forces:
                jac[:, sl] = (
                    np.einsum("ak,akd,adq->qk", k, diff, p.T @ der[take], optimize=True)
                    / self.length**2
                )
        return out, jac

    def design(self, z, r, forces=False):
        return self.design_state(z, r, 0, 1, forces)

    def fields(self):
        d = dict(
            occupied_groups=self.groups,
            occupied_radii=self.radii,
            occupied_length=self.length,
            occupied_keys=list(self.centers),
            occupied_bond_populations=self.bond_populations,
            occupied_weights=self.weights,
        )
        for g, c in self.centers.items():
            d[f"occupied_center_{g}"] = c
            d[f"occupied_projection_{g}"] = self.projections[g]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["occupied_keys"]))
        return cls(
            d["occupied_groups"],
            d["occupied_radii"],
            {g: d[f"occupied_center_{g}"] for g in keys},
            {g: d[f"occupied_projection_{g}"] for g in keys},
            d["occupied_length"],
            d["occupied_weights"],
            d.get("occupied_bond_populations", False),
        )
