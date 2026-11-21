"""Local kernels on the orientation and connectivity of valence orbitals.

The diagonal blocks of powers of a real, symmetric hopping matrix describe
closed paths through overlapping s and p orbitals. A change of orientation
changes the sigma and pi overlaps even at fixed bond lengths and coordination.
The matrix is a geometry descriptor with fixed, dimensionless parameters. Its
moments are additive between disconnected fragments and require no electronic
structure calculation. Orthonormal rotations of the p functions leave each
site trace unchanged.
"""

import numpy as np

from .four_body import switch


# Separate hydrogen's s site from the eight main-group valence families.
CHANNELS = 9
POWERS = (2, 3, 4, 5, 6, 7)
WIDTH = 2 * CHANNELS + 4 * len(POWERS)


def environments(z, r, groups, radii, gradient=False):
    """Return one invariant vector per nucleus and its Cartesian Jacobian.

    Radius-normalized separations set the length scale. The smooth gate goes
    to zero with three derivatives. At an isolated atom all features vanish.
    Cartesian derivatives use d(H^k) = sum_l H^l (dH) H^(k-1-l).
    Only interatomic blocks on actual edges have a geometry derivative.
    """
    n = len(z)
    sites = np.zeros((n, WIDTH))
    jac = np.zeros((n, WIDTH, n, 3)) if gradient else None
    types = groups[z]
    a, b = np.triu_indices(n, 1)
    dr = r[b] - r[a]
    dist = np.linalg.norm(dr, axis=1)
    if np.any(dist < 1.0e-10):
        raise ValueError("Coincident atoms in orbital descriptor")
    scale = radii[z[a]] + radii[z[b]]
    x = dist / scale
    mask = x < 2.4
    a, b, dr, dist, scale, x = [v[mask] for v in (a, b, dr, dist, scale, x)]
    if not len(a):
        return sites, jac
    u = dr / dist[:, None]
    gate, dg = switch(x, 1.55, 2.4)
    unit = np.eye(3)
    uu = u[:, :, None] * u[:, None, :]
    du = (unit - uu) / dist[:, None, None]
    for j, (lo, hi) in enumerate(((0.94, 1.57), (1.22, 2.20))):
        s, ds = switch(x, lo, hi)
        ca, cb = j * CHANNELS + types[b], j * CHANNELS + types[a]
        np.add.at(sites, (a, ca), s)
        np.add.at(sites, (b, cb), s)
        if gradient:
            grad = ds[:, None] * u / scale[:, None]
            for origin, key in ((a, ca), (b, cb)):
                np.add.at(jac, (origin, key, b), grad)
                np.add.at(jac, (origin, key, a), -grad)

    # Hydrogen and helium have a single s channel. The zero p slots make the
    # block layout uniform and contribute exactly zero to the moments.
    p = (z > 2).astype(float)
    both = p[a] * p[b]
    local_s = -0.10 - 0.055 * types
    local_p = (0.20 - 0.052 * types) * p
    onsite = np.column_stack((local_s, local_p, local_p, local_p))
    for variant, (decay, pi) in enumerate(((1.65, -0.27), (3.3, -0.14))):
        rho = gate * np.exp(-decay * (x - 1.0))
        d_rho = (dg - decay * gate) * np.exp(-decay * (x - 1.0)) / scale
        block = np.empty((len(a), 4, 4))
        block[:, 0, 0] = -0.57
        block[:, 0, 1:] = 0.85 * p[b, None] * u
        block[:, 1:, 0] = -0.85 * p[a, None] * u
        block[:, 1:, 1:] = both[:, None, None] * (pi * unit + (1.0 - pi) * uu)
        if gradient:
            ds = np.zeros((len(a), 4, 4, 3))
            ds[:, 0, 1:] = 0.85 * p[b, None, None] * du
            ds[:, 1:, 0] = -0.85 * p[a, None, None] * du
            ds[:, 1:, 1:] = (
                (1.0 - pi)
                * both[:, None, None, None]
                * (
                    du[:, :, None, :] * u[:, None, :, None]
                    + u[:, :, None, None] * du[:, None, :, :]
                )
            )
            dh = 0.64 * (
                rho[:, None, None, None] * ds
                + block[:, :, :, None] * (d_rho[:, None] * u)[:, None, None, :]
            )
        block *= 0.64 * rho[:, None, None]
        h = np.diag(onsite.ravel())
        blocks = h.reshape(n, 4, n, 4)
        blocks[a, :, b, :] = block
        blocks[b, :, a, :] = block.transpose(0, 2, 1)
        powers = [np.eye(4 * n), h]
        for k in range(2, max(POWERS) + 1):
            powers.append(powers[-1] @ h)
            off = 2 * CHANNELS + 2 * (variant * len(POWERS) + k - 2)
            diag = powers[k].diagonal().reshape(n, 4) - onsite**k
            sites[:, off] = diag[:, 0]
            sites[:, off + 1] = diag[:, 1:].sum(axis=1)
            if gradient:
                d = np.zeros((n, 4, len(a), 3))
                for l in range(k):
                    left = powers[l].reshape(n, 4, n, 4)[:, :, a, :]
                    right = powers[k - 1 - l].reshape(n, 4, n, 4)[:, :, b, :]
                    # Both transposed off-diagonal blocks vary together.
                    d += 2 * np.einsum(
                        "icep,epqv,iceq->icev", left, dh, right, optimize=True
                    )
                # Repeated atom indices occur in different bonds.
                for f, der in ((off, d[:, 0]), (off + 1, d[:, 1:].sum(axis=1))):
                    dest = jac[:, f].transpose(1, 0, 2)
                    np.add.at(dest, b, der.transpose(1, 0, 2))
                    np.add.at(dest, a, -der.transpose(1, 0, 2))
    return sites, jac


class OrbitalMomentKernel:
    """An element-group Gaussian readout of projected local orbital moments."""

    def __init__(self, groups, radii, centers, projections, length, weights):
        self.groups = np.asarray(groups, dtype=int)
        self.radii = np.asarray(radii, dtype=float)
        self.centers = centers
        self.projections = projections
        self.length = float(length)
        self.weights = np.asarray(weights)
        self.size = sum(len(c) for c in centers.values())
        if self.length <= 0 or self.weights.shape != (self.size,):
            raise ValueError("Invalid orbital-moment readout")

    def design(self, z, r, forces=False):
        raw, jac = environments(z, r, self.groups, self.radii, forces)
        tp = self.groups[z]
        values = np.zeros(self.size)
        df = np.zeros((r.size, self.size)) if forces else None
        start = 0
        for g, c in self.centers.items():
            sl = slice(start, start + len(c))
            start = sl.stop
            ix = np.flatnonzero(tp == g)
            if not len(ix):
                continue
            proj = self.projections[g]
            delta = (raw[ix] @ proj)[:, None, :] - c
            k = np.exp(-0.5 * np.einsum("acd,acd->ac", delta, delta) / self.length**2)
            empty = np.exp(-0.5 * np.einsum("cd,cd->c", c, c) / self.length**2)
            values[sl] = k.sum(axis=0) - len(ix) * empty
            if forces:
                projected = proj.T @ jac[ix].reshape(len(ix), WIDTH, -1)
                df[:, sl] = (
                    np.einsum("ac,acd,adi->ic", k, delta, projected, optimize=True)
                    / self.length**2
                )
        return values, df

    def fields(self):
        d = dict(
            orbital_groups=self.groups,
            orbital_radii=self.radii,
            orbital_length=self.length,
            orbital_keys=list(self.centers),
            orbital_weights=self.weights,
        )
        for g, c in self.centers.items():
            d[f"orbital_center_{g}"] = c
            d[f"orbital_projection_{g}"] = self.projections[g]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["orbital_keys"]))
        return cls(
            d["orbital_groups"],
            d["orbital_radii"],
            {g: d[f"orbital_center_{g}"] for g in keys},
            {g: d[f"orbital_projection_{g}"] for g in keys},
            d["orbital_length"],
            d["orbital_weights"],
        )
