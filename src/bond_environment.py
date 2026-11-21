"""Smooth bond-centered features for reaction energetics and conservative forces.

For each H/C/N/O pair, encode its scaled length and element-resolved coordination
at both ends. A neighbor can coordinate to two atoms during a hydrogen transfer.
The energy is additive over bonds and vanishes continuously at fragment separation.
"""

import numpy as np
from .four_body import RADII, switch

ELEMENTS = (1, 6, 7, 8)
PAIRS = tuple((a, b) for i, a in enumerate(ELEMENTS) for b in ELEMENTS[i:])


class BondBasis:
    """Random Fourier features of a Gaussian kernel on the two coordination shells."""

    def __init__(self, frequencies, phases):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.phases = np.asarray(phases, dtype=float)
        if (
            self.frequencies.shape[:2] != self.phases.shape
            or self.frequencies.shape[-1] not in (9, 17)
            or self.phases.shape[0] != len(PAIRS)
        ):
            raise ValueError("Incompatible bond kernel parameter arrays")
        self.angular = self.frequencies.shape[-1] == 17
        self.width = self.phases.shape[1]
        self.size = self.phases.size

    @classmethod
    def generate(cls, width, length, angular=False):
        # Fix the feature draw across kernel length trials. Distance is in units
        # of summed covalent radii, coordination components in effective neighbors.
        rng = np.random.default_rng(7312026)
        frequencies = (
            rng.normal(size=(len(PAIRS), width, 17 if angular else 9)) / length
        )
        frequencies[:, :, 0] *= 4.0
        return cls(frequencies, rng.uniform(0, 2 * np.pi, (len(PAIRS), width)))

    def design(self, z, r, forces=False):
        """Return energy features and their negative Cartesian Jacobian.

        Reverse propagation through the two coordination shells includes the force
        on third atoms. Equivalent ends of a homonuclear bond are symmetrized.
        """
        out = np.zeros((len(PAIRS), self.width))
        jac = np.zeros((len(z), 3, self.size)) if forces else None
        ix = np.flatnonzero(np.isin(z, ELEMENTS))
        if len(ix) < 2:
            return out.ravel(), jac.reshape(r.size, self.size) if forces else None
        z0 = z[ix]
        radii = np.array([RADII[int(a)] for a in z0])
        i, j = np.triu_indices(len(ix), 1)
        dr = r[ix[j]] - r[ix[i]]
        d = np.linalg.norm(dr, axis=1)
        scale = radii[i] + radii[j]
        active = d < 1.95 * scale
        i, j, dr, d, scale = (v[active] for v in (i, j, dr, d, scale))
        x = d / scale
        neighbor, dn = switch(x, 0.98, 1.65)
        gate, dg = switch(x, 1.25, 1.95)
        element = np.searchsorted(ELEMENTS, z0)
        coord = np.zeros((len(ix), 4))
        np.add.at(coord, (i, element[j]), neighbor)
        np.add.at(coord, (j, element[i]), neighbor)
        if self.angular:
            # Element-resolved sum of directed unit bonds, used for approach angles.
            unit = dr / d[:, None]
            vectors = np.zeros((len(ix), 4, 3))
            np.add.at(vectors, (i, element[j]), neighbor[:, None] * unit)
            np.add.at(vectors, (j, element[i]), -neighbor[:, None] * unit)
        norm = np.sqrt(2 / self.width)
        for k, (a, b) in enumerate(PAIRS):
            use = np.flatnonzero(
                (np.minimum(z0[i], z0[j]) == a) & (np.maximum(z0[i], z0[j]) == b)
            )
            if not len(use):
                continue
            u, v = (
                np.where(z0[i[use]] == a, i[use], j[use]),
                np.where(z0[i[use]] == a, j[use], i[use]),
            )
            # Exclude the central bond from both endpoint coordination vectors.
            ca, cb = coord[u].copy(), coord[v].copy()
            ca[:, ELEMENTS.index(b)] -= neighbor[use]
            cb[:, ELEMENTS.index(a)] -= neighbor[use]
            orientations = (
                [(u, v, ca, cb)] if a != b else [(u, v, ca, cb), (v, u, cb, ca)]
            )
            factor = norm / len(orientations)
            for left, right, cl, cr in orientations:
                features = np.column_stack((x[use] - 1.0, cl, cr))
                if self.angular:
                    orient = np.where(left == i[use], 1.0, -1.0)
                    axis = orient[:, None] * unit[use]
                    al = np.einsum("nhk,nk->nh", vectors[left], axis)
                    ar = -np.einsum("nhk,nk->nh", vectors[right], axis)
                    al[:, element[right[0]]] -= neighbor[use]
                    ar[:, element[left[0]]] -= neighbor[use]
                    features = np.column_stack((features, al, ar))
                phase = features @ self.frequencies[k].T + self.phases[k]
                cos = np.cos(phase) * factor
                out[k] += gate[use] @ cos
                if not forces:
                    continue
                # For each feature propagate dE/dx first, then dx/dR.
                sens = -np.sin(phase) * (gate[use] * factor)[:, None]
                env = np.zeros((len(ix), 4, self.width))
                dv = np.zeros((len(x), self.width))
                for h in range(4):
                    np.add.at(env[:, h], left, sens * self.frequencies[k, :, 1 + h])
                    np.add.at(env[:, h], right, sens * self.frequencies[k, :, 5 + h])
                direct = (
                    self.frequencies[k, :, 1 + element[right[0]]]
                    + self.frequencies[k, :, 5 + element[left[0]]]
                )
                dv[use] += dg[use, None] * cos + sens * (
                    self.frequencies[k, :, 0] - dn[use, None] * direct
                )
                dv += dn[:, None] * (env[i, element[j]] + env[j, element[i]])
                contribution = (dr / (d * scale)[:, None])[:, :, None] * dv[:, None, :]
                if self.angular:
                    # Differentiate both neighbor directions and the bond axis. The
                    # projection onto its perpendicular plane is the Jacobian of a unit vector.
                    vl = self.frequencies[k, :, 9:13]
                    vr = self.frequencies[k, :, 13:17]
                    vec_grad = np.zeros((len(ix), 4, 3, self.width))
                    axis_sens = axis[:, :, None] * sens[:, None, :]
                    for h in range(4):
                        np.add.at(vec_grad[:, h], left, axis_sens * vl[:, h])
                        np.add.at(vec_grad[:, h], right, -axis_sens * vr[:, h])
                    vg = vec_grad[i, element[j]] - vec_grad[j, element[i]]
                    radial = np.einsum("nk,nkf->nf", unit, vg)
                    central = vl[:, element[right[0]]] + vr[:, element[left[0]]]
                    add_radial = dn[:, None] * radial
                    add_radial[use] -= dn[use, None] * sens * central
                    du = neighbor[:, None, None] * vg
                    axis_grad = np.einsum("nhk,fh->nkf", vectors[left], vl) - np.einsum(
                        "nhk,fh->nkf", vectors[right], vr
                    )
                    du[use] += orient[:, None, None] * sens[:, None, :] * axis_grad
                    perpendicular = (
                        du
                        - unit[:, :, None]
                        * np.einsum("nk,nkf->nf", unit, du)[:, None, :]
                    )
                    contribution += (
                        perpendicular / d[:, None, None]
                        + unit[:, :, None] * (add_radial / scale[:, None])[:, None, :]
                    )
                block = jac[:, :, k * self.width : (k + 1) * self.width]
                np.add.at(block, ix[i], contribution)
                np.add.at(block, ix[j], -contribution)
        return out.ravel(), jac.reshape(r.size, self.size) if forces else None
