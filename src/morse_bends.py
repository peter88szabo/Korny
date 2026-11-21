"""An explicit finite kernel on bond stretches and three-atom bends.

The first four Morse oscillator bound-state functions describe a stretch in
units of the sum of the covalent radii. Products of two ground-state radial
functions multiply Legendre modes of the included angle. All features and
their Cartesian gradients vanish continuously when a participating bond breaks.
"""

from math import comb, factorial

import numpy as np
from numpy.polynomial import polynomial

from .four_body import switch


class MorseBendBasis:
    def __init__(self, groups, radii, weights):
        self.groups, self.radii = np.asarray(groups, int), np.asarray(radii, float)
        self.weights = np.asarray(weights, float)
        self.pair = self._pairs(9)
        self.neighbor_pair = self._pairs(3)
        self.size = 45 * 4 + 9 * 6 * 3
        if self.weights.shape != (self.size,):
            raise ValueError("Invalid Morse/bend energy readout")
        # lambda=5.5 makes s=lambda-n-1/2 a positive integer for n=0..3.
        # Associated Laguerre polynomials are evaluated in the original Morse
        # variable, with the same deterministic scale at training and deployment.
        self.coeff = np.zeros((4, 4))
        for n in range(4):
            alpha = 2 * (5 - n)
            self.coeff[n, : n + 1] = [
                (-1) ** k * comb(n + alpha, n - k) / factorial(k) for k in range(n + 1)
            ]
        grid = np.linspace(0.4, 2.5, 400)
        self.radial_scale = 1 / np.max(abs(self._modes(grid)[0]), axis=0)

    @staticmethod
    def _pairs(n):
        ix = np.triu_indices(n)
        result = np.empty((n, n), int)
        result[ix] = result[(ix[1], ix[0])] = np.arange(len(ix[0]))
        return result

    def _modes(self, x):
        y = 11 * np.exp(-2.3 * (x - 1))
        s = np.arange(5, 1, -1)
        p = polynomial.polyval(y, self.coeff.T).T
        dp = polynomial.polyval(y, polynomial.polyder(self.coeff.T)).T
        envelope = y[:, None] ** s * np.exp(-y[:, None] / 2)
        raw = envelope * p
        draw = -2.3 * envelope * ((s - y[:, None] / 2) * p + y[:, None] * dp)
        gate, dgate = switch(x, 1.8, 2.5)
        return raw * gate[:, None], draw * gate[:, None] + raw * dgate[:, None]

    def design(self, z, r, forces=False):
        z, r = np.asarray(z, int), np.asarray(r, float)
        typ = self.groups[z]
        i, j = np.triu_indices(len(z), 1)
        dr = r[j] - r[i]
        distance = np.linalg.norm(dr, axis=1)
        scale = self.radii[z[i]] + self.radii[z[j]]
        take = distance < 2.5 * scale
        i, j, dr, distance, scale = (a[take] for a in (i, j, dr, distance, scale))
        if np.any(distance < 1e-10):
            raise ValueError("Coincident nuclei")
        unit = dr / distance[:, None]
        mode, der = self._modes(distance / scale)
        mode *= self.radial_scale
        der *= self.radial_scale / scale[:, None]
        ids = self.pair[typ[i], typ[j], None] * 4 + np.arange(4)
        out = np.bincount(ids.ravel(), mode.ravel(), minlength=self.size).astype(float)
        jac = np.zeros((len(z), self.size, 3)) if forces else None
        if forces:
            f = der[:, :, None] * unit[:, None, :]
            np.add.at(jac, (i[:, None], ids), f)
            np.add.at(jac, (j[:, None], ids), -f)

        # Sort both orientations of the edges. Build only triples with a
        # common central atom, avoiding triples of mutually distant atoms.
        center = np.r_[i, j]
        order = np.argsort(center, kind="stable")
        start = np.searchsorted(center[order], np.arange(len(z) + 1))
        deg = np.diff(start)
        count = deg * (deg - 1) // 2
        if np.any(count):
            hubs = np.repeat(np.arange(len(z)), count)
            # Vectorized inverse triangular indexing within each neighbor list.
            index = np.arange(len(hubs)) - np.repeat(np.cumsum(count) - count, count)
            second = np.floor((1 + np.sqrt(1 + 8 * index)) / 2).astype(int)
            first = index - second * (second - 1) // 2
            a, b = order[start[hubs] + first], order[start[hubs] + second]
            other = np.r_[j, i]
            edge_a, edge_b = a % len(i), b % len(i)
            ea = unit[edge_a] * np.where(a < len(i), 1.0, -1.0)[:, None]
            eb = unit[edge_b] * np.where(b < len(i), 1.0, -1.0)[:, None]
            cos = np.einsum("ij,ij->i", ea, eb)
            pol = np.column_stack(
                (cos, (3 * cos * cos - 1) / 2, (5 * cos**3 - 3 * cos) / 2)
            )
            dpol = np.column_stack(
                (np.ones_like(cos), 3 * cos, (15 * cos * cos - 3) / 2)
            )
            wa, wb = mode[edge_a, 0], mode[edge_b, 0]
            w = wa * wb
            cat = np.where(typ == 0, 0, np.where(typ == 4, 1, 2))
            key = 6 * typ[hubs] + self.neighbor_pair[cat[other[a]], cat[other[b]]]
            cols = 180 + key[:, None] * 3 + np.arange(3)
            out += np.bincount(
                cols.ravel(), (w[:, None] * pol).ravel(), minlength=self.size
            )
            if forces:
                v1 = (
                    (der[edge_a, 0] * wb)[:, None, None]
                    * pol[:, :, None]
                    * ea[:, None, :]
                )
                v2 = (
                    (der[edge_b, 0] * wa)[:, None, None]
                    * pol[:, :, None]
                    * eb[:, None, :]
                )
                v1 += (
                    (w / distance[edge_a])[:, None, None]
                    * dpol[:, :, None]
                    * (eb - cos[:, None] * ea)[:, None, :]
                )
                v2 += (
                    (w / distance[edge_b])[:, None, None]
                    * dpol[:, :, None]
                    * (ea - cos[:, None] * eb)[:, None, :]
                )
                np.add.at(jac, (other[a, None], cols), -v1)
                np.add.at(jac, (other[b, None], cols), -v2)
                np.add.at(jac, (hubs[:, None], cols), v1 + v2)
        return out, (
            jac.transpose(0, 2, 1).reshape(r.size, self.size) if forces else None
        )

    def fields(self):
        return dict(
            morse_weights=self.weights, morse_groups=self.groups, morse_radii=self.radii
        )

    @classmethod
    def load(cls, d):
        return cls(d["morse_groups"], d["morse_radii"], d["morse_weights"])
