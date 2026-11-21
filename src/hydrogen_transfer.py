"""Smooth, atom-order-invariant connectivity features for a migrating H.

A hydrogen can share covalent contact with two nearby heavy atoms. The three
coordinates resolve which bond is stronger, the total occupancy of both bonds,
and their included angle. A smooth cutoff on each contact makes the
contribution vanish at fragment separation. The same coordinate derivatives
drive energy, force and Hessian evaluation along a reaction.
"""

import numpy as np
from numpy.polynomial import chebyshev

from .four_body import switch

# Low-order products of the bond imbalance, total occupancy, and angle.
POWERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (4, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
        (0, 0, 1),
        (0, 0, 2),
        (2, 1, 0),
        (1, 1, 0),
        (2, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 1),
    ]
)


class HydrogenTransferKernel:
    """Sum smooth two-contact kernels over every H and nearby heavy-atom pair."""

    def __init__(self, radii, family, coefficients, weights, angular_window=None):
        self.radii = np.asarray(radii, dtype=float)
        self.family = str(family)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.angular_window = (
            None if angular_window is None else tuple(map(float, angular_window))
        )
        if self.angular_window is not None and not (
            -1 <= self.angular_window[0] < self.angular_window[1] <= 1
        ):
            raise ValueError(
                "Invalid cosine interval for the H-transfer angular window"
            )
        if self.family not in (
            "chebyshev",
            "krawtchouk",
        ) or self.coefficients.shape != (5, 5):
            raise ValueError("Invalid H-transfer polynomial basis")
        self.groups = np.full(len(self.radii), 5, dtype=int)
        # Resolve C, N, O, S and halogens. Remaining supported heavy atoms
        # share a channel while retaining element-specific covalent radii.
        for group, elements in enumerate(
            ((6,), (7,), (8,), (16,), (9, 17, 35, 53, 85))
        ):
            self.groups[[z for z in elements if z < len(self.radii)]] = group
        self.pairs = np.empty((6, 6), int)
        a, b = np.triu_indices(6)
        self.pairs[a, b] = self.pairs[b, a] = np.arange(len(a))
        self.size = len(a) * len(POWERS)
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (self.size,):
            raise ValueError("Incompatible H-transfer energy weights")

    def contacts(self, z, r):
        """Enumerate H-centred triples with two in-range heavy atoms."""
        h, heavy = np.flatnonzero(z == 1), np.flatnonzero(z != 1)
        if len(h) == 0 or len(heavy) < 2:
            return None
        delta = r[heavy] - r[h, None]
        length = np.linalg.norm(delta, axis=2)
        if np.any(length < 1e-10):
            raise ValueError("Coincident H and heavy atom")
        scale = self.radii[1] + self.radii[z[heavy]]
        reduced = length / scale
        owners, neighbors = np.nonzero(reduced < 2.12)
        degree = np.bincount(owners, minlength=len(h))
        maximum = degree.max()
        if maximum < 2:
            return None
        # Enumerate pairs within each H's contact list. A large molecule with
        # sparse contacts never needs an all-heavy-pairs-by-H array.
        table = np.zeros((len(h), maximum), dtype=int)
        offset = degree.cumsum() - degree
        table[owners, np.arange(len(owners)) - offset[owners]] = neighbors
        left, right = np.triu_indices(maximum, 1)
        owner, pair = np.nonzero(right[None, :] < degree[:, None])
        a, b = table[owner, left[pair]], table[owner, right[pair]]
        groups = self.groups[z[heavy]]
        swap = groups[a] > groups[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        ends = np.stack((a, b), axis=1)
        lengths = length[owner[:, None], ends]
        unit = delta[owner[:, None], ends] / lengths[:, :, None]
        return (
            self.pairs[groups[a], groups[b]],
            groups[a] == groups[b],
            h[owner],
            heavy[ends],
            lengths,
            scale[ends],
            unit,
        )

    def coordinates(self, contact, derivatives=False):
        """Compute two smooth bond occupancies, their angle, and local slopes."""
        _, _, _, _, length, scale, unit = contact
        reduced = length / scale
        occ, docc = switch(reduced, 0.88, 1.90)
        gate, dgate = switch(reduced, 1.45, 2.12)
        cos = np.einsum("nd,nd->n", unit[:, 0], unit[:, 1])
        x = np.column_stack((occ[:, 0] - occ[:, 1], occ.sum(1) - 1, cos))
        w = gate.prod(1)
        # An optional angle window confines this added energy to approximately
        # linear abstractions. Both switches have continuous second derivatives.
        angular, dangular = (
            (1.0, 0.0)
            if self.angular_window is None
            else switch(cos, *self.angular_window)
        )
        if not derivatives:
            return x, w * angular, None, None
        # Derivatives w.r.t. the outward Cartesian vectors H→A and H→B.
        slope = (docc / scale)[:, :, None] * unit
        jac = np.stack(
            (
                slope * np.array([1, -1])[None, :, None],
                slope,
                (unit[:, ::-1] - cos[:, None, None] * unit) / length[:, :, None],
            ),
            axis=1,
        )
        dw = (dgate / scale * gate[:, ::-1])[:, :, None] * unit
        dw = (
            dw * np.asarray(angular)[..., None, None]
            + (w * dangular)[:, None, None] * jac[:, 2]
        )
        return x, w * angular, jac, dw

    def modes(self, x, symmetric, derivatives=False):
        """Evaluate tensor products and symmetrize chemically identical endpoints."""
        vals = chebyshev.chebvander(x, 4) @ self.coefficients.T
        at = [vals[:, i, POWERS[:, i]] for i in range(3)]
        v = at[0] * at[1] * at[2]
        odd = (POWERS[:, 0] % 2) == 1
        mask = symmetric[:, None] & odd
        v[mask] = 0.0
        if not derivatives:
            return v, None
        dc = chebyshev.chebder(self.coefficients.T)
        dv = chebyshev.chebvander(x, 3) @ dc
        jac = np.stack(
            [
                dv[:, i, POWERS[:, i]] * at[(i + 1) % 3] * at[(i + 2) % 3]
                for i in range(3)
            ],
            axis=-1,
        )
        jac[mask] = 0.0
        return v, jac

    def design(self, z, r, forces=False):
        """Scatter the energy features and their negative derivatives to the atoms."""
        z, r = np.asarray(z, int), np.asarray(r, float)
        out = np.zeros(self.size)
        force = np.zeros((len(z), self.size, 3)) if forces else None
        c = self.contacts(z, r)
        if c is not None:
            pair, symmetric, h, ends = c[:4]
            x, w, jac, dw = self.coordinates(c, forces)
            val, dv = self.modes(x, symmetric, forces)
            cols = pair[:, None] * len(POWERS) + np.arange(len(POWERS))
            np.add.at(out, cols, val * w[:, None])
            if forces:
                grad = w[:, None, None, None] * np.einsum("nmc,nced->nmed", dv, jac)
                grad += val[:, :, None, None] * dw[:, None, :, :]
                for e in range(2):
                    np.add.at(force, (ends[:, e, None], cols), -grad[:, :, e])
                np.add.at(force, (h[:, None], cols), grad.sum(2))
        return out, (
            force.transpose(0, 2, 1).reshape(r.size, self.size) if forces else None
        )

    def fields(self):
        fields = dict(
            htransfer_family=self.family,
            htransfer_radii=self.radii,
            htransfer_coeff=self.coefficients,
            htransfer_weights=self.weights,
        )
        if self.angular_window is not None:
            fields["htransfer_angular_window"] = self.angular_window
        return fields

    @classmethod
    def load(cls, d):
        return cls(
            d["htransfer_radii"],
            str(d["htransfer_family"]),
            d["htransfer_coeff"],
            d["htransfer_weights"],
            d.get("htransfer_angular_window"),
        )
