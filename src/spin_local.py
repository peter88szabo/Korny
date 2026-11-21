"""Electronic spin populations and a conservative kernel for open-shell molecules.

Geometry-dependent site scores allocate a fixed number of unpaired electrons.
A signed polarization term permits negative local spin populations while
preserving their molecular sum. Molecular energy features are sums of local
environments weighted by these spin populations. The force includes both the
motion of a local environment and the response of spin elsewhere in the molecule.
"""

import numpy as np

from .four_body import switch

CENTERS = np.array([0.92, 1.18, 1.55, 2.15])
WIDTHS = np.array([0.14, 0.20, 0.31, 0.45])
GROUPS = 9
DIM = 1 + GROUPS * len(CENTERS) + 3


def local_sites(groups, radii, z, r, forces=False):
    """Smooth element-resolved coordination densities and their Cartesian Jacobian."""
    z = np.asarray(z)
    r = np.asarray(r, float)
    n = len(z)
    types = groups[z]
    a, b = np.triu_indices(n, 1)
    dr = r[a] - r[b]
    d = np.linalg.norm(dr, axis=1)
    if np.any(d < 1.0e-9):
        raise ValueError("Coincident nuclei")
    scale = radii[z[a]] + radii[z[b]]
    dist = d / scale
    keep = dist < 4.0
    a, b, dr, d, scale, dist = (v[keep] for v in (a, b, dr, d, scale, dist))
    delta = (dist[:, None] - CENTERS) / WIDTHS
    gauss = np.exp(-0.5 * delta**2)
    cutoff, slope = switch(dist, 2.85, 4.0)
    shell = gauss * cutoff[:, None]
    raw = np.zeros((n, GROUPS, len(CENTERS)))
    np.add.at(raw, (a, types[b]), shell)
    np.add.at(raw, (b, types[a]), shell)
    dens = np.tanh(raw / 3)
    s = raw.sum(axis=1)
    h = np.tanh(s[:, :2] / 3)
    cross = np.column_stack((h[:, 0] ** 2, h[:, 0] * h[:, 1], h[:, 1] ** 2))
    x = np.column_stack((np.ones(n), dens.reshape(n, -1), cross))
    if not forces:
        return types, x, None
    jac = np.zeros((n, GROUPS, len(CENTERS), n, 3))
    dw = gauss * slope[:, None] - shell * delta / WIDTHS
    pair = dw[:, :, None] * (dr / (d * scale)[:, None])[:, None, :]
    for center, neighbor in ((a, b), (b, a)):
        np.add.at(jac, (center, types[neighbor], slice(None), a), pair)
        np.add.at(jac, (center, types[neighbor], slice(None), b), -pair)
    jac = jac.reshape(n, GROUPS, len(CENTERS), 3 * n)
    dh = jac[:, :, :2].sum(axis=1) * ((1 - h**2) / 3)[:, :, None]
    dcr = np.stack(
        [
            2 * h[:, 0, None] * dh[:, 0],
            h[:, 0, None] * dh[:, 1] + h[:, 1, None] * dh[:, 0],
            2 * h[:, 1, None] * dh[:, 1],
        ],
        axis=1,
    )
    dd = jac * ((1 - dens**2) / 3)[:, :, :, None]
    j = np.concatenate((np.zeros((n, 1, 3 * n)), dd.reshape(n, -1, 3 * n), dcr), axis=1)
    return types, x, j


class SpinLocalKernel:
    """Predict signed atom populations, then a smooth spin-dependent energy contribution."""

    def __init__(self, groups, radii, logits, polarization, weights):
        self.groups = np.asarray(groups, int)
        self.radii = np.asarray(radii, float)
        self.logits = np.asarray(logits, float)
        self.polarization = np.asarray(polarization, float)
        self.weights = np.asarray(weights, float)
        self.size = GROUPS * DIM
        expected = (3, GROUPS, DIM)
        if (
            self.logits.shape != expected
            or self.polarization.shape != expected
            or self.weights.shape != (self.size,)
        ):
            raise ValueError("Invalid local electronic spin model")

    def sites(self, z, r, forces=False):
        return local_sites(self.groups, self.radii, z, r, forces)

    @staticmethod
    def charge_channel(q):
        return int(np.sign(q)) + 1

    def population(self, types, x, q, m, j=None):
        """Normalize the positive spin distribution and add a zero-sum signed response."""
        k = self.charge_channel(q)
        a = self.logits[k, types]
        b = self.polarization[k, types]
        logits = np.einsum("nd,nd->n", x, a)
        p = np.exp(logits - np.max(logits))
        p /= p.sum()
        pol = np.einsum("nd,nd->n", x, b)
        shift = 1.0 + pol - p @ pol
        s = (m - 1) * p * shift
        if j is None:
            return s, None
        dlog = np.einsum("nd,ndc->nc", a, j)
        dp = p[:, None] * (dlog - p @ dlog)
        dh = np.einsum("nd,ndc->nc", b, j)
        ds = (m - 1) * (shift[:, None] * dp + p[:, None] * (dh - (pol @ dp + p @ dh)))
        return s, ds

    def design_state(self, z, r, q, m, forces=False):
        out = np.zeros((GROUPS, DIM))
        der = np.zeros((GROUPS, DIM, r.size)) if forces else None
        if m > 1:
            types, x, j = self.sites(z, r, forces)
            spin, ds = self.population(types, x, q, m, j)
            np.add.at(out, types, spin[:, None] * x)
            if forces:
                np.add.at(
                    der,
                    types,
                    -(spin[:, None, None] * j + x[:, :, None] * ds[:, None, :]),
                )
        return out.ravel(), der.reshape(self.size, r.size).T if forces else None

    def fields(self):
        return {
            "spinlocal_groups": self.groups,
            "spinlocal_radii": self.radii,
            "spinlocal_logits": self.logits,
            "spinlocal_polarization": self.polarization,
            "spinlocal_weights": self.weights,
        }

    @classmethod
    def load(cls, d):
        return cls(
            *(
                d["spinlocal_" + s]
                for s in ("groups", "radii", "logits", "polarization", "weights")
            )
        )
