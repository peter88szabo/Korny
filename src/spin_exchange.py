"""A local neighbor update for atomic spin populations on an open-shell PES.

Every edge transfers equal and opposite amounts of spin between its two atoms.
The update can therefore polarize a radical's neighbors without changing the
total imposed by multiplicity. A compact, smooth radial function turns each
edge off at separation. The molecular energy readout uses the *change* from
the original populations, allowing this feature to augment a trained model.
"""

import numpy as np

from .four_body import switch
from .spin_local import DIM, GROUPS, SpinLocalKernel

RADII = np.array([0.95, 1.45, 2.05])
WIDTHS = np.array([0.20, 0.32, 0.52])
PAIRS = GROUPS * (GROUPS + 1) // 2


def edge_basis(types, z, r, radii):
    """Return an undirected edge basis and its derivative in scaled distance."""
    i, j = np.triu_indices(len(z), 1)
    vectors = r[i] - r[j]
    d = np.linalg.norm(vectors, axis=1)
    if np.any(d < 1e-9):
        raise ValueError("Coincident nuclei in spin exchange")
    scale = radii[z[i]] + radii[z[j]]
    t = d / scale
    use = t < 3.2
    i, j, vectors, d, scale, t = (v[use] for v in (i, j, vectors, d, scale, t))
    cutoff, slope = switch(t, 2.5, 3.2)
    a = (t[:, None] - RADII) / WIDTHS
    gaussian = np.exp(-0.5 * a * a)
    values = cutoff[:, None] * gaussian
    deriv = (slope[:, None] - cutoff[:, None] * a / WIDTHS) * gaussian
    hi = np.maximum(types[i], types[j]).astype(int)
    pair = hi * (hi + 1) // 2 + np.minimum(types[i], types[j])
    return i, j, pair, values, deriv, vectors / (d * scale)[:, None]


def population_design(types, z, r, radii, spin):
    """Design matrix for fitting the exchange to atom-resolved population labels."""
    i, j, pair, values, _, _ = edge_basis(types, z, r, radii)
    x = np.zeros((len(z), PAIRS, len(RADII)))
    transfer = (spin[j] - spin[i])[:, None] * values
    np.add.at(x, (i, pair), transfer)
    np.add.at(x, (j, pair), -transfer)
    return x.reshape(len(z), -1)


class SpinExchangeKernel(SpinLocalKernel):
    def __init__(self, groups, radii, logits, polarization, weights, exchange):
        super().__init__(groups, radii, logits, polarization, weights)
        self.exchange = np.asarray(exchange, float)
        if self.exchange.shape != (3, PAIRS, len(RADII)) or not np.all(
            np.isfinite(self.exchange)
        ):
            raise ValueError("Invalid pair spin exchange coefficients")

    def redistribution(self, z, r, types, spin, q, ds=None):
        """Differentiate both the transfer distances and the source populations."""
        i, j, pair, values, dvalues, grad = edge_basis(types, z, r, self.radii)
        w = self.exchange[self.charge_channel(q), pair]
        amplitude = np.einsum("ed,ed->e", values, w)
        diff = spin[j] - spin[i]
        flow = amplitude * diff
        delta = np.zeros(len(z))
        np.add.at(delta, i, flow)
        np.add.at(delta, j, -flow)
        if ds is None:
            return delta, None
        jac = np.zeros_like(ds)
        response = amplitude[:, None] * (ds[j] - ds[i])
        direct = np.einsum("ed,ed->e", dvalues, w)[:, None] * diff[:, None] * grad
        edges = np.arange(len(i))[:, None]
        axis = np.arange(3)
        response[edges, 3 * i[:, None] + axis] += direct
        response[edges, 3 * j[:, None] + axis] -= direct
        np.add.at(jac, i, response)
        np.add.at(jac, j, -response)
        return delta, jac

    def design_state(self, z, r, q, m, forces=False):
        out = np.zeros((GROUPS, DIM))
        jac = np.zeros((GROUPS, DIM, r.size)) if forces else None
        if m == 2:
            types, x, dx = self.sites(z, r, forces)
            s, ds = self.population(types, x, q, m, dx)
            delta, dd = self.redistribution(z, r, types, s, q, ds)
            np.add.at(out, types, delta[:, None] * x)
            if forces:
                np.add.at(
                    jac,
                    types,
                    -(delta[:, None, None] * dx + x[:, :, None] * dd[:, None, :]),
                )
        return out.ravel(), jac.reshape(self.size, r.size).T if forces else None

    def fields(self):
        return {**super().fields(), "spinexchange_weights": self.exchange}

    @classmethod
    def load(cls, data):
        return cls(
            *(
                data["spinlocal_" + k]
                for k in ("groups", "radii", "logits", "polarization", "weights")
            ),
            data["spinexchange_weights"],
        )
