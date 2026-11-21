"""An atom-centered positive-semidefinite kernel with trainable Fourier frequencies.

The feature map uses sine/cosine pairs of projected local density coordinates.
Its frequencies control nonlinear chemical similarity and its linear readout
sets the energy. Subtracting the isolated cosine value makes each isolated
atom's contribution zero. Cartesian derivatives include the density response.
"""

import numpy as np
from .local import descriptor_array


class SpectralKernel:
    def __init__(self, species, spec, projections, frequencies, length, weights):
        self.species = tuple(map(int, species))
        self.spec = spec
        self.projections = projections
        self.frequencies = frequencies
        self.length = float(length)
        self.weights = np.asarray(weights)
        self.size = sum(2 * w.shape[1] for w in frequencies.values())
        if self.weights.shape != (self.size,) or self.length <= 0:
            raise ValueError("Invalid learned spectrum")

    def design(self, z, r, forces=False):
        keep = np.isin(z, self.species)
        zz, rr = z[keep], r[keep]
        out = np.zeros(self.size)
        jac = np.zeros((len(z), 3, self.size)) if forces else None
        off = 0
        for s, w in self.frequencies.items():
            n = w.shape[1]
            sl = slice(off, off + 2 * n)
            off = sl.stop
            proj = self.projections[s] @ w / self.length
            for atom in np.flatnonzero(zz == s):
                raw, d = descriptor_array(rr, zz, atom, **self.spec, gradient=forces)
                phase = raw @ proj
                c, sin = np.cos(phase), np.sin(phase)
                out[sl] += np.r_[c - 1.0, sin]
                if forces:
                    dphase = proj.T @ d.reshape(len(raw), -1)
                    negative = np.vstack(
                        (sin[:, None] * dphase, -c[:, None] * dphase)
                    ).T
                    jac[keep, :, sl] += negative.reshape(len(rr), 3, 2 * n)
        return out, jac.reshape(r.size, self.size) if forces else None

    def fields(self):
        d = dict(
            spectral_species=self.species,
            spectral_keys=list(self.frequencies),
            spectral_length=self.length,
            spectral_weights=self.weights,
            spectral_spec=[self.spec[k] for k in ("nr", "lmax", "ron", "rc")],
        )
        for k, w in self.frequencies.items():
            d[f"spectral_frequency_{k}"] = w
            d[f"spectral_projection_{k}"] = self.projections[k]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["spectral_keys"]))
        n, l, a, b = d["spectral_spec"]
        species = tuple(map(int, d["spectral_species"]))
        spec = dict(
            nr=int(n), lmax=int(l), ron=a, rc=b, inverse_radius=True, species=species
        )
        return cls(
            species,
            spec,
            {s: d[f"spectral_projection_{s}"] for s in keys},
            {s: d[f"spectral_frequency_{s}"] for s in keys},
            d["spectral_length"],
            d["spectral_weights"],
        )
