"""Smooth atom-local mixing of radical reaction-family energy contributions.

Each environment supplies probabilities for different radical mechanisms.
The derivative of those probabilities contributes to the force. Family
labels are used during training. Inference requires only the molecular state
and one set of coordinates.
"""

import numpy as np
from scipy.special import softmax

from .local import descriptor_array
from .periodic_kernel import PeriodicKernel


class PathFamilyKernel:
    def __init__(self, banks, native, logits, weights):
        self.banks = banks
        self.native = native
        self.logits = logits
        self.weights = np.asarray(weights, float)
        self.size = len(weights)
        k = next(iter(native.values())).shape[1]
        if self.size != 2 * k - 1:
            raise ValueError(
                "Inconsistent shared, family, and locally weighted energy columns"
            )
        for key, w in native.items():
            if w.shape != (sum(len(b.centers[key]) for b in banks), k) or logits[
                key
            ].shape != (len(w) + 1, k):
                raise ValueError("Inconsistent local family energies or logits")

    def atomic(self, key, sites, gradient=False):
        val = [b.values(sites, b.centers[key], gradient) for b in self.banks]
        x = np.column_stack([a for a, _ in val])
        dx = np.concatenate([j for _, j in val], axis=1) if gradient else None
        return x, dx

    def features(self, key, x, dx=None):
        """Differentiate both the constituent energies and their softmax weights."""
        w = self.native[key]
        log = self.logits[key]
        energy = x @ w
        p = softmax(x @ log[:-1] + log[-1], axis=1)
        values = np.column_stack([energy, p[:, 1:] * energy[:, 1:]])
        if dx is None:
            return values, None
        de = np.einsum("bf,nbd->nfd", w, dx)
        dl = np.einsum("bf,nbd->nfd", log[:-1], dx)
        dp = p[:, :, None] * (dl - np.einsum("nf,nfd->nd", p, dl)[:, None, :])
        gated = dp[:, 1:] * energy[:, 1:, None] + p[:, 1:, None] * de[:, 1:]
        return values, np.concatenate([de, gated], axis=1)

    def design_state(self, z, r, q, m, forces=False):
        b = self.banks[0]
        out = np.zeros(self.size)
        jac = np.zeros((r.size, self.size)) if forces else None
        if m != 2:
            return out, jac
        types = b.groups[z]
        for i, key in enumerate(types):
            if key not in self.native:
                continue
            desc, j = descriptor_array(
                r,
                types,
                i,
                nr=b.shells.shape[1],
                lmax=2,
                ron=b.cutoff[0],
                rc=b.cutoff[1],
                species=tuple(range(9)),
                gaussian_shells=b.shells,
                atom_radii=b.radii[z],
                gradient=forces,
            )
            proj = b.projections[key]
            x, dx = self.atomic(key, (desc @ proj)[None, :], forces)
            val, dv = self.features(key, x, dx)
            out += val[0]
            if forces:
                jac -= (dv[0] @ (proj.T @ j.reshape(len(desc), -1))).T
        return out, jac

    def fields(self):
        out = {"path_family_weights": self.weights, "banks": len(self.banks)}
        for i, b in enumerate(self.banks):
            out.update({f"bank{i}_{k}": v for k, v in b.fields().items()})
        for key in self.native:
            out[f"native_{key}"] = self.native[key]
            out[f"logits_{key}"] = self.logits[key]
        return out

    @classmethod
    def load(cls, f):
        banks = []
        for i in range(int(f["banks"])):
            prefix = f"bank{i}_"
            banks.append(
                PeriodicKernel.load(
                    {k[len(prefix) :]: v for k, v in f.items() if k.startswith(prefix)}
                )
            )
        keys = banks[0].centers
        return cls(
            banks,
            {k: f[f"native_{k}"] for k in keys},
            {k: f[f"logits_{k}"] for k in keys},
            f["path_family_weights"],
        )
