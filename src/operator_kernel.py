"""Finite eigenfunction kernels for smooth, additive molecular energies.

Each local environment maps to a vector of real mode amplitudes. Its inner
product with another environment is a positive-semidefinite kernel. The
molecular feature is the sum of local vectors after subtracting the isolated
atom. Input coordinates are the same projected radial/angular invariants used
by ReactiveBasis. The saved mode amplitudes include their spectral weighting.
"""

import numpy as np
from numpy.polynomial import chebyshev, polynomial

from .reactive_basis import ReactiveBasis


class OperatorKernel(ReactiveBasis):
    def __init__(
        self, species, spec, projections, centers, params, family, coeff, weights
    ):
        self.species = tuple(map(int, species))
        self.spec = dict(
            nr=int(spec[0]),
            lmax=int(spec[1]),
            ron=float(spec[2]),
            rc=float(spec[3]),
            inverse_radius=True,
            species=self.species,
        )
        self.projections, self.centers, self.params = projections, centers, params
        self.family, self.coeff = family, np.asarray(coeff)
        self.four, self.bond = False, None
        self.widths = {s: len(p["normalizer"]) for s, p in params.items()}
        self.size = sum(self.widths.values())
        self.weights = np.asarray(weights, float)
        if self.weights.shape != (self.size,):
            raise ValueError("Incompatible operator kernel weights")

    def _scalar(self, u, grad):
        """Evaluate five modes of the selected 1-D operator and their derivatives."""
        fam = self.family
        if fam == "hermite":
            x, dx = u, 1.0
            env = np.exp(-0.5 * x * x)
            dlog = -x
        elif fam == "laguerre":
            x = np.logaddexp(0.0, u)
            dx = np.exp(-np.logaddexp(0.0, -u))
            env, dlog = np.exp(-0.5 * x), -0.5
        else:
            x = np.tanh(0.5 * u)
            dx, env, dlog = 0.5 * (1 - x * x), 1.0, 0.0
        if fam in ("dirichlet_sine", "neumann_cosine"):
            phase = (x[..., None] + 1) * self.coeff / 2
            sine, cosine = np.sin(phase), np.cos(phase)
            val, deriv = (sine, cosine) if fam == "dirichlet_sine" else (cosine, -sine)
            return val, (deriv * (dx[..., None] * self.coeff / 2) if grad else None)
        if fam in ("krawtchouk", "hahn"):
            # The finite-chain eigenvectors are extended by averaging at a
            # binomially distributed index. Their Bernstein representations
            # are stored as equivalent Chebyshev coefficients for stability.
            use_cheb = True
        else:
            use_cheb = fam == "bessel_disk"
        module = chebyshev if use_cheb else polynomial
        evaluate = module.chebval if use_cheb else module.polyval
        differentiate = module.chebder if use_cheb else module.polyder
        poly = evaluate(x, self.coeff.T).transpose(1, 2, 0)
        v = poly * np.asarray(env)[..., None]
        if not grad:
            return v, None
        d = evaluate(x, differentiate(self.coeff.T)).transpose(1, 2, 0)
        return (
            v,
            (d + np.asarray(dlog)[..., None] * poly) * np.asarray(env * dx)[..., None],
        )

    def raw_modes(self, s, x, grad=False):
        p = self.params[s]
        if self.family == "graph_diffusion":
            c, width, right = self.centers[s], p["scale"][0], p["modes"]
            # Common terms in the Gaussian log affinity cancel from softmax.
            logits = (x @ c.T - np.sum(c * c, axis=1) / 2) / width**2
            w = np.exp(logits - logits.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            v = w @ right
            if not grad:
                return v, None
            j = (
                np.einsum("nc,cm,cd->nmd", w, right, c, optimize=True)
                - v[:, :, None] * (w @ c)[:, None, :]
            ) / width**2
            return v, j
        directions = p["directions"]
        u = (x @ directions - p["mean"]) / p["scale"]
        v, d = self._scalar(u, grad)
        j = None
        if grad:
            j = (
                d[:, :, :, None]
                * (directions.T / p["scale"][:, None])[None, :, None, :]
            ).reshape(len(x), -1, x.shape[1])
        flat = v.reshape(len(x), -1)
        if self.family == "tensor_legendre":
            # Retain every univariate mode. Add all distinct products of the
            # degree-one modes, so cross-coordinate curvature is independent
            # of the coefficients of the single-coordinate basis.
            left, right = np.triu_indices(v.shape[1], 1)
            linear = v[:, :, 0]
            if grad:
                linear_j = j[:, :: v.shape[2], :]
                mixed_j = (
                    linear_j[:, left] * linear[:, right, None]
                    + linear_j[:, right] * linear[:, left, None]
                )
                j = np.concatenate((j, mixed_j), axis=1)
            flat = np.column_stack((flat, linear[:, left] * linear[:, right]))
        return flat, j

    def values(self, s, sites, grad=False):
        v, j = self.raw_modes(s, sites, grad)
        p = self.params[s]
        return (v - p["zero"]) * p["normalizer"], (
            j * p["normalizer"][None, :, None] if grad else None
        )

    def site_values(self, s, sites):
        return self.values(s, sites)[0]

    def fields(self):
        d = dict(
            operator_family=self.family,
            operator_coeff=self.coeff,
            operator_weights=self.weights,
            operator_keys=list(self.centers),
            operator_species=self.species,
            operator_spec=[self.spec[k] for k in ("nr", "lmax", "ron", "rc")],
        )
        for s in self.centers:
            d[f"proj_{s}"] = self.projections[s]
            d[f"centers_{s}"] = self.centers[s]
            for key, v in self.params[s].items():
                d[f"param_{s}_{key}"] = v
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["operator_keys"]))
        params = {
            s: {
                k[len(f"param_{s}_") :]: v
                for k, v in d.items()
                if k.startswith(f"param_{s}_")
            }
            for s in keys
        }
        return cls(
            d["operator_species"],
            d["operator_spec"],
            {s: d[f"proj_{s}"] for s in keys},
            {s: d[f"centers_{s}"] for s in keys},
            params,
            str(d["operator_family"]),
            d["operator_coeff"],
            d["operator_weights"],
        )
