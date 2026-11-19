"""Predict the Hirshfeld volume ratio required by TS from nuclear geometry.

The target is the dimensionless atom-in-molecule / free-atom volume ratio.
Fitting its logarithm ensures positive ratios after exponentiation. The model
has its own labels and checkpoint; the PBE0 energy fit is independent.
"""
import json

import numpy as np

from ..local import descriptor_array


class HirshfeldVolumes:
    """Linear invariant descriptors per central element, anchored at log ratio zero."""
    def __init__(self, specification, coefficients):
        self.spec = dict(specification)
        self.spec["species"] = tuple(map(int, self.spec["species"]))
        self.coefficients = coefficients

    @classmethod
    def fit(cls, records, specification, ridge=1e-2, scale_floor=0.1):
        """Fit (Z, coordinates in Å, positive hRAT) records from training molecules."""
        if not 0 < ridge < np.inf or not 0 < scale_floor < np.inf:
            raise ValueError("Expected finite positive ridge and scale floor")
        species = tuple(map(int, specification["species"]))
        spec = dict(specification, species=species)
        rows, targets = ({s: [] for s in species} for _ in range(2))
        for z, r, ratios in records:
            z, r, ratios = np.asarray(z), np.asarray(r), np.asarray(ratios)
            if ratios.shape == (len(z), 1):
                ratios = ratios[:, 0]
            if not np.isin(z, species).all() or r.shape != (len(z), 3) or not np.isfinite(r).all():
                raise ValueError("Invalid geometry or unconfigured species for volume fit")
            if ratios.shape != (len(z),) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
                raise ValueError("Expected one finite positive Hirshfeld ratio per atom")
            for i, s in enumerate(z):
                rows[s].append(descriptor_array(r, z, i, **spec)[0])
                targets[s].append(np.log(ratios[i]))
        coefficients = {}
        for s in species:
            if not rows[s]:
                raise ValueError(f"No training environments for species {s}")
            design = np.stack(rows[s])
            scale = np.maximum(design.std(axis=0), scale_floor)
            n = len(scale)
            augmented = np.vstack((design/scale, np.sqrt(ridge)*np.eye(n)))
            fit = np.linalg.lstsq(augmented, np.r_[targets[s], np.zeros(n)], rcond=None)[0]
            coefficients[s] = fit/scale
        return cls(spec, coefficients)

    def evaluate(self, z, r):
        """Return log ratios (N,) and d log(v_i)/d R_j (N,N,3) in inverse Å.

        The force chain rule needs the response to *all* atoms in a local
        environment, even when a given atom is outside a particular TS pair.
        """
        z, r = np.asarray(z), np.asarray(r, dtype=float)
        if (z.ndim != 1 or r.shape != (len(z), 3) or not np.isfinite(r).all()
                or not np.isin(z, self.spec["species"]).all()):
            raise ValueError("Invalid geometry or unconfigured species for volume prediction")
        values, jac = np.empty(len(z)), np.empty((len(z), len(z), 3))
        for i, s in enumerate(z):
            raw, derivative = descriptor_array(r, z, i, gradient=True, **self.spec)
            values[i] = raw @ self.coefficients[s]
            jac[i] = np.einsum("f,fjk->jk", self.coefficients[s], derivative)
        return values, jac

    def save(self, path):
        np.savez(path, format="hirshfeld_log_volume/1", spec=json.dumps(self.spec),
                 coefficients=np.stack([self.coefficients[s] for s in self.spec["species"]]))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"]) != "hirshfeld_log_volume/1":
                raise ValueError("Incompatible volume checkpoint")
            spec = json.loads(str(data["spec"]))
            return cls(spec, dict(zip(spec["species"], data["coefficients"], strict=True)))
