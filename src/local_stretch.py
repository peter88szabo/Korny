"""Environment-dependent anharmonic stretching corrections around covalent X-H bonds.

For each H-C, H-N or H-O bond, a smooth interpolation of training environments
estimates the equilibrium length. Each energy feature is quadratic near that
length and goes smoothly to zero far from it. This lets a force constant change
with little shift in the equilibrium energy. Both the length predictor and the
environment interpolation contribute to the conservative Cartesian forces.
"""

import numpy as np
from .bond_order_kernel import environments
from .bond_environment import PAIRS
from .four_body import RADII

FEATURE_SCALE = np.array([1.0] * 8 + [1.6] * 8 + [0.36] * 8)


class StretchKernel:
    def __init__(self, centers, lengths, width, weights):
        self.centers = centers
        self.lengths = lengths
        self.width = float(width)
        self.weights = np.asarray(weights)
        self.size = sum(map(len, centers.values()))
        if (
            self.width <= 0
            or self.weights.shape != (self.size,)
            or any(len(lengths[k]) != len(v) for k, v in centers.items())
        ):
            raise ValueError("Invalid environment-dependent stretch coefficients")

    def interpolation(self, k, context):
        delta = (context[:, None, :] - self.centers[k]) * FEATURE_SCALE
        log_slope = -delta * FEATURE_SCALE / 0.80**2
        g = np.exp(-0.5 * np.einsum("bcd,bcd->bc", delta, delta) / 0.80**2)
        denom = 0.002 + g.sum(axis=1)
        prob = g / denom[:, None]
        # The fallback has constant small weight; in an unfamiliar environment
        # its effect is to turn the correction off continuously.
        length = (g @ self.lengths[k] + 0.002 * np.median(self.lengths[k])) / denom
        return prob, length, log_slope

    def design(self, z, r, forces=False):
        val = np.zeros(self.size)
        jac = np.zeros((len(z), 3, self.size)) if forces else None
        offsets = {}
        off = 0
        for k, c in self.centers.items():
            offsets[k] = slice(off, off + len(c))
            off += len(c)
        for k, x, gate, dx, dgate, edge, geometry in environments(z, r, forces):
            if k not in offsets:
                continue
            scale = sum(RADII[s] for s in PAIRS[k])
            prob, r0, logs = self.interpolation(k, x[:, 1:])
            displacement = (x[:, 0] + 1.0) * scale - r0
            damp = np.exp(-0.5 * (displacement / self.width) ** 2)
            profile = 0.5 * displacement**2 * damp
            sl = offsets[k]
            val[sl] = (gate * profile) @ prob
            if forces:
                dlength = np.einsum(
                    "bc,bcd->bd", prob * (self.lengths[k] - r0[:, None]), logs
                )
                mean = np.einsum("bc,bcd->bd", prob, logs)
                dprob = prob[:, :, None] * (logs - mean[:, None, :])
                slope = (
                    displacement * damp * (1.0 - 0.5 * (displacement / self.width) ** 2)
                )
                local = np.empty((len(x), len(self.centers[k]), x.shape[1]))
                local[:, :, 0] = slope[:, None] * prob * scale
                local[:, :, 1:] = (
                    profile[:, None, None] * dprob
                    - (slope[:, None] * prob)[:, :, None] * dlength[:, None, :]
                )
                derivatives = np.einsum("b,bcd,bde->ec", gate, local, dx, optimize=True)
                np.add.at(derivatives, edge, (dgate * profile)[:, None] * prob)
                a, b, unit = geometry
                f = derivatives[:, None, :] * unit[:, :, None]
                np.add.at(jac[:, :, sl], a, f)
                np.add.at(jac[:, :, sl], b, -f)
        return val, jac.reshape(r.size, self.size) if forces else None

    def fields(self):
        result = dict(
            stretch_keys=list(self.centers),
            stretch_width=self.width,
            stretch_weights=self.weights,
        )
        for k, c in self.centers.items():
            result[f"stretch_centers_{k}"] = c
            result[f"stretch_lengths_{k}"] = self.lengths[k]
        return result

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["stretch_keys"]))
        return cls(
            {k: d[f"stretch_centers_{k}"] for k in keys},
            {k: d[f"stretch_lengths_{k}"] for k in keys},
            d["stretch_width"],
            d["stretch_weights"],
        )
