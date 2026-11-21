"""Atom-centered Gaussian derivative kernels aligned with observed reaction paths.

Each center stores a local environment, a unit tangent in descriptor space,
and a length scale. First and second directional Gaussian derivatives let the
energy vary asymmetrically across a bond rearrangement and change its midpoint
height independently. Only the training step requires paths or Hessians.
"""

import numpy as np

from .reactive_basis import ReactiveBasis


class PathHermiteKernel(ReactiveBasis):
    def values(self, s, sites, grad=False):
        """Evaluate odd/even local features and their descriptor gradients."""
        packed = self.centers[s]
        width = sites.shape[1]
        center = packed[:, :width]
        tangent = packed[:, width : 2 * width]
        sigma = packed[:, -2]
        even = packed[:, -1].astype(bool)
        # Pairwise products avoid materializing an atoms × centers × width
        # tensor when the fit needs energies only.
        squared = (
            (sites * sites).sum(1)[:, None]
            + (center * center).sum(1)
            - 2 * (sites @ center.T)
        )
        gaussian = np.exp(-0.5 * np.maximum(squared, 0.0) / sigma**2)
        along = (sites @ tangent.T - (center * tangent).sum(1)) / sigma
        poly = np.where(even, along**2 - 1.0, along)
        values = gaussian * poly
        jac = None
        if grad:
            dpoly = np.where(even, 2 * along, 1.0)
            jac = (gaussian * dpoly / sigma)[:, :, None] * tangent
            jac -= (values / sigma**2)[:, :, None] * (sites[:, None, :] - center)
        origin = -(center * tangent).sum(1) / sigma
        isolated = np.exp(-0.5 * (center * center).sum(1) / sigma**2) * np.where(
            even, origin**2 - 1.0, origin
        )
        return values - isolated, jac

    def fields(self):
        return {**super().fields(), "path_hermite": 1}
