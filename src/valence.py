"""Short-range rotationally invariant energy correction for equilibrium structure and curvature.

The features form a finite polynomial kernel on bond lengths and bond-angle
cosines. Neighbour weights go continuously to zero during dissociation. The
same fitted coefficients enter the energy, gradient and vibrational Hessian.
"""
from itertools import combinations, combinations_with_replacement

import numpy as np


class Valence:
    """Small shared basis for neutral first-row hydrides and their organic analogues.

    Angular coefficients are shared across different substituents at one center.
    A separate linear term accounts for substitution by heavier neighbors. For
    tricoordinate nitrogen, the square of the sum of angular displacements
    distinguishes collective umbrella motion from the other bends.
    """
    radii = {1: .31, 6: .76, 7: .71, 8: .66, 9: .57}
    pairs = tuple(combinations_with_replacement(radii, 2))
    centers = (6, 7, 8)
    size = 2*len(pairs) + 4*len(centers)

    def __init__(self, weights=None):
        self.weights = np.zeros(self.size) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != (self.size,) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid valence coefficients")

    def design(self, z, r):
        """Return features and their Cartesian gradients (positive energy derivatives)."""
        n = len(z)
        features = np.zeros(self.size)
        grad = np.zeros((n, 3, self.size))
        neighbors = [[] for _ in z]
        for i, j in combinations(range(n), 2):
            a, b = sorted((z[i], z[j]))
            if a not in self.radii or b not in self.radii:
                continue
            scale = self.radii[a] + self.radii[b]
            delta = r[j] - r[i]
            length = np.linalg.norm(delta)
            if length >= 1.50*scale:
                continue
            if length < 1e-10:
                raise ValueError("Coincident nuclei")
            direction = delta/length
            # Quintic switching gives continuous forces and Hessians at both ends.
            t = np.clip((length/scale - 1.15)/.35, 0., 1.)
            w = 1 - t**3*(10 - 15*t + 6*t*t)
            dw = -30*t*t*(1-t)**2/(.35*scale)
            neighbors[i].append((j, direction, length, w, dw))
            neighbors[j].append((i, -direction, length, w, dw))
            index = 2*self.pairs.index((a, b))
            x = length - scale
            powers = np.array([x, .5*x*x])
            features[index:index+2] += w*powers
            deriv = np.outer(direction, dw*powers + w*np.array([1., x]))
            grad[j, :, index:index+2] += deriv
            grad[i, :, index:index+2] -= deriv

        for i, site in enumerate(neighbors):
            if z[i] not in self.centers or len(site) < 2:
                continue
            index = 2*len(self.pairs) + 4*self.centers.index(z[i])
            collective = 0.
            collective_grad = np.zeros((n, 3))
            for (j, u, ru, wu, dwu), (k, v, rv, wv, dwv) in combinations(site, 2):
                cosine = u @ v
                x = cosine + 1/3
                heavy_fraction = (int(z[j] != 1) + int(z[k] != 1)) / 2
                powers = np.array([x, .5*x*x, heavy_fraction*x])
                slopes = np.array([1., x, heavy_fraction])
                w = wu*wv
                features[index:index+3] += w*powers
                # The cosine derivatives are transverse to their respective bonds.
                # The weight derivatives supply the radial part in the switching region.
                gj = np.outer(u, dwu*wv*powers) + np.outer((v-cosine*u)/ru, w*slopes)
                gk = np.outer(v, dwv*wu*powers) + np.outer((u-cosine*v)/rv, w*slopes)
                grad[j, :, index:index+3] += gj
                grad[k, :, index:index+3] += gk
                grad[i, :, index:index+3] -= gj + gk
                collective += w*x
                collective_grad[j] += gj[:, 0]
                collective_grad[k] += gk[:, 0]
                collective_grad[i] -= gj[:, 0] + gk[:, 0]
            # Use the square also for two neighbors. It reduces there to a
            # single-angle feature and stays smooth if a third bond dissociates.
            features[index+3] += .5*collective**2
            grad[:, :, index+3] += collective*collective_grad
        return features, grad.reshape(r.size, self.size)

    def predict(self, z, r):
        """Contract the energy basis and its negative gradient with the fitted weights."""
        e, g = self.design(z, r)
        return float(e @ self.weights), -(g @ self.weights).reshape(r.shape)
