"""A sparse local Matérn 5/2 energy model with analytic, conservative forces.

sGDML uses mixed derivatives of a molecular kernel on inverse distances. Here
the same kernel calculus acts on *atomic* radial and angular features so that
an environment can contribute to different molecules. Equivalent neighbors
are summed in the descriptor. Each center has an ordinary energy basis and a
directional center derivative, permitting force-like localized corrections.
"""
import numpy as np

from .electronic import state_key
from .local import descriptor_array, descriptor_layout


class LocalMatern:
    def __init__(self, species, projections, centers, tangents, radial_dims, length, angular, weights=None, family="matern52"):
        self.species = tuple(species)
        self.projections, self.centers, self.tangents = projections, centers, tangents
        self.radial_dims = radial_dims
        if family not in ("matern52", "double_matern52", "gaussian"):
            raise ValueError(f"Unknown local kernel {family}")
        self.family = family
        self.length, self.angular = float(length), float(angular)
        if min(self.length, self.angular) <= 0 or not np.isfinite([length,angular]).all():
            raise ValueError("Kernel scales must be positive and finite")
        self.spec = dict(species=self.species, nr=2, lmax=2, ron=1.9, rc=3.2, inverse_radius=True)
        self.slices = {}
        self.size = 0
        for s in self.species:
            c, p, t = centers[s], projections[s], tangents[s]
            if (c.ndim != 2 or c.shape != t.shape or p.shape[1] != c.shape[1]
                    or not 0 <= radial_dims[s] <= c.shape[1]
                    or p.shape[0] != sum(descriptor_layout(len(species), 2, 2)[0])):
                raise ValueError("Inconsistent local kernel basis")
            if not all(np.isfinite(v).all() for v in (c,p,t)):
                raise ValueError("Nonfinite local kernel basis")
            self.slices[s] = slice(self.size, self.size+2*len(c))
            self.size += 2*len(c)
        self.weights = {} if weights is None else dict(weights)
        if any(np.shape(w)!=(self.size,) or not np.isfinite(w).all() for w in self.weights.values()):
            raise ValueError("Invalid local kernel coefficients")

    def project(self, z, r, *, gradient):
        """Pack local environments by species, with optional Cartesian Jacobians."""
        result = {}
        for s in np.unique(z):
            if not len(self.centers[s]):
                continue
            sites, jac = [], []
            for i in np.flatnonzero(z==s):
                raw, deriv = descriptor_array(r, z, i, **self.spec, gradient=gradient)
                sites.append(raw @ self.projections[s])
                if gradient:
                    jac.append(self.projections[s].T @ deriv.reshape(len(raw), -1))
            result[s] = (np.array(sites), np.array(jac) if gradient else None)
        return result

    def _kernel(self, s, x, derivative):
        """Evaluate k(x,c), its directional center derivative and their x gradients."""
        weight = np.ones(x.shape[1])
        weight[self.radial_dims[s]:] = self.angular
        delta = (x[:, None, :] - self.centers[s]) * weight
        direction = self.tangents[s] * weight
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        if self.family == "gaussian":
            inv = self.length**-2
            k = np.exp(-.5*inv*squared)
            b, curvature = inv*k, inv*inv*k
        else:
            # A positive sum is again a positive-definite kernel. The two widths
            # share a fitted scale so local curvature and broad environment
            # similarity can coexist without independent unconstrained parameters.
            factors = (.5,2.) if self.family=="double_matern52" else (1.,)
            k, b, curvature = (np.zeros_like(squared) for _ in range(3))
            for factor in factors:
                a2 = 5./(factor*self.length)**2
                t = np.sqrt(a2*squared)
                exp = np.exp(-t)/len(factors)
                k += (1+t+t*t/3)*exp
                b += a2/3*(1+t)*exp
                curvature += a2*a2/3*exp
        du = np.einsum("ijk,jk->ij", delta, direction)
        values = np.concatenate((k, self.length*b*du), axis=1)
        if not derivative:
            return values, None
        # These formulas have finite limits at x=c, including the mixed derivative.
        g = -b[:,:,None]*delta
        mixed = (b[:,:,None]*direction - (curvature*du)[:,:,None]*delta)*self.length
        return values, np.concatenate((g, mixed), axis=1)*weight

    def design(self, sites, ncart=0):
        """Sum site basis energies and pull their negative gradients back to nuclei."""
        energy = np.zeros(self.size)
        forces = np.zeros((ncart,self.size)) if ncart else None
        for s,(x,jac) in sites.items():
            cols = self.slices[s]
            values, deriv = self._kernel(s, x, bool(ncart))
            empty, _ = self._kernel(s, np.zeros((1,x.shape[1])), False)
            energy[cols] = values.sum(axis=0) - len(x)*empty[0]
            if ncart:
                forces[:,cols] = -np.einsum("ipd,idr->rp", deriv, jac, optimize=True)
        return energy, forces

    def evaluate(self, z, r, state, *, forces):
        values, df = self.design(self.project(z,r,gradient=forces), r.size if forces else 0)
        w = self.weights[state]
        return float(values@w), (df@w).reshape(r.shape) if forces else None

    def fields(self):
        """Store all nonlinear parameters and atom projections in the model bundle."""
        fields = dict(matern_family=self.family, matern_scales=[self.length,self.angular], matern_states=list(self.weights),
                      matern_weights=list(self.weights.values()))
        for s in self.species:
            for label, values in (("proj",self.projections),("center",self.centers),
                                  ("direction",self.tangents),("radial",self.radial_dims)):
                fields[f"matern_{label}_{s}"] = values[s]
        return fields

    @classmethod
    def load(cls, d):
        species = tuple(d["species"])
        def mapping(key):
            return {s:d[f"matern_{key}_{s}"] for s in species}
        weights = {state_key(*k):w for k,w in zip(d["matern_states"],d["matern_weights"],strict=True)}
        return cls(species,mapping("proj"),mapping("center"),mapping("direction"),
                   {s:int(v) for s,v in mapping("radial").items()},*d["matern_scales"],weights,
                   family=str(d["matern_family"]) if "matern_family" in d else "matern52")
