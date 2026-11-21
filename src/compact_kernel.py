"""A sparse local density kernel with chemistry-dependent radii of support.

Each prototype overlaps a bounded neighborhood in projected density space.
The fourth power window reaches zero with three continuous derivatives. A
search tree returns every overlapping center; crossing an arbitrary nearest-
neighbor rank does not change the energy. The explicit feature inner product
defines a positive-semidefinite molecular kernel.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from .local import descriptor_array


class CompactKernel:
    def __init__(self, species, spec, projections, centers, radii, weights):
        self.species = tuple(map(int, species))
        self.spec = spec
        self.projections = projections
        self.centers = centers
        self.radii = radii
        self.weights = np.asarray(weights)
        self.size = sum(map(len, centers.values()))
        if self.weights.shape != (self.size,):
            raise ValueError("Invalid sparse readout dimensions")
        self.trees = {}
        for s, c in centers.items():
            radius = radii[s]
            if np.any(radius <= 0) or len(radius) != len(c):
                raise ValueError("Invalid prototype support")
            bins = np.searchsorted([0.2, 0.35, 0.55, 0.85, 1.25], radius)
            self.trees[s] = []
            for k in np.unique(bins):
                ix = np.flatnonzero(bins == k)
                self.trees[s].append((ix, radius[ix].max(), cKDTree(c[ix])))

    def overlaps(self, s, x):
        """Yield site/center pairs inside the individually fitted supports."""
        for ids, bound, tree in self.trees[s]:
            hits = tree.query_ball_point(x, bound, return_sorted=False)
            counts = np.fromiter((len(h) for h in hits), dtype=int, count=len(x))
            if not counts.sum():
                continue
            row = np.repeat(np.arange(len(x)), counts)
            col = ids[np.concatenate([v for v in hits if len(v)]).astype(int)]
            diff = x[row] - self.centers[s][col]
            inv = 1.0 / self.radii[s][col] ** 2
            t = 1.0 - np.einsum("ij,ij->i", diff, diff) * inv
            use = t > 0
            if use.any():
                yield row[use], col[use], diff[use], inv[use], t[use]

    def sparse(self, s, x, owner, nmol):
        """Assemble the summed site features for a batch without a dense design."""
        rows = []
        cols = []
        vals = []
        for i, j, _, _, t in self.overlaps(s, x):
            rows.append(owner[i])
            cols.append(j)
            vals.append(t**4)
        # Energy zero at separated isolated atoms. Store only nonzero corrections.
        count = np.bincount(owner, minlength=nmol)
        present = np.flatnonzero(count)
        for _, j, _, _, t in self.overlaps(s, np.zeros((1, x.shape[1]))):
            rows.append(np.repeat(present, len(j)))
            cols.append(np.tile(j, len(present)))
            vals.append((-count[present, None] * t**4).ravel())
        if not rows:
            return csr_matrix((nmol, len(self.centers[s])))
        return csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nmol, len(self.centers[s])),
        )

    def design(self, z, r, forces=False):
        keep = np.isin(z, self.species)
        zz, rr = z[keep], r[keep]
        values = np.zeros(self.size)
        jac = np.zeros((len(z), 3, self.size)) if forces else None
        offset = 0
        for s, c in self.centers.items():
            sl = slice(offset, offset + len(c))
            offset = sl.stop
            take = np.flatnonzero(zz == s)
            proj = self.projections[s]
            if not len(take):
                continue
            sites = []
            der = []
            for i in take:
                raw, j = descriptor_array(rr, zz, i, **self.spec, gradient=forces)
                sites.append(raw @ proj)
                if forces:
                    der.append(proj.T @ j.reshape(len(raw), -1))
            sites = np.asarray(sites)
            values[sl] = self.sparse(s, sites, np.zeros(len(take), int), 1).toarray()[0]
            if forces:
                local = np.zeros((len(c), rr.size))
                d = np.asarray(der)
                for i, k, v, inv, t in self.overlaps(s, sites):
                    grad = np.einsum("ij,ijd->id", v, d[i]) * (8 * inv * t**3)[:, None]
                    np.add.at(local, k, grad)
                jac[keep, :, sl] = local.T.reshape(len(zz), 3, len(c))
        return values, jac.reshape(r.size, self.size) if forces else None

    def fields(self):
        d = dict(
            compact_species=self.species,
            compact_keys=list(self.centers),
            compact_weights=self.weights,
            compact_spec=[self.spec[n] for n in ("nr", "lmax", "ron", "rc")],
        )
        for s, c in self.centers.items():
            d[f"compact_center_{s}"] = c
            d[f"compact_proj_{s}"] = self.projections[s]
            d[f"compact_support_{s}"] = self.radii[s]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["compact_keys"]))
        n, l, a, b = d["compact_spec"]
        species = tuple(map(int, d["compact_species"]))
        spec = dict(
            nr=int(n), lmax=int(l), ron=a, rc=b, inverse_radius=True, species=species
        )
        return cls(
            species,
            spec,
            {k: d[f"compact_proj_{k}"] for k in keys},
            {k: d[f"compact_center_{k}"] for k in keys},
            {k: d[f"compact_support_{k}"] for k in keys},
            d["compact_weights"],
        )
