"""Twelve smooth four-atom invariants and analytic Cartesian force features.

A determinant measures squared out-of-plane volume around a central atom.
A second contraction measures relative orientation of neighbors on opposite
sides of a bond. Both remain smooth as bonds form and break, and neither
requires assigning a graph, dihedral angle or atom correspondence.
"""

import numpy as np

RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02}
CENTRAL = (6, 7, 8, 16)
PAIRS = {(6, 6): 8, (6, 7): 9, (6, 8): 10, (8, 8): 11}


def switch(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return 1 + t**4 * (-35 + t * (84 + t * (-70 + 20 * t))), -140 * t**3 * (
        1 - t
    ) ** 3 / (hi - lo)


def four_body(z, r, forces=False):
    """Return additive energy features and minus their Cartesian Jacobian."""
    keep = np.flatnonzero(np.isin(z, tuple(RADII)))
    v = np.zeros(12)
    full = np.zeros((len(z), 3, 12)) if forces else None
    if len(keep) < 4:
        return v, full.reshape(r.size, 12) if forces else None
    zz, rr = z[keep], r[keep]
    rad = np.array([RADII[int(a)] for a in zz])
    n = len(zz)
    delta = rr[None, :, :] - rr[:, None, :]
    d = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(d, 1.0)
    u = delta / d[:, :, None]
    scale = rad[:, None] + rad[None, :]
    x = d / scale
    grad = np.zeros((n, 3, 12)) if forces else None
    for shell, (lo, hi) in enumerate(((0.95, 1.4), (1.1, 1.85))):
        w, dw = switch(x, lo, hi)
        np.fill_diagonal(w, 0.0)
        dw /= scale
        for typ, s in enumerate(CENTRAL):
            c = 2 * typ + shell
            for i in np.flatnonzero(zz == s):
                js = np.flatnonzero(w[i] > 0)
                if len(js) < 3:
                    continue
                unit = u[i, js]
                weights = w[i, js]
                mat = unit.T @ (weights[:, None] * unit)
                # Cofactors stay finite at a planar geometry, unlike inv(mat).
                cof = np.cross(mat[[1, 2, 0]], mat[[2, 0, 1]])
                v[c] += mat[0] @ cof[0]
                if forces:
                    cu = unit @ cof
                    h = (cu * unit).sum(axis=1)
                    dv = (dw[i, js] * h)[:, None] * unit + (2 * weights / d[i, js])[
                        :, None
                    ] * (cu - h[:, None] * unit)
                    grad[js, :, c] += dv
                    grad[i, :, c] -= dv.sum(axis=0)
    w, dw = switch(x, 1.0, 1.65)
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(dw, 0.0)
    dw /= scale
    species_weight = np.where(zz == 1, 1.0, 1.7)
    for i, j in zip(*np.nonzero(np.triu(w > 0, 1))):
        c = PAIRS.get(tuple(sorted((zz[i], zz[j]))))
        if c is None:
            continue
        wi = w[i] * species_weight
        wj = w[j] * species_weight
        wi[j] = wj[i] = 0.0
        if not wi.any() or not wj.any():
            continue
        a = wi[:, None] * u[i]
        b = wj[:, None] * u[j]
        e = u[i, j]
        P = np.eye(3) - np.outer(e, e)
        A = a.sum(axis=0)
        B = b.sum(axis=0)
        # Subtract shared-neighbor terms so every contribution has four distinct atoms.
        t = A @ P @ B - np.sum((a @ P) * b)
        v[c] += w[i, j] * t
        if not forces:
            continue
        for origin, other, ww, adj in [
            (i, j, wi, (B - b) @ P),
            (j, i, wj, (A - a) @ P),
        ]:
            js = np.flatnonzero(ww)
            unit = u[origin, js]
            k = adj[js]
            h = (k * unit).sum(axis=1)
            dv = w[i, j] * (
                (dw[origin, js] * species_weight[js] * h)[:, None] * unit
                + (ww[js] / d[origin, js])[:, None] * (k - h[:, None] * unit)
            )
            grad[js, :, c] += dv
            grad[origin, :, c] -= dv.sum(axis=0)
        de = -(A @ e) * B - (B @ e) * A + (a @ e) @ b + (b @ e) @ a
        dv = dw[i, j] * t * e + w[i, j] * (P @ de) / d[i, j]
        grad[j, :, c] += dv
        grad[i, :, c] -= dv
    if forces:
        full[keep] = -grad
    return v, full.reshape(r.size, 12) if forces else None
