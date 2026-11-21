"""Three-atom energy kernels for competing bonds during an exchange reaction.

For a central atom and two neighboring atoms, two diabatic curves favor the
opposite bonding arrangements. Their smooth lower eigenvalue and mixing provide
coordinates that resolve barrier tops and partially formed bonds. They are
dimensionless descriptor functions. Energies and forces are learned from the
archived labels. Each triad vanishes smoothly when either bond leaves range.
"""

import numpy as np
from .four_body import switch

WIDTH = 15


def events(z, r, groups, radii, derivatives=False):
    """Enumerate local triads, smooth amplitudes, and optional Cartesian Jacobians."""
    n = len(z)
    a, b = np.triu_indices(n, 1)
    delta = r[b] - r[a]
    dist = np.linalg.norm(delta, axis=1)
    if np.any(dist < 1.0e-10):
        raise ValueError("Coincident atoms in bond competition")
    scale = radii[z[a]] + radii[z[b]]
    use = dist < 1.92 * scale
    a, b, delta, dist, scale = (v[use] for v in (a, b, delta, dist, scale))
    x = dist / scale
    order, dorder = switch(x, 1.03, 1.65)
    gate, dgate = switch(x, 1.40, 1.92)
    # Build the edge list of each central atom without enumerating triples
    # outside the bond cutoff. Edge vectors always point out from the center.
    edge = np.arange(len(a))
    owner = np.r_[a, b]
    neighbor = np.r_[b, a]
    index = np.tile(edge, 2)
    sort = np.argsort(owner, kind="stable")
    degree = np.bincount(owner, minlength=n)
    maximum = int(degree.max(initial=0))
    empty = (np.empty(0, int), np.empty((0, WIDTH)), np.empty(0), None, None)
    if maximum < 2:
        return empty
    table = np.zeros((n, maximum), int)
    offset = np.cumsum(degree) - degree
    table[owner[sort], np.arange(len(sort)) - offset[owner[sort]]] = sort
    left, right = np.triu_indices(maximum, 1)
    i, t = np.nonzero(right[None, :] < degree[:, None])
    d1, d2 = table[i, left[t]], table[i, right[t]]
    typ = groups[z]
    reverse = typ[neighbor[d1]] > typ[neighbor[d2]]
    d1, d2 = np.where(reverse, d2, d1), np.where(reverse, d1, d2)
    same = typ[neighbor[d1]] == typ[neighbor[d2]]
    # Both orientations of a chemically equivalent pair carry half weight.
    # This keeps the descriptor invariant to arbitrary atom order.
    weight = np.r_[np.where(same, 0.5, 1.0), np.full(same.sum(), 0.5)]
    d1, d2 = np.r_[d1, d2[same]], np.r_[d2, d1[same]]
    i = np.r_[i, i[same]]
    j, k = neighbor[d1], neighbor[d2]
    l, h = index[d1], index[d2]
    key = (typ[i] * 9 + typ[j]) * 9 + typ[k]
    u = (r[j] - r[i]) / dist[l, None]
    v = (r[k] - r[i]) / dist[h, None]
    angle = (u * v).sum(axis=1)
    one = np.eye(2)[(z != 1).astype(int)]
    coordination = np.zeros((n, 2))
    np.add.at(coordination, owner, order[index, None] * one[neighbor])
    leftx, rightx = x[l], x[h]
    amp = weight * gate[l] * gate[h]
    feature = [(leftx - 1.0) * 3.0, (rightx - 1.0) * 3.0, angle]

    # A bound Morse coordinate and monotone repulsion define each diabatic
    # energy. Two finite couplings span sharp and broad avoided crossings.
    def diabatic(x):
        ex = np.exp(-2.5 * (x - 1.0))
        rep = 0.48 * np.exp(-3.4 * (x - 1.0))
        return (1 - ex) ** 2, rep, 5 * (1 - ex) * ex, -3.4 * rep

    p, pr, dp, dpr = diabatic(leftx)
    q, qr, dq, dqr = diabatic(rightx)
    mean = (p + qr + q + pr) * 0.5
    gap = (p + qr - q - pr) * 0.5
    partial = [(3.0, 0.0), (0.0, 3.0), (0.0, 0.0)]
    for coupling in (0.07, 0.24):
        root = np.hypot(gap, coupling)
        mix = (coupling / root) ** 2
        feature.extend((mean - root, mix))
        partial.extend(
            (
                (
                    (dp + dpr) * 0.5 - gap / root * (dp - dpr) * 0.5,
                    (dq + dqr) * 0.5 - gap / root * (dqr - dq) * 0.5,
                ),
                (-mix * gap * (dp - dpr) / root**2, -mix * gap * (dqr - dq) / root**2),
            )
        )
    # The gap itself differentiates donors and acceptors of unlike elements.
    feature.extend((gap, 0.5 * (leftx + rightx) - 1.0))
    partial.extend((((dp - dpr) * 0.5, (dqr - dq) * 0.5), (0.5, 0.5)))
    local = [
        coordination[i] - order[l, None] * one[j] - order[h, None] * one[k],
        coordination[j] - order[l, None] * one[i],
        coordination[k] - order[h, None] * one[i],
    ]
    values = np.column_stack([*feature, *local])
    if not derivatives:
        return key, values, amp, None, None
    count = len(i)
    row = np.arange(count)
    dx = np.zeros((len(a), n, 3))
    dx[edge, b] = delta / (dist * scale)[:, None]
    dx[edge, a] = -dx[edge, b]
    dl, dh = dx[l], dx[h]
    dcos = np.zeros((count, n, 3))
    dcos[row, j] = (v - angle[:, None] * u) / dist[l, None]
    dcos[row, k] = (u - angle[:, None] * v) / dist[h, None]
    dcos[row, i] = -dcos[row, j] - dcos[row, k]
    jac = np.empty((count, WIDTH, n, 3))
    for channel, (left_der, right_der) in enumerate(partial):
        jac[:, channel] = (
            np.asarray(left_der)[..., None, None] * dl
            + np.asarray(right_der)[..., None, None] * dh
        )
    jac[:, 2] = dcos
    dcoord = np.zeros((n, 2, n, 3))
    do = dorder[:, None, None] * dx
    np.add.at(dcoord, owner, one[neighbor, :, None, None] * do[index, None])
    jac[:, 9:11] = (
        dcoord[i]
        - one[j, :, None, None] * do[l, None]
        - one[k, :, None, None] * do[h, None]
    )
    jac[:, 11:13] = dcoord[j] - one[i, :, None, None] * do[l, None]
    jac[:, 13:15] = dcoord[k] - one[i, :, None, None] * do[h, None]
    damp = (weight * dgate[l] * gate[h])[:, None, None] * dl + (
        weight * gate[l] * dgate[h]
    )[:, None, None] * dh
    return (
        key,
        values,
        amp,
        jac.reshape(count, WIDTH, r.size),
        damp.reshape(count, r.size),
    )


class CompetitionKernel:
    """Sum element-resolved local exchange kernels into a conservative scalar PES."""

    def __init__(self, groups, radii, centers, projections, length, weights):
        self.groups = np.asarray(groups)
        self.radii = np.asarray(radii)
        self.centers = centers
        self.projections = projections
        self.length = float(length)
        self.weights = np.asarray(weights)
        self.size = sum(map(len, centers.values()))
        if self.size != len(self.weights) or self.length <= 0:
            raise ValueError("Invalid competition kernel")

    def design(self, z, r, forces=False):
        key, val, w, dv, dw = events(z, r, self.groups, self.radii, forces)
        out = np.zeros(self.size)
        df = np.zeros((r.size, self.size)) if forces else None
        offset = 0
        for g, c in self.centers.items():
            sl = slice(offset, offset + len(c))
            offset = sl.stop
            ix = np.flatnonzero(key == g)
            if not len(ix):
                continue
            p = self.projections[g]
            dx = (val[ix] @ p)[:, None, :] - c
            k = np.exp(-0.5 * np.einsum("ikd,ikd->ik", dx, dx) / self.length**2)
            weighted = w[ix, None] * k
            out[sl] = weighted.sum(axis=0)
            if forces:
                df[:, sl] = (
                    np.einsum(
                        "ik,ikd,idv->vk", weighted, dx, p.T @ dv[ix], optimize=True
                    )
                    / self.length**2
                    - dw[ix].T @ k
                )
        return out, df

    def fields(self):
        d = dict(
            competition_groups=self.groups,
            competition_radii=self.radii,
            competition_length=self.length,
            competition_keys=list(self.centers),
            competition_weights=self.weights,
        )
        for k, c in self.centers.items():
            d[f"competition_center_{k}"] = c
            d[f"competition_projection_{k}"] = self.projections[k]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["competition_keys"]))
        return cls(
            d["competition_groups"],
            d["competition_radii"],
            {k: d[f"competition_center_{k}"] for k in keys},
            {k: d[f"competition_projection_{k}"] for k in keys},
            d["competition_length"],
            d["competition_weights"],
        )
