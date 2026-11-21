"""Sparse Gaussian kernels on smooth, element-resolved bond environments.

Each environment contains the central length, first-shell coordination in two
length windows and the elemental coordination one bond further away. The short
window resolves bond contraction and partial multiple-bond character without
assigning integer bond orders or changing graphs during dynamics.
"""
import numpy as np
from .bond_environment import ELEMENTS, PAIRS
from .four_body import RADII, switch

METRIC = np.array([4.5] + [1.] * 8 + [1.5] * 8 + [.4] * 8)


def environments(z, r, gradient=False):
    """Return both atom orders for like-element bonds and their pair-distance Jacobian."""
    keep = np.flatnonzero(np.isin(z, ELEMENTS))
    z = np.asarray(z)[keep]
    a, b = np.triu_indices(len(z), 1)
    dr = r[keep[b]] - r[keep[a]]
    d = np.linalg.norm(dr, axis=1)
    rad = np.array([RADII[int(s)] for s in z])
    scale = rad[a] + rad[b]
    use = d < 1.95 * scale
    a, b, dr, d, scale = (v[use] for v in (a, b, dr, d, scale))
    typ = np.searchsorted(ELEMENTS, z)
    edge = np.arange(len(d))
    x = d / scale
    near, dn = switch(x, .98, 1.65)
    short, ds = switch(x, .83, 1.09)
    gate, dg = switch(x, 1.25, 1.95)
    # Four columns count H/C/N/O independently. A neighbor's own shell
    # contributes to the two-hop message after removing the return edge.
    atoms = np.eye(4)[typ]
    coord = np.zeros((len(z), 4))
    tight = np.zeros_like(coord)
    u, v = np.r_[a, b], np.r_[b, a]
    edges = np.tile(edge, 2)
    np.add.at(coord, u, near[edges, None] * atoms[v])
    np.add.at(tight, u, short[edges, None] * atoms[v])
    other = coord[v] - near[edges, None] * atoms[u]
    second = np.zeros_like(coord)
    np.add.at(second, u, near[edges, None] * other)
    if gradient:
        # Propagate derivatives with respect to dimensionless distances first.
        # Only the last contraction maps those distances back to atoms.
        eye = np.eye(len(d))
        dn_edge = dn[edges, None, None] * eye[edges, None, :]
        ds_edge = ds[edges, None, None] * eye[edges, None, :]
        dc = np.zeros((len(z), 4, len(d)))
        dt = np.zeros_like(dc)
        np.add.at(dc, u, dn_edge * atoms[v, :, None])
        np.add.at(dt, u, ds_edge * atoms[v, :, None])
        do = dc[v] - dn_edge * atoms[u, :, None]
        dsecond = np.zeros_like(dc)
        np.add.at(dsecond, u, dn_edge * other[:, :, None] + near[edges, None, None] * do)
    l, rr = typ[u], typ[v]
    # Heteronuclear descriptors use a canonical order. Homonuclear ones average
    # both orders so identical nuclei can exchange through a reaction.
    for k, (zl, zr) in enumerate(PAIRS):
        ids = np.flatnonzero((z[u] == zl) & (z[v] == zr))
        if not len(ids):
            continue
        i, j, e = u[ids], v[ids], edges[ids]
        reverse = (ids + len(d)) % (2 * len(d))
        feat = np.column_stack((x[e]-1, other[reverse], other[ids],
            tight[i] - short[e, None] * atoms[j], tight[j] - short[e, None] * atoms[i],
            second[i] - near[e, None] * other[ids], second[j] - near[e, None] * other[reverse]))
        df = None
        if gradient:
            df = np.concatenate((eye[e, None, :], do[reverse], do[ids],
                dt[i] - ds_edge[ids] * atoms[j, :, None], dt[j] - ds_edge[ids] * atoms[i, :, None],
                dsecond[i] - dn_edge[ids] * other[ids, :, None] - near[e, None, None] * do[ids],
                dsecond[j] - dn_edge[ids] * other[reverse, :, None] - near[e, None, None] * do[reverse]), axis=1)
        norm = .5 if zl == zr else 1.
        geometry = (keep[a], keep[b], dr / (scale*d)[:, None]) if gradient else None
        yield k, feat, norm*gate[e], df, norm*dg[e], e, geometry


class BondOrderKernel:
    """A local energy readout using element-pair-specific training exemplars."""
    def __init__(self, centers, offsets, length, weights=None, transforms=None, gate_coefficients=None, gate_bias=None, energy_coefficients=None, center_energies=None):
        self.centers = np.asarray(centers, dtype=float)
        self.offsets = np.asarray(offsets, dtype=int)
        self.length = float(length)
        self.transforms = None if transforms is None else np.asarray(transforms, dtype=float)
        if self.transforms is not None and (self.transforms.shape != (len(PAIRS), len(METRIC), len(METRIC))
                                            or not np.isfinite(self.transforms).all()):
            raise ValueError("Invalid element-pair kernel metrics")
        self.size = len(self.centers)
        self.gate_coefficients = None if gate_coefficients is None else np.asarray(gate_coefficients)
        self.gate_bias = None if gate_bias is None else np.asarray(gate_bias)
        if self.gate_coefficients is not None and (self.gate_coefficients.shape != (self.size,)
                                                  or self.gate_bias is None or self.gate_bias.shape != (len(PAIRS),)):
            raise ValueError("Invalid reaction-center gate")
        self.energy_coefficients = None if energy_coefficients is None else np.asarray(energy_coefficients, dtype=float)
        self.center_energies = None if center_energies is None else np.asarray(center_energies, dtype=float)
        if (self.energy_coefficients is None) != (self.center_energies is None):
            raise ValueError("The local-energy metric requires coefficients and center energies")
        if self.energy_coefficients is not None and (self.energy_coefficients.shape != (self.size,)
                or self.center_energies.shape != (self.size,) or self.gate_coefficients is not None):
            raise ValueError("Invalid local-energy metric")
        self.weights = np.zeros(self.size) if weights is None else np.asarray(weights)
        if (self.centers.shape != (self.size, len(METRIC)) or self.weights.shape != (self.size,)
                or self.offsets.shape != (len(PAIRS)+1,) or self.offsets[0] != 0
                or self.offsets[-1] != self.size or np.any(np.diff(self.offsets)<0)
                or self.length <= 0):
            raise ValueError("Invalid bond kernel dimensions or length")

    def design(self, z, r, forces=False):
        """Evaluate kernel sums and their analytic negative Cartesian Jacobians."""
        out = np.zeros(self.size)
        jac = np.zeros((len(z), 3, self.size)) if forces else None
        for k, feat, gate, df, dg, e, geometry in environments(z, r, forces):
            sl = slice(*self.offsets[k:k+2])
            delta = (feat[:, None] - self.centers[sl]) * (METRIC / self.length)
            if self.transforms is not None:
                delta = delta @ self.transforms[k]
            kernel = np.exp(-.5*np.einsum('nbd,nbd->nb', delta, delta))
            probability = None
            if self.gate_coefficients is not None:
                # Geometry alone supplies the deployed probability. Training used
                # participation in the archived unstable Hessian direction.
                gate_k = np.exp(-.5*np.einsum("nbd,nbd->nb",delta,delta)*(self.length/1.15)**2)
                logits = self.gate_bias[k] + gate_k @ self.gate_coefficients[sl]
                probability = np.exp(-np.logaddexp(0.,-logits))
            energy_difference = None
            if self.energy_coefficients is not None:
                # A separate path fit supplies a smooth local energy coordinate.
                # The coefficients and center energies include its fitted scale.
                teacher_kernel = np.exp(-.5*np.einsum("nbd,nbd->nb", delta, delta)*(self.length/1.15)**2)
                latent = teacher_kernel @ self.energy_coefficients[sl]
                energy_difference = latent[:,None] - self.center_energies[sl]
                kernel *= np.exp(-.5*(energy_difference/self.length)**2)
            effective = kernel if probability is None else probability[:,None]*kernel
            out[sl] = gate @ effective
            if forces:
                a, b, unit = geometry
                if self.transforms is not None:
                    # Pull the derivative of the anisotropic distance back
                    # through the learned linear transformation of this pair.
                    delta = delta @ self.transforms[k].T
                local = -kernel[:, :, None] * delta * (METRIC / self.length)
                if energy_difference is not None:
                    slope = -np.einsum("nb,nbd->nd", teacher_kernel*self.energy_coefficients[sl], delta)
                    slope *= METRIC * self.length / 1.15**2
                    local -= (kernel*energy_difference/self.length**2)[:,:,None]*slope[:,None,:]
                if probability is not None:
                    gate_slope = -np.einsum("nb,nbd->nd", gate_k*self.gate_coefficients[sl], delta)
                    gate_slope *= (METRIC*self.length/1.15**2)
                    dp = (probability*(1.-probability))[:,None]*gate_slope
                    local = probability[:,None,None]*local + kernel[:,:,None]*dp[:,None,:]
                dx = np.einsum('n,nbd,ndm->mb', gate, local, df, optimize=True)
                np.add.at(dx, e, dg[:, None] * effective)
                f = dx[:, None, :] * unit[:, :, None]
                np.add.at(jac[:, :, sl], a, f)
                np.add.at(jac[:, :, sl], b, -f)
        return out, jac.reshape(r.size, self.size) if forces else None

    def fields(self):
        fields = dict(order_centers=self.centers, order_offsets=self.offsets,
                    order_length=self.length, order_weights=self.weights)
        if self.transforms is not None:
            fields["order_transforms"] = self.transforms
        if self.gate_coefficients is not None:
            fields["order_gate_coefficients"] = self.gate_coefficients
            fields["order_gate_bias"] = self.gate_bias
        if self.energy_coefficients is not None:
            fields["order_energy_coefficients"] = self.energy_coefficients
            fields["order_center_energies"] = self.center_energies
        return fields

    @classmethod
    def load(cls, d):
        return cls(*(d[k] for k in ("order_centers","order_offsets","order_length","order_weights")), transforms=d.get("order_transforms"),
                   gate_coefficients=d.get("order_gate_coefficients"),gate_bias=d.get("order_gate_bias"), energy_coefficients=d.get("order_energy_coefficients"), center_energies=d.get("order_center_energies"))
