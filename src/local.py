"""A differentiable, additive molecular energy model.

Each atom sees a smooth finite neighborhood. Radial sums and angular power
spectra encode the neighborhood in an element-resolved, rotation-invariant
vector. Gaussian or compressed tensor-train heads map this vector to an energy.
Summing atomic contributions gives the molecular energy. The NumPy Jacobian
path supplies forces for fitting and dynamics; Jets supply directional second
derivatives for the reference implementation and curvature tests.
"""
from functools import lru_cache
from itertools import combinations_with_replacement
from math import pi

import numpy as np

from .jets import Jet, bilinear, join


def descriptor(R, Z, center, nr=4, lmax=3, ron=4., rc=5.5,
               species=(1, 6, 7, 8)):
    """Build the reference descriptor using differentiable arithmetic.

    Chebyshev polynomials describe radial positions. Legendre polynomials of
    neighbor direction cosines describe bond angles. Species-specific sums
    remove dependence on the input ordering of chemically identical neighbors.
    Each returned block keeps its two directional derivatives for curvature.
    """
    Z = np.asarray(Z)
    if not np.all(np.isin(Z, species)):
        raise ValueError("Unknown chemical species")
    rel = R - R[center]
    lengths = np.linalg.norm(rel.x, axis=1)
    mask = (np.arange(len(Z)) != center) & (lengths < rc)
    if np.any(lengths[mask] < 1e-10):
        raise ValueError("Coincident nuclei")
    a = len(species)*nr
    rows = []
    for j in np.flatnonzero(mask):
        vec = rel[j]
        r = (vec @ vec).power(0.5)
        t = (r - ron)*(1./(rc-ron))
        cutoff = Jet(1.)
        if r.x > ron:
            cutoff = 1. + t.power(4)*(-35.+t*(84.+t*(-70.+20.*t)))
        s = r*(2./rc)-1.
        cheb = [Jet(1.)]
        if nr > 1:
            cheb.append(s)
        for n in range(2, nr):
            cheb.append(2.*s*cheb[-1]-cheb[-2])
        channels = [Jet(0.) for _ in range(a)]
        first = species.index(int(Z[j]))*nr
        channels[first:first+nr] = [cutoff*c for c in cheb]
        rows.append((join(channels), vec*r.power(-1.)))
    if rows:
        U = join([p[0] for p in rows])
        directions = join([p[1] for p in rows])
        cosine = directions @ directions.linear(np.transpose)
        legendre = [Jet(np.ones_like(cosine.x)), cosine]
        for l in range(2, lmax+1):
            legendre.append(((2*l-1)*cosine*legendre[-1]
                             -(l-1)*legendre[-2])*(1./l))
        radial = U.linear(lambda arr: arr.sum(axis=0))
        angles = [(U.linear(np.transpose) @ legendre[l] @ U) *
                  ((2*l+1)/(4*pi)) for l in range(1,lmax+1)]
    else:
        radial = Jet(np.zeros(a))
        angles = [Jet(np.zeros((a,a))) for _ in range(lmax)]
    groups = [radial[s*nr:(s+1)*nr] for s in range(len(species))]
    for s,t in combinations_with_replacement(range(len(species)),2):
        for P in angles:
            block = P[s*nr:(s+1)*nr, t*nr:(t+1)*nr]
            if s == t:
                groups.append(block[np.triu_indices(nr)])
            else:
                groups.append(block.linear(np.ravel))
    return groups


@lru_cache(maxsize=32)
def descriptor_layout(nspecies, nr, lmax):
    """Indices in the block order used by descriptor(), grouped by angular degree."""
    pairs = list(combinations_with_replacement(range(nspecies), 2))
    widths = [nr] * nspecies
    entries = [[] for _ in range(lmax)]
    offset = nspecies * nr
    for s, t in pairs:
        i, j = np.triu_indices(nr) if s == t else np.indices((nr, nr)).reshape(2, -1)
        for group in entries:
            group.append((np.arange(offset, offset + len(i)), s*nr + i, t*nr + j))
            widths.append(len(i))
            offset += len(i)
    groups = [tuple(map(np.concatenate, zip(*group))) for group in entries]
    return tuple(widths), groups


def descriptor_array(R, Z, center, nr=4, lmax=3, ron=4., rc=5.5,
                     species=(1, 6, 7, 8), *, gradient=False):
    """Return the packed descriptor and optionally its Cartesian Jacobian.

    The Jacobian axes are (feature, atom, xyz) and coordinates are in Å. Each
    neighbor derivative is accumulated with the opposite sign on the central
    atom, preserving invariance under uniform translation. The ordering of
    channels must match the Jet implementation and the saved checkpoint.
    """
    widths, groups = descriptor_layout(len(species), nr, lmax)
    rel = R - R[center]
    distance = np.linalg.norm(rel, axis=1)
    neighbors = np.flatnonzero((np.arange(len(Z)) != center) & (distance < rc))
    r = distance[neighbors]
    if np.any(r < 1e-10):
        raise ValueError("Coincident nuclei")
    direction = rel[neighbors] / r[:, None]
    t = np.clip((r - ron) / (rc - ron), 0., 1.)
    cutoff = 1. + t**4 * (-35. + t*(84. + t*(-70. + 20.*t)))
    slope = -140. * t**3 * (1.-t)**3 / (rc-ron)
    x = 2.*r/rc - 1.
    cheb = np.polynomial.chebyshev.chebvander(x, nr-1).reshape(len(r), nr)
    # dT_n(x)/dx from differentiating the same Chebyshev recurrence.
    dcheb = np.zeros_like(cheb)
    if nr > 1:
        dcheb[:, 1] = 1.
    for n in range(2, nr):
        dcheb[:, n] = 2.*(cheb[:, n-1] + x*dcheb[:, n-1]) - dcheb[:, n-2]

    channels = len(species)*nr
    U = np.zeros((len(r), channels))
    columns = np.array([species.index(int(z)) for z in Z[neighbors]], dtype=int)
    idx = (np.arange(len(r))[:, None], columns[:, None]*nr + np.arange(nr))
    U[idx] = cutoff[:, None]*cheb
    raw = np.empty(sum(widths))
    raw[:channels] = U.sum(axis=0)
    jac = None
    if gradient:
        dU = np.zeros_like(U)
        dU[idx] = slope[:, None]*cheb + (2./rc)*cutoff[:, None]*dcheb
        jac = np.zeros((len(raw), len(Z), 3))
        jac[:channels, neighbors] = dU.T[:, :, None]*direction

    cosine = direction @ direction.T
    previous, leg = np.ones_like(cosine), cosine
    dprevious, dleg = np.zeros_like(cosine), np.ones_like(cosine)
    for ell, (pos, a, b) in enumerate(groups, 1):
        if ell > 1:
            previous, leg, dprevious, dleg = (
                leg, ((2*ell-1)*cosine*leg - (ell-1)*previous)/ell,
                dleg, ((2*ell-1)*(leg + cosine*dleg) - (ell-1)*dprevious)/ell)
        factor = (2*ell+1)/(4*pi)
        LU = leg @ U
        raw[pos] = factor*(U.T @ LU)[a, b]
        if gradient:
            # Pull dP_l/dcos back through both unit directions in each pair.
            angular = np.einsum('ij,jc,jk->ick', dleg, U, direction)
            parallel = np.einsum('ick,ik->ic', angular, direction)
            angular = (angular - parallel[:, :, None]*direction[:, None, :])/r[:, None, None]
            radial = dU[:, a]*LU[:, b] + dU[:, b]*LU[:, a]
            deriv = (radial[:, :, None]*direction[:, None, :]
                     + U[:, a, None]*angular[:, b] + U[:, b, None]*angular[:, a])
            jac[pos[:, None], neighbors] = factor*deriv.transpose(1, 0, 2)
    if gradient:
        jac[:, center] = -jac.sum(axis=1)
    return raw, jac


def gaussian_rows(y, centers):
    """One exponential per query-center pair; y is pre-scaled."""
    delta = y - centers
    return (-0.5*(delta*delta).linear(lambda a:a.sum(axis=-1))).exp()


def eig_factor(K, tol=1e-12):
    vals, U = np.linalg.eigh((K+K.T)*0.5)
    keep = vals > tol*vals[-1]
    if not np.any(keep):
        raise ValueError("Zero kernel subspace")
    return vals[keep], U[:,keep]


def pivots(K, limit, tol=1e-12):
    return _pivot_columns(np.diag(K).copy(), lambda p: K[:, p], limit, tol)


def _pivot_columns(d, column, limit, tol=1e-12):
    cols = np.zeros((len(d),min(limit,len(d))))
    selected = []
    for j in range(cols.shape[1]):
        p = int(np.argmax(d))
        if d[p] <= tol:
            break
        cols[:,j] = (column(p)-cols[:,:j]@cols[p,:j])/np.sqrt(d[p])
        selected.append(p)
        d = np.maximum(d-cols[:,j]**2,0.)
        d[selected] = 0.
    return np.array(selected,dtype=int)


class SparseGaussian:
    """Gaussian similarities to representative training environments of one element.

    Whitening the center Gram matrix gives a stable basis for the linear
    energy readout. At inference, the fitted weights absorb this transformation.
    """
    def __init__(self, centers, widths):
        self.centers = np.asarray(centers)
        self.widths = tuple(widths)
        K = np.stack([gaussian_rows(Jet(y),centers).x for y in centers])
        e,U = eig_factor(K)
        self.W = U/np.sqrt(e)
        self.size = len(e)

    def features(self, blocks):
        return gaussian_rows(join(blocks,True),self.centers) @ self.W

    def value_gradient(self, y, factor):
        delta = y - self.centers
        kernel = np.exp(-0.5*np.einsum("ij,ij->i", delta, delta))
        return kernel @ factor, -(kernel[:, None]*delta).T @ factor

    def compiled(self, blocks, weights):
        # Cache W @ weights once during model export in production.
        return gaussian_rows(join(blocks,True),self.centers) @ (self.W@weights)


def multiindices(p, order):
    return [idx for n in range(order+1)
            for idx in combinations_with_replacement(range(p),n)]


def polynomial(y, indices):
    # Recurrence y_j * b_parent / sqrt(alpha_j), safe at y_j=0.
    b = {():Jet(1.)}
    for idx in indices[1:]:
        j = idx[-1]
        b[idx] = b[idx[:-1]]*y[j]*(1./np.sqrt(idx.count(j)))
    return join([b[t] for t in indices])


@lru_cache(maxsize=64)
def monomial_layout(width, order):
    """Parent indices for normalized monomials and their first derivatives."""
    indices = multiindices(width, order)
    positions = {idx: i for i, idx in enumerate(indices)}
    levels = [[] for _ in range(order)]
    derivatives = []
    for i, idx in enumerate(indices[1:], 1):
        j = idx[-1]
        levels[len(idx)-1].append((positions[idx[:-1]], j, 1./np.sqrt(idx.count(j))))
        for j in set(idx):
            k = idx.index(j)
            derivatives.append((j, i, positions[idx[:k]+idx[k+1:]], np.sqrt(idx.count(j))))
    return ([tuple(map(np.asarray, zip(*level))) for level in levels],
            tuple(map(np.asarray, zip(*derivatives))))


def monomial_values(y, order, size, gradient=False):
    levels, derivatives = monomial_layout(len(y), order)
    values = np.ones(size)
    start = 1
    for parent, j, factor in levels:
        values[start:start+len(j)] = values[parent]*y[j]*factor
        start += len(j)
    jac = None
    if gradient:
        jac = np.zeros((len(y), len(values)))
        if derivatives:
            j, i, parent, factor = derivatives
            jac[j, i] = values[parent]*factor
    return values, jac


def tensor_sweep(Vs, W, cap, tau):
    """Feature-map TT-SVD. Each V has shape (site width, M)."""
    M = W.shape[0]
    suffix = [None]*(len(Vs)+1)
    suffix[-1] = np.ones((M,M))
    for g in range(len(Vs)-1,-1,-1):
        suffix[g] = suffix[g+1]*(Vs[g].T@Vs[g])
    metric = W@W.T
    norm2 = np.sum(metric*suffix[0])
    C = np.ones((1,M))
    cores = []
    tails = []
    for g,V in enumerate(Vs):
        old,m = C.shape[0],V.shape[0]
        J = (C[:,None,:]*V[None,:,:]).reshape(old*m,M)
        # Thin QR confines the reduced density matrix to at most M axes.
        Q,R = np.linalg.qr(J,mode="reduced")
        small = R@(metric*suffix[g+1])@R.T
        lam,U = np.linalg.eigh(0.5*(small+small.T))
        lam = np.maximum(lam[::-1],0.)
        tail = np.r_[np.cumsum(lam[::-1])[::-1],0.]
        target = tau*tau*norm2/len(Vs)
        k = min(cap,len(lam),max(1,int(np.flatnonzero(tail<=target)[0])))
        left = Q@U[:,::-1][:,:k]
        cores.append(left.reshape(old,m,k))
        C = left.T@J
        tails.append(tail[k])
    return cores,C@W,np.asarray(tails)


class TaylorTT:
    """Compress the Gaussian expansion across descriptor blocks into tensor cores.

    Each site contracts one block of polynomial features. The Gaussian envelope
    multiplies the resulting product, and the same contractions propagate its
    derivatives. This is an alternative to the directly evaluated Gaussian head.
    """
    def __init__(self, sparse, order, cap, tau=1e-9):
        self.widths = sparse.widths
        self.indices = [multiindices(p,order) for p in self.widths]
        cuts = np.cumsum(self.widths)[:-1]
        zs = np.split(sparse.centers,cuts,axis=1)
        Vs = []
        for centers,indices in zip(zs,self.indices):
            Vs.append(np.stack([monomial_values(z, order, len(indices))[0]*
                                np.exp(-0.5*(z@z)) for z in centers],axis=1))
        self.cores,self.B,self.tails = tensor_sweep(Vs,sparse.W,cap,tau)
        # Thin factor with the same metric as B B^T.
        U,s,_ = np.linalg.svd(self.B,full_matrices=False)
        keep = s > 1e-12*s[0]
        self.L = U[:,keep]*s[keep]
        self.size = self.L.shape[1]

    def chain(self,blocks):
        row = Jet(np.ones(1))
        norm2 = Jet(0.)
        for y,indices,core in zip(blocks,self.indices,self.cores):
            p = polynomial(y,indices)
            A = p.linear(lambda a: np.tensordot(a,core,axes=(0,1)))
            row = row @ A
            norm2 = norm2 + (y@y)
        return (-0.5*norm2).exp()*row

    def value_gradient(self, y, factor):
        matrices, left_derivatives = [], []
        left = np.ones(1)
        start = 0
        for width, indices, core in zip(self.widths, self.indices, self.cores):
            block = y[start:start+width]
            start += width
            p, dp = monomial_values(block, len(indices[-1]), len(indices), gradient=True)
            # Store derivatives of the contracted prefix at this site only.
            left_core = left @ core.reshape(len(left), -1)
            left_derivatives.append(dp @ left_core.reshape(len(p), -1))
            matrix = np.tensordot(p, core, axes=(0, 1))
            matrices.append(matrix)
            left = left @ matrix
        envelope = np.exp(-0.5*(y @ y))
        value = envelope*(left @ factor)
        right = factor
        gradients = []
        for matrix, dleft in zip(reversed(matrices), reversed(left_derivatives)):
            gradients.append(dleft @ right)
            right = matrix @ right
        return value, envelope*np.concatenate(gradients[::-1]) - np.multiply.outer(y, value)

    def features(self,blocks):
        return self.chain(blocks) @ self.L

    def teacher_features(self,blocks):
        return self.chain(blocks) @ self.B

    def compiled(self,blocks,weights):
        # Cache L @ weights once, or fold it into the final core.
        return self.chain(blocks) @ (self.L@weights)


def refine_tt(head, sparse, probes, sites=2, cap=8, sweeps=1,
              derivative_weights=(1., 1., 0., 0.), proximal=1e-9):
    """Small dense DMRG reference. Probes are lists of four-component Jets.
    The supplied weights multiply residuals for value, u, v, uv respectively.
    """
    if sites not in (1,2) or sites > len(head.cores):
        raise ValueError("Choose one or two neighboring sites")
    def outer(a,b):
        return bilinear(a,b,np.multiply.outer)
    def local(p,C):
        return p.linear(lambda a: np.tensordot(a,C,axes=(0,1)))
    examples = []
    for blocks in probes:
        env = (-0.5*sum(y@y for y in blocks)).exp()
        pp = [polynomial(y,idx) for y,idx in zip(blocks,head.indices)]
        examples.append((env,pp,sparse.features(blocks)))
    C = head.cores
    def left_qr(g):
        a,p,b = C[g].shape
        Q,R = np.linalg.qr(C[g].reshape(a*p,b),mode="reduced")
        C[g] = Q.reshape(a,p,-1)
        C[g+1] = np.tensordot(R,C[g+1],axes=(1,0))
    def right_qr(g):
        a,p,b = C[g].shape
        Q,R = np.linalg.qr(C[g].reshape(a,p*b).T,mode="reduced")
        C[g] = Q.T.reshape(-1,p,b)
        C[g-1] = C[g-1]@R.T
    Q,R = np.linalg.qr(head.B.T,mode="reduced")
    head.B = Q.T
    C[-1] = C[-1]@R.T
    for g in range(len(C)-1,0,-1):
        right_qr(g)
    for sweep in range(sweeps):
        for forward in (True,False):
            starts = list(range(len(C)-sites+1))
            for g in (starts if forward else reversed(starts)):
                previous = C[g] if sites==1 else np.tensordot(
                    C[g],C[g+1],axes=(-1,0))
                nparam = previous.size
                design,targets = [],[]
                for env,pp,target in examples:
                    left = env*Jet(np.ones(1))
                    for h in range(g):
                        left = left @ local(pp[h],C[h])
                    right = Jet(head.B)
                    for h in range(len(C)-1,g+sites-1,-1):
                        right = local(pp[h],C[h]) @ right
                    row = outer(left,pp[g])
                    if sites==2:
                        row = outer(row,pp[g+1])
                    row = outer(row,right)
                    for w,arr,t in zip(derivative_weights,row.parts(),
                                       target.parts()):
                        if w:
                            design.append(w*arr.reshape(nparam,-1).T)
                            targets.append(w*t)
                if proximal:
                    design.append(np.sqrt(proximal)*np.eye(nparam))
                    targets.append(np.sqrt(proximal)*previous.ravel())
                theta = np.linalg.lstsq(np.vstack(design),
                                        np.concatenate(targets),rcond=None)[0]
                if sites==1:
                    C[g] = theta.reshape(previous.shape)
                    if forward and g+1<len(C):
                        left_qr(g)
                    if not forward and g:
                        right_qr(g)
                else:
                    a,p,q,b = previous.shape
                    U,s,Vt = np.linalg.svd(theta.reshape(a*p,q*b),
                                           full_matrices=False)
                    k = min(cap,max(1,np.count_nonzero(s>1e-11*s[0])))
                    U,s,Vt = U[:,:k],s[:k],Vt[:k]
                    C[g] = (U if forward else U*s).reshape(a,p,k)
                    C[g+1] = (s[:,None]*Vt if forward else Vt).reshape(k,q,b)
    design,targets = [],[]
    for blocks,(_,_,target) in zip(probes,examples):
        for w,arr,t in zip(derivative_weights,head.chain(blocks).parts(),
                           target.parts()):
            if w:
                design.append(w*arr)
                targets.append(w*t)
    head.B = np.linalg.lstsq(design,targets,rcond=1e-12)[0]
    U,s,_ = np.linalg.svd(head.B,full_matrices=False)
    keep = s>1e-12*s[0]
    head.L = U[:,keep]*s[keep]
    head.size = int(keep.sum())


class Molecules:
    """Fit local atomic heads and sum their contributions to molecular energies.

    Fit scaling and choose sparse centers using training environments only.
    Subtracting each head's value on an empty neighborhood sets its isolated-atom
    contribution to zero. Composition offsets, pair terms, and dispersion are
    therefore explicit contributions in the full force field.
    """
    def __init__(self, structures, specification, heads="gaussian", M=16,
                 order=3, cap=8, ell=2., scale_floor=0.1):
        self.spec = dict(nr=4, lmax=3, ron=4., rc=5.5)
        self.spec.update(specification)
        self.species = tuple(self.spec["species"])
        self.spec["species"] = self.species
        if (self.spec["nr"] < 1 or self.spec["lmax"] < 0
                or not 0 < self.spec["ron"] < self.spec["rc"]):
            raise ValueError("Invalid radial basis, angular degree or cutoff")
        if not self.species or len(set(self.species)) != len(self.species):
            raise ValueError("Species must be a nonempty sequence of distinct atomic numbers")
        if M < 1 or cap < 1 or order < 0:
            raise ValueError("M and cap must be positive and order nonnegative")
        self.heads,self.transform,self.empty = {},{},{}
        widths, _ = descriptor_layout(len(self.species), self.spec["nr"], self.spec["lmax"])
        candidates = {s: [] for s in self.species}
        for Z, R in structures:
            Z, R = self.geometry(Z, R)
            for i, s in enumerate(Z):
                candidates[s].append(descriptor_array(R, Z, i, **self.spec)[0])
        for s in self.species:
            if not candidates[s]:
                raise ValueError("Missing central species in training")
            raw = np.stack(candidates.pop(s))
            origin = raw.mean(axis=0)
            scale = np.maximum(raw.std(axis=0),scale_floor)
            # ell is a scalar or one positive value per block.
            ls = np.broadcast_to(ell,(len(widths),))
            if not np.isfinite(ls).all() or np.any(ls <= 0) or not 0 < scale_floor < np.inf:
                raise ValueError("Scales and bandwidths must be positive")
            divisor = scale*np.repeat(ls,widths)
            cuts = np.cumsum(widths)[:-1]
            self.transform[s] = (np.split(origin,cuts),
                                 np.split(1./divisor,cuts))
            y = (raw-origin)/divisor
            def column(p):
                delta = y-y[p]
                return np.exp(-0.5*np.einsum("ij,ij->i", delta, delta))
            selected = _pivot_columns(np.ones(len(y)), column, M)
            sparse = SparseGaussian(y[selected], widths)
            if heads=="gaussian":
                self.heads[s] = sparse
            elif heads=="tt":
                self.heads[s] = TaylorTT(sparse,order,cap)
            else:
                raise ValueError("Unknown head")
            self.empty[s] = self.heads[s].features(self.scale(s,[
                Jet(np.zeros(p)) for p in widths])).x

    def geometry(self, Z, R):
        Z, R = np.asarray(Z), np.asarray(R, dtype=float)
        if Z.ndim != 1 or not len(Z) or R.shape != (len(Z), 3) or not np.isfinite(R).all():
            raise ValueError("Expected finite (N, 3) coordinates and N atomic numbers")
        if not np.isin(Z, self.species).all():
            raise ValueError("Unknown chemical species")
        return Z, R

    def _first_order(self, Z, R, *, frozen):
        """Energy and Cartesian gradient, or all corresponding linear design columns."""
        Z, R = self.geometry(Z, R)
        values, gradients = [], []
        for s in self.species:
            h = self.heads[s]
            if frozen:
                factor, zero = self.production[s]
            else:
                factor = h.W if isinstance(h, SparseGaussian) else h.L
                zero = self.empty[s]
            mean, inverse = map(np.concatenate, self.transform[s])
            # Keep a common output axis for both scalar and feature evaluations.
            factor = factor.reshape(len(factor), -1)
            value = np.zeros(factor.shape[1])
            grad = np.zeros((R.size, len(value)))
            for i in np.flatnonzero(Z == s):
                raw, jac = descriptor_array(R, Z, i, gradient=True, **self.spec)
                site, derivative = h.value_gradient((raw-mean)*inverse, factor)
                value += site-zero
                grad += jac.reshape(len(raw), -1).T @ (inverse[:, None]*derivative)
            values.append(value)
            gradients.append(grad)
        if frozen:
            return np.sum(values).item(), np.sum(gradients, axis=0).reshape(R.shape)
        return np.concatenate(values), np.concatenate(gradients, axis=1)

    def energy(self, numbers, positions):
        """Evaluate frozen atomic energies without allocating a Cartesian Jacobian.

        The head and the isolated-atom subtraction are identical to the force
        path. Omitting descriptor derivatives matters when scoring an entire
        torsion scan or comparing thousands of conformer energies.
        """
        z, r = self.geometry(numbers, positions)
        total = 0.
        for species in self.species:
            head = self.heads[species]
            coeff, empty = self.production[species]
            mean, inverse = map(np.concatenate, self.transform[species])
            for i in np.flatnonzero(z == species):
                raw, _ = descriptor_array(r, z, i, **self.spec)
                y = (raw-mean)*inverse
                if isinstance(head, SparseGaussian):
                    delta = y-head.centers
                    values = np.exp(-.5*np.einsum("ij,ij->i", delta, delta))
                    total += values @ coeff - empty
                else:
                    total += head.value_gradient(y, coeff)[0] - empty
        return float(total)

    def refresh_empty(self):
        # Call after any refinement of a species basis and before refitting.
        for s,h in self.heads.items():
            zero = [Jet(np.zeros(p)) for p in h.widths]
            self.empty[s] = h.features(self.scale(s,zero)).x

    def scale(self,s,blocks):
        means,inverses = self.transform[s]
        return [(a-b)*c for a,b,c in zip(blocks,means,inverses)]

    def row(self,Z,R,u=0.,v=0.):
        coords = Jet(R,u,v)
        out = {s:Jet(np.zeros(h.size)) for s,h in self.heads.items()}
        for i,s in enumerate(Z):
            blocks = self.scale(s,descriptor(coords,Z,i,**self.spec))
            out[s] = out[s]+self.heads[s].features(blocks)-self.empty[s]
        return join([out[s] for s in self.species],True)

    def fit(self, records, ridge=1e-9, curvature=(), force_weight=1.):
        # records already contain physical-baseline-subtracted labels.
        self.refresh_empty()
        A,y = [],[]
        if not np.isfinite([ridge, force_weight]).all() or ridge < 0 or force_weight < 0:
            raise ValueError("Ridge and force weight must be finite and nonnegative")
        for Z,R,energy,force in records:
            R, force = np.asarray(R), np.asarray(force)
            if force.shape != R.shape or not np.isfinite(force).all():
                raise ValueError("Expected one finite force vector per atom")
            if np.ndim(energy) != 0 or not np.isfinite(energy):
                raise ValueError("Expected a finite scalar energy")
            value, grad = self._first_order(Z, R, frozen=False)
            A.append(value)
            y.append(energy)
            scale = force_weight/np.sqrt(R.size)
            A.extend(-scale*grad)
            y.extend(scale*force.ravel())
        for Z,R,u,v,target,weight in curvature:
            A.append(weight*self.row(Z,R,u,v).uv)
            y.append(weight*target)
        A = np.stack(A)
        reg = np.sqrt(ridge)*np.eye(A.shape[1])
        self.w = np.linalg.lstsq(np.vstack((A,reg)),
                                np.r_[y,np.zeros(A.shape[1])],rcond=None)[0]
        self.freeze()

    def freeze(self):
        """Fold trained readout weights into each head for scalar E/F prediction."""
        self.production = {}
        pos = 0
        for s in self.species:
            h = self.heads[s]
            w = self.w[pos:pos+h.size]
            pos += h.size
            if isinstance(h,SparseGaussian):
                coeff = h.W@w
            else:
                coeff = h.L@w
            self.production[s] = (coeff,self.empty[s]@w)

    def save_local(self, filename):
        import json
        data = {"format":np.array("local_gaussian_tt/1"),
                "spec":np.array(json.dumps(self.spec))}
        for s in self.species:
            key = str(s)+"_"
            h = self.heads[s]
            means,invs = self.transform[s]
            data[key+"widths"] = h.widths
            data[key+"mean"] = np.concatenate(means)
            data[key+"inv"] = np.concatenate(invs)
            data[key+"coeff"],data[key+"empty"] = self.production[s]
            if isinstance(h,SparseGaussian):
                data[key+"kind"] = "gaussian"
                data[key+"centers"] = h.centers
            else:
                data[key+"kind"] = "taylor"
                data[key+"degrees"] = [max(map(len,idx)) for idx in h.indices]
                for g,c in enumerate(h.cores):
                    data[key+"core"+str(g)] = c
        np.savez(filename,**data)

    @classmethod
    def load_local(cls, filename):
        import json
        obj = cls.__new__(cls)
        obj.transform,obj.heads,obj.production = {},{},{}
        with np.load(filename,allow_pickle=False) as data:
            if str(data["format"])!="local_gaussian_tt/1":
                raise ValueError("Incompatible model")
            obj.spec = json.loads(str(data["spec"]))
            obj.spec["species"] = tuple(obj.spec["species"])
            obj.species = obj.spec["species"]
            for s in obj.species:
                key = str(s)+"_"
                widths = tuple(data[key+"widths"])
                cuts = np.cumsum(widths)[:-1]
                obj.transform[s] = (np.split(data[key+"mean"],cuts),
                                    np.split(data[key+"inv"],cuts))
                obj.production[s] = (data[key+"coeff"],data[key+"empty"])
                kind = str(data[key+"kind"])
                if kind=="gaussian":
                    h = SparseGaussian.__new__(SparseGaussian)
                    h.centers = data[key+"centers"]
                elif kind=="taylor":
                    h = TaylorTT.__new__(TaylorTT)
                    h.indices = [multiindices(p,int(n)) for p,n in
                                 zip(widths,data[key+"degrees"])]
                    h.cores = [data[key+"core"+str(g)] for g in range(len(widths))]
                else:
                    raise ValueError("Unknown stored head")
                h.widths = widths
                obj.heads[s] = h
        return obj

    def scalar(self,Z,R,u=0.,v=0.):
        coords = Jet(R,u,v)
        total = Jet(0.)
        for i,s in enumerate(Z):
            y = self.scale(s,descriptor(coords,Z,i,**self.spec))
            h = self.heads[s]
            coeff,zero = self.production[s]
            if isinstance(h,SparseGaussian):
                out = gaussian_rows(join(y,True),h.centers)@coeff
            else:
                out = h.chain(y)@coeff
            total = total+out-zero
        return total

    def observables(self,Z,R,v=None):
        if v is None:
            energy, gradient = self._first_order(Z, R, frozen=True)
            return energy, -gradient, np.zeros_like(gradient)
        energy = self.scalar(Z,R).x
        force,hv = [],[]
        for u in np.eye(R.size).reshape(-1,*R.shape):
            out = self.scalar(Z,R,u,v)
            force.append(-out.u)
            hv.append(out.uv)
        return energy,np.array(force).reshape(R.shape),np.array(hv).reshape(R.shape)
