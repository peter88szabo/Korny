"""NumPy reference for the local Gaussian kernel force field.
Run: python gaussian_tt_reference.py
Units of lengths, energies and residual labels must be consistent.
Local features and all derivative directions use float64.
"""
from itertools import combinations_with_replacement
from math import pi
import numpy as np


class Jet:
    """Value, directional derivatives u and v, and mixed derivative uv."""
    def __init__(self, x, u=0., v=0., uv=0.):
        self.x = np.asarray(x, dtype=float)
        self.u, self.v, self.uv = [np.broadcast_to(a, self.x.shape)
                                   for a in (u, v, uv)]

    def parts(self):
        return self.x, self.u, self.v, self.uv

    def linear(self, op):
        return Jet(*(op(p) for p in self.parts()))

    def __getitem__(self, idx):
        return self.linear(lambda p: p[idx])

    def __add__(self, other):
        b = jet(other)
        return Jet(*(a + c for a, c in zip(self.parts(), b.parts())))

    __radd__ = __add__

    def __neg__(self):
        return self.linear(lambda p: -p)

    def __sub__(self, other):
        return self + (-jet(other))

    def __rsub__(self, other):
        return jet(other) + (-self)

    def __mul__(self, other):
        return bilinear(self, jet(other), np.multiply)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return bilinear(self, jet(other), np.matmul)

    def unary(self, f, df, ddf):
        return Jet(f, df*self.u, df*self.v,
                   df*self.uv + ddf*self.u*self.v)

    def exp(self):
        f = np.exp(self.x)
        return self.unary(f, f, f)

    def power(self, p):
        if p == 0:
            return Jet(np.ones_like(self.x))
        if p == 1:
            return self
        x = self.x
        return self.unary(x**p, p*x**(p-1), p*(p-1)*x**(p-2))


def jet(a):
    return a if isinstance(a, Jet) else Jet(a)


def bilinear(a, b, op):
    return Jet(op(a.x, b.x),
               op(a.u, b.x) + op(a.x, b.u),
               op(a.v, b.x) + op(a.x, b.v),
               op(a.uv, b.x) + op(a.u, b.v) +
               op(a.v, b.u) + op(a.x, b.uv))


def join(items, concatenate=False):
    op = np.concatenate if concatenate else np.stack
    return Jet(*(op(a) for a in zip(*(j.parts() for j in items))))


def descriptor(R, Z, center, nr=4, lmax=3, ron=4., rc=5.5,
               species=(1, 6, 7, 8)):
    """Return list of blocks; R is an (N,3) Jet in angstroms."""
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
    d = np.diag(K).copy()
    cols = np.zeros((len(K),min(limit,len(K))))
    selected = []
    for j in range(cols.shape[1]):
        p = int(np.argmax(d))
        if d[p] <= tol:
            break
        cols[:,j] = (K[:,p]-cols[:,:j]@cols[p,:j])/np.sqrt(d[p])
        selected.append(p)
        d = np.maximum(d-cols[:,j]**2,0.)
        d[selected] = 0.
    return np.array(selected,dtype=int)


class SparseGaussian:
    def __init__(self, centers, widths):
        self.centers = np.asarray(centers)
        self.widths = tuple(widths)
        K = np.stack([gaussian_rows(Jet(y),centers).x for y in centers])
        e,U = eig_factor(K)
        self.W = U/np.sqrt(e)
        self.size = len(e)

    def features(self, blocks):
        return gaussian_rows(join(blocks,True),self.centers) @ self.W

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
    def __init__(self, sparse, order, cap, tau=1e-9):
        self.widths = sparse.widths
        self.indices = [multiindices(p,order) for p in self.widths]
        cuts = np.cumsum(self.widths)[:-1]
        zs = np.split(sparse.centers,cuts,axis=1)
        Vs = []
        for centers,indices in zip(zs,self.indices):
            Vs.append(np.stack([polynomial(Jet(z),indices).x*
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
    """One local head per species. Atomic offsets and physics are external."""
    def __init__(self, structures, specification, heads="gaussian", M=16,
                 order=3, cap=8, ell=2., scale_floor=0.1):
        self.spec = specification
        self.species = tuple(specification["species"])
        self.heads,self.transform,self.empty = {},{},{}
        for s in self.species:
            rows = [descriptor(Jet(R),Z,i,**self.spec) for Z,R in structures
                    for i in range(len(Z)) if Z[i]==s]
            if not rows:
                raise ValueError("Missing central species in training")
            widths = [q.x.size for q in rows[0]]
            raw = np.stack([join(row,True).x for row in rows])
            origin = raw.mean(axis=0)
            scale = np.maximum(raw.std(axis=0),scale_floor)
            # ell is a scalar or one positive value per block.
            ls = np.broadcast_to(ell,(len(widths),))
            if np.any(ls<=0) or scale_floor<=0:
                raise ValueError("Scales and bandwidths must be positive")
            divisor = scale*np.repeat(ls,widths)
            cuts = np.cumsum(widths)[:-1]
            self.transform[s] = (np.split(origin,cuts),
                                 np.split(1./divisor,cuts))
            y = (raw-origin)/divisor
            K = np.stack([gaussian_rows(Jet(t),y).x for t in y])
            sparse = SparseGaussian(y[pivots(K,M)], widths)
            if heads=="gaussian":
                self.heads[s] = sparse
            elif heads=="tt":
                self.heads[s] = TaylorTT(sparse,order,cap)
            else:
                raise ValueError("Unknown head")
            self.empty[s] = self.heads[s].features(self.scale(s,[
                Jet(np.zeros(p)) for p in widths])).x

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

    def fit(self, records, ridge=1e-9, curvature=()):
        # records already contain physical-baseline-subtracted labels.
        self.refresh_empty()
        A,y = [],[]
        for Z,R,energy,force in records:
            A.append(self.row(Z,R).x)
            y.append(energy)
            scale = 1./np.sqrt(R.size)
            for v,f in zip(np.eye(R.size).reshape(-1,*R.shape),force.ravel()):
                A.append(-scale*self.row(Z,R,v).u)
                y.append(scale*f)
        for Z,R,u,v,target,weight in curvature:
            A.append(weight*self.row(Z,R,u,v).uv)
            y.append(weight*target)
        A = np.stack(A)
        reg = np.sqrt(ridge)*np.eye(A.shape[1])
        self.w = np.linalg.lstsq(np.vstack((A,reg)),
                                np.r_[y,np.zeros(A.shape[1])],rcond=None)[0]

    def freeze(self):
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
        energy = self.scalar(Z,R).x
        force,hv = [],[]
        direction = 0. if v is None else v
        for u in np.eye(R.size).reshape(-1,*R.shape):
            out = self.scalar(Z,R,u,direction)
            force.append(-out.u)
            hv.append(out.uv)
        return energy,np.array(force).reshape(R.shape),np.array(hv).reshape(R.shape)


def ts_energy(R, Z, log_volumes, free, sR, damping=20.):
    """Atomic units. free[Z] holds (alpha0, C6_0, RvdW_0).
    R and log_volumes carry matching geometry derivatives as Jets.
    """
    vol = jet(log_volumes).exp()
    pars = np.array([free[int(z)] for z in Z])
    alpha = vol*pars[:,0]
    c6 = vol*vol*pars[:,1]
    radius = vol.power(1./3.)*pars[:,2]
    energy = Jet(0.)
    for i in range(len(Z)):
        for j in range(i):
            dr = R[i]-R[j]
            r = (dr@dr).power(.5)
            # Algebraically stable at large positive damping argument.
            exponent = -damping*(r*(sR*(radius[i]+radius[j])).power(-1.)-1.)
            damp = (1.+exponent.exp()).power(-1.)
            ratio = alpha[j]*alpha[i].power(-1.)
            mixed = 2.*c6[i]*c6[j]*(ratio*c6[i]+
                                    ratio.power(-1.)*c6[j]).power(-1.)
            energy = energy-damp*mixed*r.power(-6.)
    return energy


def d3_energy_gradient_bohr(Z, R, method, atm):
    from dftd3.interface import DispersionModel, RationalDampingParam
    model = DispersionModel(np.asarray(Z),np.asarray(R))
    params = RationalDampingParam(method=method,atm=atm)
    result = model.get_dispersion(params,grad=True)
    return float(result["energy"]),np.asarray(result["gradient"])


def mbd_energy_bohr(Z, R, log_volumes, beta):
    from pymbd.fortran import MBDGeom
    symbols = {1:"H",6:"C",7:"N",8:"O"}
    geom = MBDGeom(np.asarray(R))
    return float(geom.mbd_energy_species([symbols[int(s)] for s in Z],
                                          np.exp(log_volumes),beta=beta))


def check():
    # TS asymptote for a pair with fixed illustrative atomic data, in a.u.
    # This checks the algebra, without introducing elemental reference data.
    dummy = {1:(1.,2.,1.)}
    pair = np.array([[0.,0.,0.],[20.,0.,0.]])
    translation = np.array([[0.,0.,0.],[1.,0.,0.]])
    e = ts_energy(Jet(pair,translation,translation),[1,1],Jet(np.zeros(2)),
                  dummy,1.)
    scaled = [-e.x*20**6,e.u*20**7/6,-e.uv*20**8/42]
    assert np.allclose(scaled,[2.,2.,2.],rtol=1e-10)
    print("TS tail from energy, force, and curvature =",scaled)
    def varying_ts(r):
        rpos = Jet(np.array([[0.,0.,0.],[r,0.,0.]]),
                   translation,translation)
        dr = rpos[1]-rpos[0]
        dist = (dr@dr).power(.5)
        logv = join([.15*(-.2*dist).exp(),-.10*(-.3*dist).exp()])
        return ts_energy(rpos,[1,1],logv,dummy,1.)
    center,h = 3.,1e-5
    value = varying_ts(center)
    above,below = varying_ts(center+h),varying_ts(center-h)
    assert abs((above.x-below.x)/(2*h)-value.u)<1e-8
    assert abs((above.u-below.u)/(2*h)-value.uv)<1e-8
    rng = np.random.default_rng(19)
    widths = [4]*4+[10]*12+[16]*18
    n,m,D = 12,256,sum(widths)
    x = rng.normal(0,.13,(n,D))
    z = rng.normal(0,.13,(m,D))
    v = rng.normal(size=D)
    v /= np.linalg.norm(v)
    ell = rng.uniform(.8,1.6,len(widths))
    part = Jet(np.ones((n,m)))
    log = Jet(np.zeros((n,m)))
    start = 0
    for p,l in zip(widths,ell):
        end = start+p
        d = Jet(x[:,None,start:end]-z[None,:,start:end],
                v[start:end],v[start:end])*(1./l)
        term = -0.5*(d*d).linear(lambda a:a.sum(axis=-1))
        part = part*term.exp()
        log = log+term
        start = end
    fused = log.exp()
    relative = [np.linalg.norm(a-b)/np.linalg.norm(a)
                for a,b in zip(part.parts(),fused.parts())]
    assert max(relative)<2e-14
    print("34 blocks: relative error k, u, v, uv =",relative)

    sparse = SparseGaussian(rng.uniform(-.45,.45,(5,3)),[1,1,1])
    probes = [[Jet(np.array([t]),.21,-.17,.01) for t in row]
              for row in rng.uniform(-.4,.4,(10,3))]
    cols = sparse.centers.T
    Vs,trans = [],[]
    for points in cols:
        G = np.exp(-.5*(points[:,None]-points[None,:])**2)
        # Directly projecting the local span can expose ill conditioning.
        val,U = eig_factor(G,1e-14)
        T = (U/np.sqrt(val)).T
        trans.append(T)
        Vs.append(T@G)
    cores,B,_ = tensor_sweep(Vs,sparse.W,cap=12,tau=1e-9)
    target = sparse.features(probes[0])
    row = Jet(np.ones(1))
    for p,zg,T,C in zip(probes[0],cols,trans,cores):
        q = gaussian_rows(p,zg[:,None])@T.T
        row = row @ q.linear(lambda a:np.tensordot(a,C,axes=(0,1)))
    err = max(np.max(abs(a-b)) for a,b in zip((row@B).parts(),target.parts()))
    print("All-center TT, max value/derivative difference =",err)
    assert err<2e-7

    tt = TaylorTT(sparse,order=5,cap=12)
    y = probes[0]
    actual = tt.teacher_features(y)
    # Independent direct finite Taylor product, including kernel curvature.
    k = Jet(np.ones(len(sparse.centers)))
    for g,p in enumerate(y):
        z = sparse.centers[:,g]
        h = p[0]*z
        term,series = Jet(np.ones_like(z)),Jet(np.ones_like(z))
        for j in range(1,6):
            term = term*h*(1./j)
            series = series+term
        k = k*((p[0]*p[0]+z*z)*(-.5)).exp()*series
    err = max(np.max(abs(a-b)) for a,b in zip(actual.parts(),(k@sparse.W).parts()))
    print("Taylor TT versus direct polynomial product =",err)
    assert err<2e-7
    for sites in (1,2):
        small = TaylorTT(sparse,order=2,cap=2)
        def loss():
            return sum(sum(np.sum((a-b)**2)*w for a,b,w in zip(
                small.teacher_features(p).parts(),sparse.features(p).parts(),
                (1.,1.,0.,1.))) for p in probes)
        before = loss()
        refine_tt(small,sparse,probes,sites=sites,cap=4,sweeps=1,
                  derivative_weights=(1.,1.,0.,1.))
        after = loss()
        print("DMRG sites, calibration loss before/after =",sites,before,after)
        assert after < before

    Z = np.array([8,1,1])
    base = np.array([[0.,0.,0.],[.96,0.,0.],[-.26,.93,0.]])
    assert sum(b.x.size for b in descriptor(Jet(base),Z,0)) == 424
    spec = dict(nr=1,lmax=1,ron=1.4,rc=3.2,species=(1,8))
    structures = [(Z,base+rng.normal(0,.045,base.shape)) for _ in range(8)]
    # An uncompressed local model supplies exact synthetic residual labels.
    teacher = Molecules(structures,spec,M=3,ell=2.)
    teacher.w = rng.normal(size=sum(h.size for h in teacher.heads.values()))
    teacher.freeze()
    records = []
    for zz,rr in structures:
        en,f,_ = teacher.observables(zz,rr)
        records.append((zz,rr,en,f))
    direc = rng.normal(size=base.shape)
    direc /= np.linalg.norm(direc)
    for kind in ("gaussian","tt"):
        model = Molecules(structures,spec,kind,M=3,order=4,cap=3,ell=2.)
        zz,rr = structures[0]
        curv = teacher.scalar(zz,rr,direc,direc).uv
        model.fit(records,ridge=1e-10,curvature=[(zz,rr,direc,direc,curv,.1)])
        model.freeze()
        en,f,hv = model.observables(Z,base,direc)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.npz"
            model.save_local(path)
            other = Molecules.load_local(path)
            predicted = other.scalar(Z,base,direc,direc)
            expected = model.scalar(Z,base,direc,direc)
            assert all(np.allclose(a,b,rtol=0,atol=1e-12)
                       for a,b in zip(predicted.parts(),expected.parts()))
        assert abs(en-model.row(Z,base).x@model.w)<1e-10
        rot,_ = np.linalg.qr(rng.normal(size=(3,3)))
        er,fr,hr = model.observables(Z,base@rot,direc@rot)
        assert abs(er-en)<1e-10 and np.max(abs(fr-f@rot))<1e-9
        assert np.max(abs(hr-hv@rot))<1e-8
        H = np.stack([model.observables(Z,base,e)[2].ravel()
                      for e in np.eye(base.size).reshape(-1,*base.shape)],axis=1)
        assert np.max(abs(H-H.T))<1e-9
        assert np.max(abs(H@direc.ravel()-hv.ravel()))<1e-9
        en0,f0,_ = teacher.observables(Z,base)
        step = 2e-5
        ep,fp,_ = model.observables(Z,base+step*direc)
        em,fm,_ = model.observables(Z,base-step*direc)
        ferr = abs((ep-em)/(2*step)+np.sum(f*direc))
        herr = np.max(abs(hv+(fp-fm)/(2*step)))
        assert ferr<1e-6 and herr<1e-5
        perm = [2,0,1]
        assert abs(model.row(Z[perm],base[perm]).x@model.w-en)<1e-10
        pair = np.vstack((base,base+np.array([15.,0.,0.])))
        residual = model.row(np.tile(Z,2),pair).x@model.w-2.*en
        assert abs(residual)<1e-10 and np.linalg.norm(f.sum(axis=0))<1e-10
        print(kind,"held-out |dE|, max|dF| =",abs(en-en0),np.max(abs(f-f0)))
        print(kind,"directional FD residuals and additivity =",ferr,herr,residual)


def time_gaussian():
    """Frozen descriptors, E, all descriptor gradients, and one HVP."""
    from time import perf_counter
    rng = np.random.default_rng(191)
    widths = np.array([4]*4+[10]*12+[16]*18)
    ends = np.cumsum(widths)
    slices = [slice(i,j) for i,j in zip(np.r_[0,ends[:-1]],ends)]
    X = rng.normal(0,.13,(32,424))
    Z = rng.normal(0,.13,(256,424))
    c = rng.normal(size=256)
    v = rng.normal(size=424)
    v /= np.linalg.norm(v)
    def batch(mode):
        energies,gradients,hvps = [],[],[]
        for x in X:
            delta = x-Z
            if mode=="product":
                h = np.ones(len(Z))
                for sl in slices:
                    d = delta[:,sl]
                    h *= np.exp(-.5*np.einsum("ij,ij->i",d,d))
            elif mode=="sum":
                q = np.zeros(len(Z))
                for sl in slices:
                    d = delta[:,sl]
                    q += np.einsum("ij,ij->i",d,d)
                h = np.exp(-.5*q)
            else:
                h = np.exp(-.5*np.einsum("ij,ij->i",delta,delta))
            a = c*h
            energies.append(a.sum())
            gradients.append(-a@delta)
            hvps.append((a*(delta@v))@delta-v*a.sum())
        return list(map(np.asarray,(energies,gradients,hvps)))
    modes = ["product","sum","flat"]
    outputs = [batch(mode) for mode in modes]
    assert all(np.allclose(x,y,rtol=1e-10,atol=1e-12)
               for result in outputs[1:] for x,y in zip(outputs[0],result))
    times = {m:[] for m in modes}
    for j in range(11):
        for mode in modes[j%3:]+modes[:j%3]:
            start = perf_counter()
            batch(mode)
            times[mode].append(1000*(perf_counter()-start))
    for mode in modes:
        print(mode,"median ms for 32 queries =",round(np.median(times[mode]),3))


if __name__ == "__main__":
    import sys
    if "--timing" in sys.argv:
        time_gaussian()
    else:
        check()




