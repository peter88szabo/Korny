"""Numerical consistency checks and the Gaussian readout microbenchmark."""
import numpy as np

from .jets import Jet, join
from .local import (Molecules, SparseGaussian, TaylorTT, descriptor, descriptor_array,
                    eig_factor, gaussian_rows, refine_tt, tensor_sweep)
from .dispersion.ts import ts_energy


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
        fast_energy, fast_force, _ = other.observables(Z, base)
        assert abs(fast_energy-en) < 1e-10 and np.max(abs(fast_force-f)) < 1e-9
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

        # A second fit must replace the exported coefficients immediately.
        model.fit([(zz, rr, 0., np.zeros_like(rr))])
        assert model.scalar(Z, base).x == 0.
        try:
            model.fit([(zz, rr, 0., np.zeros_like(rr[:1]))])
        except ValueError as exc:
            assert "force vector" in str(exc)
        else:
            raise AssertionError("Truncated force labels were accepted")

    # Cross-check packed Cartesian derivatives, including a taper and an empty site.
    atoms = np.array([1, 6, 7, 8, 16, 17])
    xyz = rng.normal(size=(len(atoms), 3))
    xyz[-1] = 12.
    direction = rng.normal(size=xyz.shape)
    spec = dict(nr=3, lmax=3, ron=1.2, rc=3., species=tuple(atoms))
    for i in range(len(atoms)):
        raw, jac = descriptor_array(xyz, atoms, i, gradient=True, **spec)
        ref = join(descriptor(Jet(xyz, direction), atoms, i, **spec), True)
        assert np.max(abs(raw-ref.x)) < 1e-12
        assert np.max(abs(jac.reshape(len(raw), -1) @ direction.ravel()-ref.u)) < 1e-11



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
