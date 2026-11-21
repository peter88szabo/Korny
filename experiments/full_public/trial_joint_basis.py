"""Jointly refit bond Fourier banks, local kernels and two-hop bond kernels.

Streaming the 91,156 paired energy/force anchors avoids constructing the full
HDF5 design in RAM. The high-level and radical readouts each produce a scalar
potential. A saved joint readout can be unpacked into the exact existing Korny
basis blocks, including their force derivatives and dispersion switch.
"""
import json
import pickle
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_multiscale as ms, trial_fit
from experiments.full_public.trial_features import cache
from experiments.full_public.trial_isomers import gram as isomer_gram
from src.forcefield import ForceField
from src.bond_order_kernel import BondOrderKernel
from src.reactive_basis import ReactiveBasis

WORK = kt.WORK
STAGE = "p14-public"
RICH = np.concatenate([np.arange(i*256, i*256+128) for i in range(4)])
BASE = RICH


def arrays(radical=False):
    side = "aux" if radical else "main"
    off = json.loads((WORK/"base-aux-ready.json").read_text())["private"] if radical else 0
    def read(path, extra=off):
        return np.load(path, mmap_mode="r")[extra:]
    return [
        (read(cache("base",1.65,.65,"aux" if radical else None)), BASE),
        (read(WORK/f"four-{side}.npy"), None),
        (read(WORK/f"bond-144-1-angle-{side}.npy"), None),
        (read(WORK/f"bond-256-1-angle-{side}.npy"), None),
        (read(cache("rich",1.,1.,"aux" if radical else None)), RICH),
        (read(WORK/("bo-1.15-rad.npy" if radical else "bo-1.15-main.npy"),0), None),
    ]


def gather(parts, lo, hi):
    return np.column_stack([data[lo:hi] if cols is None else data[lo:hi][:,cols] for data,cols in parts])


def init():
    """Cache only chemical centers, radical displacements, and the HDF sufficient statistic."""
    parts = arrays()
    n = len(np.load(WORK/"prior.npy",mmap_mode="r"))
    pub = gather(parts,0,n)
    rad_parts = arrays(True)
    rad = gather(rad_parts,0,len(rad_parts[0][0]))
    np.save(WORK/"joint14-pub.npy",pub)
    np.save(WORK/"joint14-rad.npy",rad)
    sources = np.load(WORK/"hdf-sources.npy")
    weights = {s:4500 / k for s,k in zip(*np.unique(sources,return_counts=True))}
    w = np.sqrt([weights[s] for s in sources])[:,None]
    anchor = np.zeros((pub.shape[1],pub.shape[1]), order='F')
    # BLAS symmetric rank-k touches only half the matrix and reuses one buffer.
    from scipy.linalg.blas import dsyrk
    for i in range(0,len(sources),512):
        stop = min(i+512,len(sources))
        data = gather(parts,n+3*i,n+3*stop).reshape(stop-i,3,-1)
        e = data[:,0] * w[i:stop]
        f = (data[:,1]-data[:,2]) * w[i:stop] * (4/.002)
        a = np.asfortranarray(np.vstack([e,f]).T)
        anchor = dsyrk(1.,a,beta=1.,c=anchor,overwrite_c=1,lower=1)
    tri = np.tril_indices(len(anchor),-1)
    anchor[(tri[1],tri[0])] = anchor[tri]
    np.save(WORK/"joint14-anchor.npy",anchor)
    print("Finished joint PBE0 energy/force anchor",len(sources),pub.shape[1],flush=True)


def unpack(coef):
    """Merge Fourier blocks using the correct per-bank normalization for this checkpoint."""
    from src.bond_environment import BondBasis
    model = ForceField.load(WORK/"p11-public-high.npz")
    atom = model.reactive_basis
    k = len(BASE)+12
    # The original banks are normalized by sqrt(2 / 144) and sqrt(2 / 256).
    small = coef[k:k+1440].reshape(10,144)*np.sqrt(400/144)
    large = coef[k+1440:k+4000].reshape(10,256)*np.sqrt(400/256)
    atom.weights = np.r_[coef[:k],np.column_stack([small,large]).ravel()]
    k += 4000
    with np.load(WORK/"rich-basis.npz") as f:
        rich = ReactiveBasis.load(f)
    rich = ReactiveBasis(rich.species,[rich.spec[s] for s in ("nr","lmax","ron","rc")],
             rich.projections,{s:c[:128] for s,c in rich.centers.items()},rich.radial_dims,1.,1.,coef[k:k+512])
    k+=512
    with np.load(WORK/"bo-basis.npz") as f:
        order = BondOrderKernel.load(f)
    order.length = 1.15
    order.weights = coef[k:]
    assert len(order.weights) == order.size
    return atom, rich, order


def nist_constraints(target, h, size):
    """Fit actual PBE0 stationary forces and curvatures at the 14 training-only minima."""
    from experiments.full_public.trial_constraints import nist_rows
    parent = ForceField.load(WORK/f"spin-{target}.npz",dispersion="d3")
    if h >= 0:
        cfg = json.loads((WORK/"spin-choices.json").read_text())[target]["best"]
        weights = np.load(WORK / (f"spin-{target}-{h}-" + "-".join(map(str,cfg))+".npy"))
        old = ForceField.load(kt.OUT/"joint-pbe0-reaction.npz")
        parent.weights = old.weights + weights
    atom,rich,order = unpack(np.zeros(size))
    def design(z,r):
        raw = atom.design(z,r,True)[1]
        n = len(BASE)+12
        b = raw[:,n:].reshape(r.size,10,400)
        return np.column_stack((raw[:,:n],
            b[:,:,:144].reshape(r.size,-1)*np.sqrt(400/144),
            b[:,:,144:].reshape(r.size,-1)*np.sqrt(400/256),
            rich.design(z,r,True)[1],order.design(z,r,True)[1]))
    return nist_rows(design,parent)


if __name__ == "__main__" and sys.argv[1] == "prepare":
    with threadpool_limits(5):
        init()


def fit_joint():
    """Profile shape protection and ridge on two grouped inner holdouts, then refit."""
    from experiments.full_public.trial_spectra import evaluate_trial
    from experiments.full_public.tauri_validation import compare
    from experiments.full_public.trial_validation import run as validate
    from scipy.linalg.blas import dsyrk
    import shutil
    pub=np.load(WORK/"joint14-pub.npy",mmap_mode="r")
    rad=np.load(WORK/"joint14-rad.npy",mmap_mode="r").reshape(-1,7,pub.shape[1])
    anchor=np.load(WORK/"joint14-anchor.npy",mmap_mode="r")
    meta=np.load(WORK/"rad-meta.npy")
    rb=np.load(WORK/"rad-prior.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    results={}
    for t in kt.TARGETS:
        d=kt.equation_data(t)
        s,y,src,fold,kind,case=d
        primary=trial_fit.selection_mask(d,t)
        x=s@pub
        options=((.05,.1,40),(.25,.4,160),(1.0,1.,400))
        dev={str(opt):[] for opt in options}
        for h in (0,1,-1):
            col=h if h>=0 else 2
            base=np.load(WORK/f"spin-energy-{t}-{h if h>=0 else 'all'}.npy")
            residual=y-s@base
            train=(fold>=0)&(fold!=h)
            inc=kt.grouped_moments(d,x,residual,train)
            other=inc.pop("other")
            other+=48*inc.pop("mid")
            other+=42*inc.pop("gap")
            other+=3*isomer_gram(t,h,pub,base)
            nist=nist_constraints(t,h,pub.shape[1])
            ng=nist.T@nist
            if t=="wb97xd3":
                zz,yy=trial_fit.rad_rows(rad,rb[:,:,col],meta,np.where(saddle,3.,1.))
                trainshape=np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3)
                other+=trial_fit.reduce_gram(zz,yy,trainshape)
                del zz,yy
            if h<0:
                def score(opt):
                    vals=dev[str(opt)]
                    rx,fw,bw=np.mean([v["maes"] for v in vals],axis=0)
                    physical=np.mean([v.get("saddle", [0])[0] for v in vals])/350
                    return .40*rx+.60*(fw+bw)/2+physical
                selected=min(options,key=score)
            for ridge,scale,wshape in options if h>=0 else [selected]:
                from scipy.linalg import cho_factor,cho_solve
                mat=other[:-1,:-1]+scale*anchor+wshape*ng[:-1,:-1]
                mat.flat[::len(mat)+1]+=ridge
                rhs=other[:-1,-1]+wshape*ng[:-1,-1]
                weights=cho_solve(cho_factor(mat,check_finite=False),rhs,check_finite=False)
                del mat
                coef_key=str((ridge,scale,wshape))
                energy=base+pub@weights
                err=x@weights-residual
                if t=="wb97xd3":
                    modes=rad@weights+rb[:,:,col]
                prefix=f"joint14-{t}-{ridge:g}-{scale:g}-{wshape:g}"
                np.save(WORK/f"{prefix}-weights-{h}.npy",weights)
                np.save(WORK/f"{prefix}-parent-{h}.npy",energy)
                if t=="wb97xd3":
                    np.save(WORK/f"{prefix}-modes-{h}.npy",modes)
                if h>=0:
                    entry={"maes":ms.metrics(d,err,(fold==h)&primary)}
                    if t=="wb97xd3":
                        entry["saddle"]=ms.saddle_metrics(modes,(meta[:,0]==h)&saddle,meta)
                    dev[coef_key].append(entry)
                else:
                    held=fold==-1
                    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",
                        source=src[held],kind=kind[held],case=case[held],error_kcal=err[held]*kt.KCAL)
                    if t=="wb97xd3":
                        np.save(WORK/f"{STAGE}-held-modes.npy",modes[meta[:,0]==-1])
                    model=ForceField.load(WORK/f"spin-{t}.npz",dispersion="d3")
                    model.reactive_basis,rich,order=unpack(weights)
                    model.additional_bases=[rich,order]
                    path=WORK/f"{STAGE}-{t}.npz"
                    model.save_with_base(kt.original.BASE,path)
                    model=ForceField.load(path,dispersion="d3")
                    z,r=trial_fit.sample()
                    v=np.random.default_rng(39).normal(size=r.shape)
                    v/=np.linalg.norm(v)
                    E,F=model.predict(z,r)
                    np.testing.assert_allclose(E,energy[trial_fit.sample.index],rtol=0,atol=1e-7)
                    fd=(model.energy(z,r-.00002*v)-model.energy(z,r+.00002*v))/.00004
                    np.testing.assert_allclose(fd,(F*v).sum(),rtol=1e-5,atol=2e-5)
            print("Completed joint kernel solve",t,h,flush=True)
        results[t]=dict(selected=selected,development=dev)
    kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,branches=results))
    compare(STAGE)
    validate(STAGE)
    print("Completed joint validation",flush=True)


if __name__=="__main__" and sys.argv[1]=="fit":
    with threadpool_limits(5):
        fit_joint()
