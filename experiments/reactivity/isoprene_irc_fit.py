"""Local calibration and conformer-disjoint validation for one isoprene H shift.

Use archived M06-2X-D3zero/aug-cc-pVTZ energies and full Cartesian Hessians.
An IRC endpoint can still have negative curvature, so it is a reference point
on a path, not necessarily a minimum. Train gradients only where the archive
contains the matching Cartesian gradient. Exclude paths similar to the
external NEB validation geometries before splitting the remaining conformers.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import null_space

from experiments.reactivity.matern_fit import OUT
from experiments.reactivity.frequency_validation import TO_CM1
from experiments.reactivity.radical_validation import frames,validate
from experiments.reactivity.thermochemistry import EV_TO_KCAL,score
from src.forcefield import ForceField
from src.units import BOHR_A,HARTREE_EV

ROOT = (Path("/home/peter/Dropbox/Research_Leuven/LIM1_reevol/Ready_calc/Saving_from_Iris/Case-2")
        / "SuperFineGrid_16-H-Shift_from_Z-delta-OH_extended_case_II")


@dataclass
class Point:
    z: np.ndarray
    r: np.ndarray
    energy: float
    hessian: np.ndarray
    force: np.ndarray | None
    negatives: int


def read_point(path,saddle):
    log=path.read_text(errors="replace")
    command=re.search(r"\|\s*\d+>\s*!([^\n]+)",log)
    if (not command or not {"uks","m062x","d3zero","aug-cc-pvtz","freq","defgrid3"}<=set(command[1].lower().split())
            or "ORCA TERMINATED NORMALLY" not in log
            or not re.search(r"Total Charge\s+Charge\s+\.+\s+0\s+Multiplicity\s+Mult\s+\.+\s+2\b",log)):
        raise ValueError("Unverified method/state "+str(path))
    energies=re.findall(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)",log)
    freq=re.findall(r"^\s*\d+:\s+([-\d.]+) cm\*\*-1",log.rsplit("VIBRATIONAL FREQUENCIES",1)[-1],re.M)
    negatives=sum(float(v)<-1 for v in freq)
    if not energies or (saddle and (negatives!=1 or "THE OPTIMIZATION HAS CONVERGED" not in log)):
        raise ValueError("Unverified transition state "+str(path))
    en=float(energies[-1])
    text=path.with_suffix(".hess").read_text()
    atoms=text.split("$atoms\n",1)[1].strip().split("\n\n",1)[0].splitlines()
    n=int(atoms[0]); xyz=[line.split() for line in atoms[1:n+1]]
    z=np.array([{"H":1,"C":6,"O":8}[a[0]] for a in xyz])
    r=np.array([a[2:5] for a in xyz],float)*BOHR_A
    xyz_log=log.rsplit("CARTESIAN COORDINATES (ANGSTROEM)",1)[1].strip().split("\n\n",1)[0]
    check=np.array([row.split()[1:] for row in xyz_log.splitlines()[1:]],float)
    np.testing.assert_allclose(check,r,atol=2e-5)
    if len(freq)!=r.size or Counter(z)!={6:5,1:9,8:3}:
        raise ValueError("Inconsistent spectrum or composition")
    data=iter(text.split("$hessian\n",1)[1].strip().splitlines())
    if int(next(data))!=r.size:
        raise ValueError("Invalid Hessian size")
    h=np.full((r.size,r.size),np.nan)
    while not np.isfinite(h[-1,-1]):
        cols=np.array(next(data).split(),int)
        for i in range(len(h)):
            row=next(data).split()
            if int(row[0])!=i:
                raise ValueError("Mismatched Hessian row")
            h[i,cols]=np.array(row[1:],float)
    # Finite differences of ORCA gradients can produce an antisymmetric
    # numerical component. Use the symmetric energy curvature and confirm
    # its INTERNAL spectrum against the frequencies printed by ORCA itself.
    if np.linalg.norm(h-h.T)/np.linalg.norm(h) > .03:
        raise ValueError("Inconsistent numerical Hessian: "+str(path))
    h=.5*(h+h.T)*HARTREE_EV/BOHR_A**2
    mass=np.array([a[1] for a in xyz],float)
    rc=r-np.average(r,axis=0,weights=mass)
    rigid=np.array([v.ravel() for axis in np.eye(3) for v in
                    (np.sqrt(mass)[:,None]*axis,np.sqrt(mass)[:,None]*np.cross(rc,axis))])
    internal=null_space(rigid)
    sm=np.repeat(np.sqrt(mass),3)
    eig=np.linalg.eigvalsh(internal.T@(h/np.outer(sm,sm))@internal)
    predicted=np.sign(eig)*np.sqrt(abs(eig))*TO_CM1
    factor=re.search(r"Scaling factor for frequencies\s*=\s*([\d.]+)",log.rsplit("VIBRATIONAL FREQUENCIES",1)[-1])
    frequency_scale=float(factor[1]) if factor else 1.
    np.testing.assert_allclose(frequency_scale*predicted,sorted(map(float,freq[6:])),atol=3.,rtol=.003)
    force=None
    if path.with_suffix(".engrad").exists():
        values=[line.strip() for line in path.with_suffix(".engrad").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        # Some archived optimization gradients precede the final frequency
        # geometry. Use only records with matching energy and coordinates.
        if int(values[0])==n and abs(float(values[1])-en)<2e-6:
            coords=np.array([v.split()[1:] for v in values[2+r.size:2+r.size+n]],float)*BOHR_A
            if np.allclose(coords,r,atol=1e-5):
                force=-np.array(values[2:2+r.size],float)*HARTREE_EV/BOHR_A
    return Point(z,r,en*HARTREE_EV,h,force,negatives)


def fingerprint(z,r):
    """Sorted pair distances within element pairs, invariant to rigid motion and atom ordering."""
    i,j=np.triu_indices(len(z),1)
    d=np.linalg.norm(r[i]-r[j],axis=1)
    return d[np.lexsort((d,np.maximum(z[i],z[j]),np.minimum(z[i],z[j])))]


def partition_paths(paths,external):
    """Keep all conformers with any near-duplicate point in the same component."""
    signatures={k:np.array([fingerprint(x.z,x.r) for x in points]) for k,points in paths.items()}
    components=[{k} for k in paths]
    def close(a,b):
        return np.min(np.mean((a[:,None,:]-b[None,:,:])**2,axis=2)) < .05**2
    for i,k in enumerate(paths):
        for other in list(paths)[:i]:
            if close(signatures[k],signatures[other]):
                left=next(c for c in components if k in c)
                right=next(c for c in components if other in c)
                if left is not right:
                    left.update(right)
                    components.remove(right)
    groups=sorted(components,key=lambda c:sha256(",".join(sorted(c)).encode()).digest())
    reserved=[c for c in groups if any(close(signatures[k],external) for k in c)]
    clean=[c for c in groups if c not in reserved]
    if len(clean)<3:
        raise ValueError("Insufficient independent conformer groups: "+str(clean))
    test_count=max(1,round(.2*len(clean)))
    dev_count=max(1,round(.2*len(clean)))
    role={k:"overlap_with_NEB" for c in reserved for k in c}
    for i,c in enumerate(clean):
        split="test" if i<test_count else "development" if i<test_count+dev_count else "train"
        role.update({k:split for k in c})
    return groups,role


def path_data():
    """Read matched stationary labels and group conformers before seeing prediction errors."""
    files=["IRC_final_and_start_freqs/IRC_Starts_Freq","TS-Opt_only_realTSs",
           "IRC_final_and_start_freqs/IRC_Finals_Freq"]
    paths={}
    for folder in sorted((ROOT/files[1]).glob("conformer_*")):
        pts=[]
        for n,base in enumerate(files):
            out,=(ROOT/base/folder.name).glob("iso*.out")
            pts.append(read_point(out,n==1))
        paths[folder.name]=pts
    external=json.loads((OUT/"matern_spin_validation.json").read_text())["paths"]
    sigs=[]
    for item in external.values():
        d=Path(item["folder"])
        trajectory=list(frames(next(d.glob("*_MEP_trj.xyz"))))
        saddle=next(frames(next(d.glob("*_NEB-TS_converged.xyz"))))
        sigs.extend(fingerprint(z,r) for z,r,_ in (trajectory[0],saddle,trajectory[-1]))
    clusters,roles=partition_paths(paths,np.array(sigs))
    return paths,external,clusters,roles


def main():
    parent=json.loads((OUT/"peroxy_transfer_bridges.json").read_text())["checkpoint"]
    model=ForceField.load(OUT/parent,dispersion="d3")
    paths,external,clusters,roles=path_data()
    print("independent groups",clusters,"path roles",Counter(roles.values()),flush=True)
    def design(z,r,forces=False):
        # The original potential stays as a prior. Refine the local kernel and
        # the chemical response only on the neutral doublet surface.
        h,j=model.chemical.design(z,r,gradient=forces)
        v,k=model.matern.design(model.matern.project(z,r,gradient=forces),r.size if forces else 0)
        return np.r_[h,.25*v],np.column_stack((j,.25*k)) if forces else None
    a=[];b=[];blocks={};n_gradient=0;n_curv=0;dev_curv=[]
    stoich=np.array([[-1,1,0],[0,1,-1],[-1,0,1]])
    for tag,pts in paths.items():
        split=roles[tag]
        if split=="overlap_with_NEB":
            continue
        X=[]; y=[]
        for p in pts:
            x,j=design(p.z,p.r,True)
            energy,f=model.predict(p.z,p.r,0,2)
            X.append(x);y.append(p.energy-energy)
            if split not in ("train","development"):
                continue
            if split=="train" and p.force is not None:
                a.extend(.8*j);b.extend(.8*(p.force-f.ravel()))
                n_gradient+=1
            # Hessian-vector labels cover the unstable/soft direction, middle
            # bends and high-frequency stretches. Each vector constrains the
            # full Cartesian response, including off-diagonal mode couplings.
            eig,vec=np.linalg.eigh(p.hessian)
            chosen=np.unique(np.r_[0,np.flatnonzero(eig>.2)[::8],len(eig)-1])
            for axis in chosen:
                u=vec[:,axis].reshape(p.r.shape)
                deriv=[];step=.001
                for sign in (-1,1):
                    r=p.r+sign*step*u
                    _,jac=design(p.z,r,True)
                    deriv.append((jac,model.predict(p.z,r,0,2)[1].ravel()))
                dx=(deriv[0][0]-deriv[1][0])/(2*step)
                df=(deriv[0][1]-deriv[1][1])/(2*step)
                target=p.hessian@u.ravel()-df
                if split=="train":
                    a.extend(.04*dx);b.extend(.04*target);n_curv+=1
                else:
                    dev_curv.append((dx,target))
        x=stoich@X; y=stoich@y
        blocks[tag]=(x,y)
        if split=="train":
            a.extend(x);b.extend(y)
    a,b=np.array(a),np.array(b)
    # Sparse local centers for absent elements can be dropped exactly. Compute
    # one Gram matrix and select regularization solely on grouped development paths.
    active=np.any(a!=0,axis=0); aa=a[:,active]
    gram=aa.T@aa; rhs=aa.T@b
    def metrics(w,split):
        err=np.array([(x@w-y)*EV_TO_KCAL for k,(x,y) in blocks.items() if roles[k]==split])
        return dict(reaction=score(err[:,2]),barrier_span=score(err[:,:2].ravel()))
    trials=[];sol=[]
    for ridge in (None,.01,.1,1.,10.,100.):
        w=np.zeros(a.shape[1])
        if ridge is not None:
            reg=gram.copy();reg.flat[::len(reg)+1]+=ridge
            w[active]=cho_solve(cho_factor(reg),rhs)
        dev=metrics(w,"development")
        curvature=float(np.sqrt(np.mean([(dx@w-target)**2 for dx,target in dev_curv])))
        objective=sum(v["mae_kcal_mol"]/5 for v in dev.values())+curvature/2
        trials.append(dict(ridge=ridge,dev=dev,curvature_rmse_eV_A2=curvature,objective=objective))
        sol.append(w)
    best=min(range(len(sol)),key=lambda i:trials[i]["objective"]);w=sol[best]
    size=model.chemical.size
    model.chemical.weights[0,2]+=w[:size]
    model.matern.weights[0,2]+=.25*w[size:]
    model.electronic.target+="; neutral doublet locally refined to C5H9O3 M06-2X-D3zero/aug-cc-pVTZ IRC energies, gradients and Hessians"
    blob=BytesIO();model.save(blob);payload=blob.getvalue()
    name="isoprene-"+sha256(payload).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(payload)
    restored=ForceField.load(OUT/name,dispersion="d3")
    result={}
    for k,v in external.items():
        row=result[k]=validate(restored,Path(v["folder"]))
        print(k,row["error_kcal_mol"],"negative directions",len(row["model_negative_modes_cm1"]),flush=True)
    for tag,(x,y) in blocks.items():
        es=[restored.energy(p.z,p.r,0,2)-p.energy for p in paths[tag]]
        np.testing.assert_allclose(stoich@es,x@w-y,atol=2e-7)
    report=dict(checkpoint=name,parent=parent,roles=roles,similarity_cutoff_A=.05,
                clusters=[sorted(c) for c in clusters],training_gradients=n_gradient,
                training_hessian_directions=n_curv,selected=best,trials=trials,
                control=metrics(sol[0],"test"),test=metrics(w,"test"),paths=result,
                reference="M06-2X-D3zero/aug-cc-pVTZ",scope="within C5H9O3 1,6-H-shift conformers")
    (OUT/"isoprene_irc_fit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["test"],"before",report["control"],flush=True)


if __name__=="__main__":
    main()
