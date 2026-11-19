"""Learn a local doublet surface from IRC profiles, Cartesian gradients and Hessians.

Group entire conformer paths. Ten paths overlapping the external NEBs are
unavailable to the fit. Use leave-one-group-out development on the remaining
groups, then refit the selected regularization on those groups together. Report
the scores from each omitted group AND the three external archived NEB paths.
"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor,cho_solve

from experiments.reactivity.isoprene_irc_fit import OUT, ROOT, fingerprint, path_data
from experiments.reactivity.radical_validation import frames,validate
from experiments.reactivity.thermochemistry import EV_TO_KCAL,score
from src.forcefield import ForceField
from src.chemical import ChemicalEnvironment
from src.units import HARTREE_EV


def run(*,conformers=False):
    paths,external,clusters,roles=path_data()
    groups=[sorted(c) for c in clusters if all(roles[k]!="overlap_with_NEB" for k in c)]
    parent_report="rdb7_chemical_residual.json" if conformers else "peroxy_transfer_bridges.json"
    parent=json.loads((OUT/parent_report).read_text())["checkpoint"]
    model=ForceField.load(OUT/parent,dispersion="d3")
    if conformers:
        old=model.chemical
        model.chemical=ChemicalEnvironment(schema="smooth-oxygen-groups/3")
        model.chemical.weights={s:np.pad(w,(0,model.chemical.size-old.size)) for s,w in old.weights.items()}
    n_chem=model.chemical.size
    # Only H/C/O basis columns can be nonzero on this composition. Skip all
    # other columns when accumulating the small per-conformer Gram matrices.
    indices=np.concatenate([np.arange(model.matern.size)[model.matern.slices[s]] for s in (1,6,8)])
    if conformers:
        indices=np.array([],dtype=int)
    def design(z,r,jac=False):
        v,d=model.chemical.design(z,r,gradient=jac)
        if conformers:
            return v,d
        x,j=model.matern.design(model.matern.project(z,r,gradient=jac),r.size if jac else 0)
        return np.r_[v,.25*x[indices]],np.column_stack((d,.25*j[:,indices])) if jac else None
    all_frames={}; signatures={}; ext=[]
    for ref in external.values():
        p=Path(ref["folder"])
        for z,r,_ in frames(next(p.glob("*_MEP_trj.xyz"))):
            ext.append(fingerprint(z,r))
        z,r,_=next(frames(next(p.glob("*_NEB-TS_converged.xyz"))))
        ext.append(fingerprint(z,r))
    for i,group in enumerate(groups):
        signature=[]
        for tag in group:
            p=ROOT/"IRC_from_TS-Opt_only_realTSs_finestep_morestep"/tag
            traj=list(frames(next(p.glob("*_IRC_Full_trj.xyz"))))
            all_frames[tag]=traj
            signature.extend(fingerprint(z,r) for z,r,_ in traj)
            # Verify matching energy, geometry and atom order at the two ends
            # and the saddle before using any intervening IRC energy.
            for point in paths[tag]:
                nearest=min(traj,key=lambda frame:np.linalg.norm(frame[1]-point.r))
                np.testing.assert_array_equal(point.z,nearest[0])
                np.testing.assert_allclose(nearest[1],point.r,atol=3e-5)
                np.testing.assert_allclose(float(nearest[2].split()[-1]),point.energy/HARTREE_EV,
                                           atol=2e-6,rtol=0.)
        signatures[i]=np.array(signature)
    grams=[]; targets=[]; validation=[]; curve_info=[]; ngrad=0; total_directions=0
    stoich=np.array([[-1,1,0],[0,1,-1],[-1,0,1]])
    for i,group in enumerate(groups):
        # Avoid learning a profile image which is nearly the same as one on
        # another development group or an external validation path.
        avoid=np.vstack([ext,*[s for k,s in signatures.items() if k!=i]])
        width=n_chem+len(indices)
        gram=np.zeros((width,width));rhs=np.zeros(width);measurements=[];count=0
        for tag in group:
            a=[];b=[];site=[];error=[]
            for p in paths[tag]:
                row,j=design(p.z,p.r,True)
                en,f=model.predict(p.z,p.r,0,2)
                site.append(row);error.append(p.energy-en)
                if p.force is not None:
                    a.extend(.8*j);b.extend(.8*(p.force-f.ravel()));ngrad+=1
                # Build the COMPLETE Cartesian energy Hessian from central
                # differences of analytic forces. Each derivative row provides
                # the Hessian action for one Cartesian coordinate displacement.
                # Normalize its aggregate weight to the earlier 8-direction fit.
                phi_scale=.04*np.sqrt(8/p.r.size)
                for column,u in enumerate(np.eye(p.r.size).reshape(-1,*p.r.shape)):
                    step=.001
                    plus=p.r+step*u;minus=p.r-step*u
                    jplus=design(p.z,plus,True)[1]
                    jminus=design(p.z,minus,True)[1]
                    fplus=model.predict(p.z,plus,0,2)[1]
                    fminus=model.predict(p.z,minus,0,2)[1]
                    dx=(jminus-jplus)/(2*step)
                    y=p.hessian[:,column]-(fminus-fplus).ravel()/(2*step)
                    a.extend(phi_scale*dx);b.extend(phi_scale*y)
                    total_directions+=1
            x=stoich@site;y=stoich@error
            a.extend(x);b.extend(y)
            measurements.append((x,y))
            traj=all_frames[tag]
            accepted=[]
            for idx in np.linspace(1,len(traj)-2,24,dtype=int):
                z,r,comment=traj[idx]
                sig=fingerprint(z,r)
                if np.min(np.mean((avoid-sig)**2,axis=1))<.05**2:
                    continue
                e=model.energy(z,r,0,2)
                row,_=design(z,r)
                delta=float(comment.split()[-1])*HARTREE_EV-e
                accepted.append((row-site[1],delta-error[1]))
            if accepted:
                # A conformer should get the same total profile weight regardless
                # of how many unique intermediate images were available for it.
                scale=np.sqrt(3/len(accepted))
                a.extend(scale*xx for xx,_ in accepted);b.extend(scale*yy for _,yy in accepted)
                count+=len(accepted)
            a,b=np.array(a),np.array(b)
            gram+=a.T@a;rhs+=a.T@b
        grams.append(gram);targets.append(rhs);validation.append(measurements)
        curve_info.append(count)
        print("complete curvature and profiles for group",i+1,len(group),"conformers",count,"IRC images",flush=True)
    def solve(held,ridge,n_columns):
        keep=[i for i in range(len(groups)) if i!=held]
        gram=sum(grams[i][:n_columns,:n_columns] for i in keep)
        rhs=sum(targets[i][:n_columns] for i in keep)
        gram.flat[::len(gram)+1]+=ridge
        w=np.zeros(width)
        w[:n_columns]=cho_solve(cho_factor(gram),rhs)
        return w
    trials=[]
    sizes=[width]
    if conformers:
        sizes=[ChemicalEnvironment(schema="smooth-oxygen-groups/2").size,width]
    for columns,ridge in ((size,r) for size in sizes for r in (.01,.1,1.,10.,100.)):
        per_group=[]
        for i,rows in enumerate(validation):
            w=solve(i,ridge,columns)
            err=np.array([(x@w-y)*EV_TO_KCAL for x,y in rows])
            per_group.append(dict(reaction=score(err[:,2]),barrier_span=score(err[:,:2].ravel())))
        loss=float(np.mean([v["mae_kcal_mol"] for g in per_group for v in g.values()]))
        trials.append(dict(ridge=ridge,columns=columns,held_groups=per_group,mean_group_MAE=loss))
    best=min(trials,key=lambda t:t["mean_group_MAE"])
    w=solve(None,best["ridge"],best["columns"])
    model.chemical.weights[0,2]+=w[:n_chem]
    model.matern.weights[0,2][indices]+=.25*w[n_chem:]
    model.electronic.target+="; neutral doublet refined on C5H9O3 IRC profiles and full M06-2X-D3zero/aug-cc-pVTZ Hessians"
    blob=BytesIO();model.save(blob);data=blob.getvalue()
    name="isoprene-"+sha256(data).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(data)
    restored=ForceField.load(OUT/name,dispersion="d3")
    result={}
    for k,ref in external.items():
        path=result[k]=validate(restored,Path(ref["folder"]))
        print(k,path["error_kcal_mol"],"negative directions",len(path["model_negative_modes_cm1"]),flush=True)
    for i,group in enumerate(groups):
        for tag,(x,y) in zip(group,validation[i],strict=True):
            es=[restored.energy(p.z,p.r,0,2)-p.energy for p in paths[tag]]
            np.testing.assert_allclose(stoich@es,x@w-y,atol=2e-7)
    report=dict(checkpoint=name,parent=parent,groups=groups,roles=roles,n_intermediate=curve_info,
                n_gradients=ngrad,n_hessian_directions=total_directions,selected=best,trials=trials,paths=result)
    output="isoprene_trajectory_fit.json"
    if conformers:
        output="isoprene_chemical_conformers.json"
        report["selection_by_feature_set"]={str(size):min((t for t in trials if t["columns"]==size),
                    key=lambda t:t["mean_group_MAE"]) for size in sizes}
    (OUT/output).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformers",action="store_true",help="Compare chemistry-only fits with and without H bonds and torsions")
    run(conformers=parser.parse_args().conformers)
