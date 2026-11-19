"""Profile nonlinear local-kernel scales against geometry, frequencies and thermochemistry.

Fit a linear readout for every feature setting on training labels only. Use whole
TS formulas for reaction splits and ethanol as the predefined development minimum.
The fixed reaction panel, water/ammonia spectra and larger CCCBDB panel are only
evaluated after choosing the checkpoint. No quantum calculations are launched.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, eigh

from experiments.reactivity.frequency_validation import measure, vibrations, TO_CM1
from experiments.reactivity.spin_data import Geometry, reactions
from experiments.reactivity.spin_fit import BACKBONE, KINDS, OUT, TARGET, partition
from experiments.reactivity.stationary_fit import MASSES, POINTS
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, score
from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField
from src.local import descriptor_array, descriptor_layout
from src.matern import LocalMatern


def basis(geoms, species, max_centers=48):
    """Use only training sites for dimension reduction, centers and derivative directions."""
    nr = 2
    options = dict(species=species, nr=nr, lmax=2, ron=1.9, rc=3.2, inverse_radius=True)
    rows = {s:[] for s in species}
    for g in geoms:
        for i,s in enumerate(g.z):
            rows[s].append(descriptor_array(g.r, g.z, i, **options)[0])
    projections, centers, tangents, dims = {},{},{},{}
    width = sum(descriptor_layout(len(species), nr, 2)[0])
    for s, raw in rows.items():
        raw = np.array(raw).reshape(-1,width)
        blocks = []
        # Radial and angular components retain separate length scales. The variance
        # floor limits how much rare training environments can magnify derivatives.
        for block in (raw[:,:len(species)*nr], raw[:,len(species)*nr:]):
            transform = np.empty((block.shape[1], 0))
            if len(block)>1:
                centered = block - block.mean(axis=0)
                vals, vecs = eigh(centered.T @ centered / len(block))
                keep = np.flatnonzero(vals > 1e-7)[-12:]
                if len(keep):
                    transform = vecs[:,keep] / np.sqrt((vals[keep]+.02)*len(keep))
            blocks.append(transform)
        p = block_diag(*blocks)
        projections[s], dims[s] = p, blocks[0].shape[1]
        x = raw @ p
        # Farthest-point sampling represents chemically uncommon neighborhoods too.
        ids = []
        dist = np.full(len(x), np.inf)
        pick = 0
        for _ in range(min(max_centers,len(x))):
            ids.append(pick)
            ds = np.sum((x-x[pick])**2, axis=1)
            dist = np.minimum(dist, ds)
            pick = int(dist.argmax())
            if dist[pick] < 1e-8:
                break
        c = x[ids]
        dirs = np.zeros_like(c)
        for j, site in enumerate(c):
            ds = np.sum((x-site)**2,axis=1)
            candidates = np.flatnonzero(ds>1e-6)
            if len(candidates):
                nearest = candidates[ds[candidates].argmin()]
                dirs[j] = (x[nearest]-site)/np.sqrt(ds[nearest])
        centers[s], tangents[s] = c, dirs
    return LocalMatern(species, projections, centers, tangents, dims, 1., 1.)


@dataclass
class Observation:
    geometry: Geometry
    sites: dict
    # Directions project forces to a handful of labelled derivatives. An empty
    # array indicates an ordinary energy observation.
    directions: np.ndarray | None


def experiment(*, extended=False, compare=False):
    start = perf_counter()
    model = ForceField.load(OUT/BACKBONE, dispersion="d3")
    species = model.pair.species
    geoms, records, _ = reactions(model.charge_offsets)
    splits = [partition(geoms[rec.ts].z) for rec in records]
    def keys(rec):
        return [rec.ts,*rec.reactants,*rec.products]
    training = sorted({k for rec,s in zip(records,splits) if s=="train" for k in keys(rec)})
    states = sorted({geoms[k].state for k in training})
    points = json.loads(POINTS.read_text())
    minima = {}
    for label, pt in points.items():
        z,mass = np.array([MASSES[a] for a in pt["atoms"]]).T
        minima[label] = (Geometry(z.astype(int), np.array(pt["positions_A"]), (0,1)),
                         mass,np.array(pt["frequencies_cm1"]))
    ker = basis([geoms[k] for k in training]+[g for k,(g,_,_) in minima.items()
                                               if points[k]["split"]=="train"],species)
    model.electronic = ElectronicCorrection(species,states,
                        np.zeros((len(states),ElectronicFeatures(species).size)), model.target)
    print("Training basis centers", {str(s):len(c) for s,c in ker.centers.items()},flush=True)
    snapshots, equations, force_equations, baseline, refs, row_splits, kinds = [],[],[],[],[],[],[]
    # Every observable is a linear combination of energy or projected force rows
    # at fixed geometries. Geometry-dependent projections are cached before tuning.
    def snap(g, directions=None):
        sites = ker.project(g.z,g.r,gradient=directions is not None)
        if directions is not None:
            sites = {s:(x,j @ directions.T) for s,(x,j) in sites.items()}
        snapshots.append(Observation(g,sites,directions))
        return len(snapshots)-1
    idx, prior = {},{}
    for rec, split in zip(records,splits):
        for k in keys(rec):
            if k not in idx:
                g = geoms[k]
                idx[k] = snap(g)
                prior[k] = model.energy(g.z,g.r,*g.state)
        for kind, eq, ref in zip(KINDS,rec.equations(),rec.reference,strict=True):
            equations.append([(idx[k],c) for k,c in eq])
            refs.append(ref)
            baseline.append(sum(c*prior[k] for k,c in eq))
            row_splits.append(split)
            kinds.append(kind)
    auxiliary_energies = []
    auxiliary_counts = Counter()
    if extended:
        import csv
        from experiments.reactivity.gmtkn_states import DATA, dataset
        # These are separate high-level relative-energy observations. Only Eref
        # enters this calibration. dataset() protects the fixed, development and
        # spin holdout geometries before any supplemental observation is added.
        extra, rows, _ = dataset(model.charge_offsets)
        with (DATA/"GMTKN55_PBE0/d2cdft.csv").open() as f:
            high = {(r["setname"],int(r["rxnidx"])):float(r["Eref"])
                    for r in csv.DictReader(f) if r["calctype"]=="PBE0"}
        for family,rid,split,eq,_ in rows:
            if split!="train" or any(extra[k].state not in states for k,_ in eq):
                continue
            terms=[]
            energy=0.
            for k,c in eq:
                label="GMTKN/"+k
                if label not in idx:
                    g=extra[k]
                    idx[label]=snap(g)
                    prior[label]=model.energy(g.z,g.r,*g.state)
                terms.append((idx[label],c))
                energy+=c*prior[label]
            # Avoid overweighting atomizations with many separated atoms.
            weight = .8/np.sqrt(max(1.,sum(abs(c) for _,c in eq)/3))
            auxiliary_energies.append((terms,high[family,rid]/EV_TO_KCAL-energy,weight))
            auxiliary_counts["GMTKN_high_level"]+=1
    rng = np.random.default_rng(6151)
    # Minima and saddles in the reaction training sets have zero nuclear force
    # at their respective geometry-optimization level. Use as soft observations.
    for key in training:
        g = geoms[key]
        f = model.predict(g.z,g.r,*g.state)[1].ravel()
        if np.linalg.norm(f)>1e-6:
            dirs = np.stack([f,rng.normal(size=len(f))])
            dirs /= np.linalg.norm(dirs,axis=1)[:,None]
            i = snap(g,dirs)
            force_equations.append(([(i,1.)],-dirs@f,.12))
    if extended:
        from experiments.reactivity.augmented import qm_partition
        from experiments.reactivity.radical_curvatures import curvature_samples
        from experiments.reactivity.stationary_fit import internal_transform
        from copy import copy
        bare = copy(model)
        bare.correction = None
        qm,_ = qm_partition()
        subset = rng.choice(len(qm),320,replace=False)
        for ix in subset:
            row = qm[ix]
            g = Geometry(row.z,row.r,(row.charge,row.multiplicity))
            residual = (row.force-bare.predict(g.z,g.r,*g.state)[1]).ravel()
            if np.linalg.norm(residual)<1e-8:
                continue
            direction = np.vstack((residual,row.force.ravel(),rng.normal(size=(2,len(residual)))))
            direction /= np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-12)
            force_equations.append(([(snap(g,direction),1.)],direction@residual,.3))
            auxiliary_counts["QM7X_bare_PBE0_forces"]+=1
        held = {tuple(sorted(geoms[k].z)) for rec,s in zip(records,splits)
                if s!="train" for k in keys(rec)}
        # The Grambow Hessians have a different electronic level. Only derivative
        # constraints enter this high-level variant. The bare checkpoint is frozen.
        for name,z,xyz,state,hessian in curvature_samples(held,limit=48):
            g=Geometry(z,xyz,state)
            u=internal_transform(np.ones(len(z)),xyz)
            _,v=eigh(u.T@hessian@u)
            modes=(u@v).T
            dirs=modes[[0,len(modes)//2,-1]]
            force = model.predict(z,xyz,*state)[1].ravel()
            force_equations.append(([(snap(g,dirs),1.)],-dirs@force,.18))
            for v in dirs:
                mode=v.reshape(xyz.shape)
                gp=Geometry(z,xyz+.001*mode,state)
                gm=Geometry(z,xyz-.001*mode,state)
                ip=snap(gp,v[None,:])
                im=snap(gm,v[None,:])
                fp=model.predict(z,gp.r,*state)[1].ravel()
                fm=model.predict(z,gm.r,*state)[1].ravel()
                rhs=v@hessian@v-v@(fm-fp)/.002
                force_equations.append(([(im,500.),(ip,-500.)],np.array([rhs]),.08))
            auxiliary_counts["doublet_wB97XD3_Hessians"]+=1
    for label,(g,mass,freq) in minima.items():
        if points[label]["split"] != "train":
            continue
        f = model.predict(g.z,g.r,*g.state)[1].ravel()
        force_equations.append(([(snap(g,np.eye(len(f))),1.)],-f,1.))
        # Public records give eigenvalues. Use the backbone eigenvectors only as
        # approximate mode assignments for training. Development uses exact Hessian
        # eigenvalues at the fully relaxed geometry, with no mode-vector assumption.
        f0, modes = vibrations(model,g.z,g.r,mass,.001)
        for mode, refnu, nu in zip(modes,freq,f0,strict=True):
            h = .001
            u = mode.ravel()[None,:]
            # H = -dF/dR. The Cartesian mode here has unit mass-weighted norm.
            ip = snap(Geometry(g.z,g.r+h*mode,g.state),u)
            im = snap(Geometry(g.z,g.r-h*mode,g.state),u)
            target = (refnu**2 - np.sign(nu)*nu**2)/TO_CM1**2
            force_equations.append(([(im,.5/h),(ip,-.5/h)],np.array([target]),.20))
    force_equations.extend(auxiliary_energies)
    refs, baseline = np.array(refs),np.array(baseline)
    y = refs/EV_TO_KCAL - baseline
    train, dev, test = (np.array(row_splits)==k for k in ("train","development","test"))
    rx = np.array(kinds)=="reaction"
    electronic_cache, kernel_cache = {},{}
    def observations(features, local):
        basis_rows=[]
        for obs in snapshots:
            g, dirs = obs.geometry, obs.directions
            if local:
                e,f = features.design(obs.sites, len(dirs) if dirs is not None else 0)
                val = e[None,:] if dirs is None else f
            else:
                e,f = features.design(g.z,g.r)
                val = e[None,:] if dirs is None else dirs @ f
                # State interactions receive a stronger ridge prior than shared terms.
                block = np.zeros((len(val),len(states)+1,features.size))
                block[:,0] = val
                block[:,1+states.index(g.state)] = .35*val
                val = block.reshape(len(val),-1)
            basis_rows.append(val)
        e = np.vstack([sum(c*basis_rows[i] for i,c in terms) for terms in equations])
        f = np.vstack([w*sum(c*basis_rows[i] for i,c in terms) for terms,_,w in force_equations])
        return e, f
    force_y = np.concatenate([np.atleast_1d(w*target) for _,target,w in force_equations])
    def assemble(width,length,angle,family):
        features = ElectronicFeatures(species,radial_width=width)
        if width not in electronic_cache:
            electronic_cache[width]=observations(features,False)
        ker.length,ker.angular,ker.family = length,angle,family
        if (family,length,angle) not in kernel_cache:
            kernel_cache[family,length,angle]=observations(ker,True)
        er,ef = electronic_cache[width]
        kr,kf = kernel_cache[family,length,angle]
        return features,np.column_stack((er,kr)),np.column_stack((ef,kf))
    def attach(w,features,width):
        ne = (len(states)+1)*features.size
        ew = w[:ne].reshape(-1,features.size)
        model.electronic = ElectronicCorrection(species,states,ew[0]+.35*ew[1:],TARGET,radial_width=width)
        ker.weights = {key:w[ne:] for key in states}
        model.matern = ker
    def summary(values, subset):
        return {k:score((values-refs)[subset & mask]) for k,mask in (("reaction",rx),("barrier",~rx))}
    # A bounded profile search. Each point independently solves for every linear
    # coefficient. Save both the default-scale control and the selected trial.
    configs = [(1.,1.,.25),(.5,1.,.25),(2.,1.,.25),(1.,.5,.25),(1.,1.6,.25),
               (1.,1.,.17),(1.,1.,.36),(.5,.5,.17),(2.,.5,.36),
               (2.,1.6,.17),(.5,1.6,.36)]
    if compare:
        # Compare each positive-definite kernel at exactly the same nonlinear
        # settings and select by the same four-observable development objective.
        scales = [(1.,1.,.25),(1.,1.6,.25),(.5,.5,.17),(2.,.5,.36)]
        configs = [(length,angle,width,family) for length,angle,width in scales
                   for family in ("matern52","double_matern52","gaussian")]
    else:
        configs = [(*c,"matern52") for c in configs]
    trials, solutions = [],[]
    for length,angle,width,family in configs:
        ef, x, locks = assemble(width,length,angle,family)
        a = np.vstack((x[train], locks))
        active = np.any(a!=0.,axis=0)
        aa = a[:,active]
        b = np.r_[y[train],force_y]
        # Primal/dual normal equations are positive definite under ridge. Choose
        # the smaller side for this few-hundred-reaction experiment.
        primal = aa.shape[1] < aa.shape[0]
        gram = aa.T@aa if primal else aa@aa.T
        candidates=[]
        for ridge in (.03,.3,3.):
            reg = gram.copy()
            reg.flat[::len(reg)+1] += ridge
            sol = cho_solve(cho_factor(reg),aa.T@b if primal else b)
            w=np.zeros(x.shape[1])
            w[active] = sol if primal else aa.T@sol
            dev_e = summary((baseline+x@w)*EV_TO_KCAL,dev)
            energy_loss = sum(v["mae_kcal_mol"] for v in dev_e.values())/2
            candidates.append((energy_loss,ridge,w,dev_e))
        _,ridge,w,energy_dev = min(candidates,key=lambda item:item[0])
        attach(w,ef,width)
        g, mass, refnu = minima["ethanol"]
        try:
            eq = measure(model,g.z,mass,g.r,refnu)
            geom_loss = .5*(eq["bond_mae_A"]/.03+eq["angle_mae_deg"]/3.)
            loss = (sum(v["mae_kcal_mol"]/10 for v in energy_dev.values())
                    + geom_loss + eq["mae_cm1"]/100)/4
            geometry = {k:eq[k] for k in ("bond_mae_A","angle_mae_deg","mae_cm1","aligned_rmsd_A")}
        except RuntimeError as exc:
            geometry = {"failure":str(exc)}
            loss = None
        trial=dict(kernel=family,length=length,angular_scale=angle,radial_width_A=width,ridge=ridge,
                   dev_energies=energy_dev,dev_ethanol=geometry,selection_loss=loss)
        trials.append(trial)
        solutions.append(w)
        print("profile",len(trials),"/",len(configs),length,angle,width,"dev",
              [round(v["mae_kcal_mol"],2) for v in energy_dev.values()],
              "ethanol",geometry,"loss",loss,flush=True)
    winner = min((i for i,t in enumerate(trials) if t["selection_loss"] is not None),
                 key=lambda i:trials[i]["selection_loss"])
    results = {}
    frozen = [("default",0)]
    if compare:
        for family in ("matern52","double_matern52","gaussian"):
            valid = [i for i,t in enumerate(trials) if t["kernel"]==family and t["selection_loss"] is not None]
            if valid:
                frozen.append((family,min(valid,key=lambda i:trials[i]["selection_loss"])))
    frozen.append(("selected",winner))
    for label,i in frozen:
        t = trials[i]
        ef,x,_ = assemble(t["radial_width_A"],t["length"],t["angular_scale"],t["kernel"])
        w=solutions[i]
        attach(w,ef,t["radial_width_A"])
        pred=(baseline+x@w)*EV_TO_KCAL
        results[label]=dict(trial=i,test=summary(pred,test))
    buf=BytesIO()
    model.save(buf)
    data=buf.getvalue()
    checkpoint="matern-"+sha256(data).hexdigest()[:16]+".npz"
    (OUT/checkpoint).write_bytes(data)
    # Check the on-disk deployment path against predictions from the linear design.
    deployed = ForceField.load(OUT/checkpoint,dispersion="d3")
    energy_cache={}
    for rec in records:
        for k in keys(rec):
            if k not in energy_cache:
                g=geoms[k]
                energy_cache[k]=deployed.energy(g.z,g.r,*g.state)
    actual=np.array([sum(c*energy_cache[k] for k,c in eq)*EV_TO_KCAL for rec in records for eq in rec.equations()])
    np.testing.assert_allclose(actual,pred,atol=1e-6,rtol=1e-10)
    report=dict(checkpoint=checkpoint,backbone=BACKBONE,dispersion="d3",fit_seconds=perf_counter()-start,
                selected_trial=winner,selection="equal normalized development terms for reactions, barriers, ethanol geometry and spectrum",
                reaction_groups=Counter(splits),auxiliary_labels=auxiliary_counts,training_curvature_modes=sum(len(v[2]) for k,v in minima.items()
                if points[k]["split"]=="train"),trials=trials,results=results,fixed_singlet_panel=assess(deployed))
    name = "kernels_fit.json" if compare else ("matern_extended_fit.json" if extended else "matern_fit.json")
    (OUT/name).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(checkpoint,results,"seconds",report["fit_seconds"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extended",action="store_true")
    p.add_argument("--compare-kernels",action="store_true")
    args=p.parse_args()
    experiment(extended=args.extended,compare=args.compare_kernels)
