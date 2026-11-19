"""Test whether more electronic saddle-point data improve a frozen kernel basis.

Use nested subsets of RDB7's neutral singlet reactions. Keep both barrier
directions and the reaction energy together. Group by the complete transition
state formula across RDB7, BH9 and TRIP50. Test formula assignments and the
regularization are fixed before seeing RDB7 prediction errors.
"""
import argparse
from copy import copy
from hashlib import sha256
from io import BytesIO
import json
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.frequency_validation import measure, vibrations, TO_CM1
from experiments.reactivity.larger_vibrations import references
from experiments.reactivity.matern_fit import MASSES, OUT, POINTS, partition, reactions
from experiments.reactivity.rdb7 import dataset
from experiments.reactivity.spin_data import Geometry
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, panel, score
from src.electronic import ElectronicCorrection
from src.chemical import ChemicalEnvironment
from src.forcefield import ForceField
from src.matern import LocalMatern


def signature(g):
    """Ignore rotation, translation and atom order when identifying duplicate TSs."""
    i,j=np.triu_indices(len(g.z),1)
    d=np.linalg.norm(g.r[i]-g.r[j],axis=1)
    order=np.lexsort((d,np.maximum(g.z[i],g.z[j]),np.minimum(g.z[i],g.z[j])))
    return (tuple(sorted(g.z)),g.state,d[order].round(4).tobytes())


def expand_centers(kernel, geometries, limit):
    """Add training environments farthest from the existing Gaussian centers.

    Preserve the original feature projection, length scale, and coefficients.
    Extra columns start at zero for every electronic state, so the potential
    before calibration is identical. Only the neutral singlet is subsequently fit.
    """
    sites = {s:[] for s in kernel.species}
    for g in geometries:
        for s,(projected,_) in kernel.project(g.z, g.r, gradient=False).items():
            sites[s].append(projected)
    centers, tangents = dict(kernel.centers), dict(kernel.tangents)
    for s, blocks in sites.items():
        if not blocks:
            continue
        x = np.vstack(blocks)
        weight = np.ones(x.shape[1])
        weight[kernel.radial_dims[s]:] = kernel.angular
        scaled = x * weight
        nearest_d2 = np.full(len(x), np.inf)
        def distance(c):
            delta = scaled - c*weight
            return np.einsum("ij,ij->i", delta, delta)
        for c in centers[s]:
            np.minimum(nearest_d2, distance(c), out=nearest_d2)
        extra, directions = [], []
        for _ in range(limit):
            farthest = nearest_d2.argmax()
            if nearest_d2[farthest] < 1e-6:
                break
            c = x[farthest]
            d2 = distance(c)
            np.minimum(nearest_d2, d2, out=nearest_d2)
            # Coincident environments carry the same value and cannot define a
            # tangent. Use the nearest geometrically distinct training example.
            d2[d2 < 1e-6] = np.inf
            nearest = d2.argmin()
            tangent = x[nearest]-c if np.isfinite(d2[nearest]) else np.zeros_like(c)
            norm = np.linalg.norm(tangent)
            extra.append(c)
            directions.append(tangent/norm if norm else tangent)
        if extra:
            centers[s] = np.vstack((centers[s], extra))
            tangents[s] = np.vstack((tangents[s], directions))
    larger = LocalMatern(kernel.species,kernel.projections,centers,tangents,kernel.radial_dims,
                         kernel.length,kernel.angular,family=kernel.family)
    for state, old in kernel.weights.items():
        new = np.zeros(larger.size)
        for s in kernel.species:
            count = len(kernel.centers[s])
            column = larger.slices[s].start
            previous = old[kernel.slices[s]].reshape(2, count)
            new[column:column+count] = previous[0]
            column += len(centers[s])
            new[column:column+count] = previous[1]
        larger.weights[state] = new
    return larger


def public_splits(initial):
    """Recreate the fixed formula-held-out sets for all reaction-model comparisons."""
    gs, records, excluded=dataset()
    bgeoms,brecs,_=reactions(initial.charge_offsets)
    gs.update(bgeoms)
    protected={tuple(sorted(ELEMENTS[a] for a in p["atoms"])) for p in references().values()}
    for _,terms,_,_ in panel():
        for _,p in terms:
            protected.add(tuple(sorted(ELEMENTS[t.split()[0]] for t in p.read_text().splitlines()[2:] if t.strip())))
    seen=set();groups={"train":[],"development":[],"test":[]}
    for r in sorted(records,key=lambda r:sha256(r.name.encode()).digest()):
        if any(tuple(sorted(gs[k].z)) in protected for k in [r.ts,*r.reactants,*r.products]):
            excluded["fixed_or_larger_validation_formula"]+=1
            continue
        if any(gs[k].state not in initial.supported_states for k in [r.ts,*r.reactants,*r.products]):
            excluded["unsupported_state"]+=1
            continue
        sig=signature(gs[r.ts])
        if sig in seen:
            excluded["duplicate_transition_state"]+=1
            continue
        seen.add(sig)
        groups[partition(gs[r.ts].z)].append(r)
    print("eligible reaction groups",{k:len(v) for k,v in groups.items()},flush=True)
    # Draw once. The same 400 untouched reactions are used for all training sizes.
    counts=(250,1000,4000)
    sets={"RDB7_train":groups["train"][:counts[-1]],"RDB7_dev":groups["development"][:150],
          "RDB7_test":groups["test"][:400]}
    for split in groups:
        sets["BH9_TRIP50_"+split]=[r for r in brecs if partition(gs[r.ts].z)==split]
    return gs,sets,groups,excluded,counts


def run(*, balanced=False, added_centers=0, chemical=False, residual=False, schema=1):
    """Optionally select the RDB7 loss weight on energy, geometry and frequency development data."""
    start=perf_counter()
    parent_report = "rdb7_balanced_centers128.json" if residual else "kernels_fit.json"
    parent=json.loads((OUT/parent_report).read_text())["checkpoint"]
    initial=ForceField.load(OUT/parent,dispersion="d3")
    feature=initial.electronic.features
    kernel=initial.matern
    gs,sets,groups,excluded,counts=public_splits(initial)
    if added_centers:
        records_for_centers = sets["RDB7_train"][:400]
        ids = sorted({k for rec in records_for_centers for k in (rec.ts,*rec.reactants,*rec.products)})
        kernel = expand_centers(kernel,(gs[k] for k in ids),added_centers)
        initial.matern = kernel
        print("expanded kernel basis", {s:len(c) for s,c in kernel.centers.items()}, flush=True)
    prepared=perf_counter()
    core=copy(initial)
    core.matern=None
    core.electronic=ElectronicCorrection(core.pair.species,initial.supported_states,
                            np.zeros((len(initial.supported_states),feature.size)),core.target,
                            radial_width=feature.radial.width)
    original_width = 0 if residual else feature.size+kernel.size
    basis_name = "smooth-organic-groups/1" if schema==4 else f"smooth-oxygen-groups/{schema}"
    chemistry = ChemicalEnvironment(schema=basis_name) if chemical else None
    width = original_width + (3*chemistry.size if chemical else 0)
    cached={}
    def design(g,gradient=False):
        v = m = None
        row = np.zeros(original_width)
        jac = np.empty((g.r.size,0)) if gradient else None
        if not residual:
            v,j=feature.design(g.z,g.r)
            sites=kernel.project(g.z,g.r,gradient=gradient)
            m,k=kernel.design(sites,g.r.size if gradient else 0)
            if g.state==(0,1):
                row=np.r_[v,m]
            jac=np.column_stack((j,k)) if gradient else None
        if chemical:
            # One common chemistry surface plus regularized doublet and triplet
            # deviations. RDB7 supplies singlets; spin deviations require BH9/
            # TRIP50 training structures with the corresponding multiplicities.
            h, dh = chemistry.design(g.z,g.r,gradient=gradient)
            doublet = .5*(g.state[1]==2)
            triplet = .5*(g.state[1]==3)
            row = np.r_[row,h,doublet*h,triplet*h]
            if gradient:
                jac = np.column_stack((jac,dh,doublet*dh,triplet*dh))
        return row, (v,m), jac
    def molecular(g):
        key=(g.z.tobytes(),g.r.round(6).tobytes(),g.state)
        if key not in cached:
            row,(v,m),_=design(g)
            energy=(initial.energy(g.z,g.r,*g.state) if residual else
                    core.energy(g.z,g.r,*g.state) + v@initial.electronic.weights[g.state]
                    + m@kernel.weights[g.state])
            cached[key]=(row,energy)
        return cached[key]
    matrices={}
    for label,sub in sets.items():
        x=np.empty((3*len(sub),width));prior=[];target=[]
        for i,r in enumerate(sub):
            for j,eq in enumerate(r.equations()):
                x[3*i+j]=sum(c*molecular(gs[k])[0] for k,c in eq)
                prior.append(sum(c*molecular(gs[k])[1] for k,c in eq))
            target.extend(r.reference)
            if (i+1)%1000==0:
                print(label,i+1,"reaction groups",round(perf_counter()-start,1),"s",flush=True)
        prior=np.array(prior)*EV_TO_KCAL
        target=np.array(target)
        matrices[label]=(x,prior,target)
    assembled=perf_counter()
    # Supplement energy equations with force and frequency constraints at the
    # original nine training minima. Geometry and spectrum holdouts remain sealed.
    locks=[];rhs=[]
    for point in json.loads(POINTS.read_text()).values():
        if point["split"]!="train":
            continue
        z,mass=np.array([MASSES[a] for a in point["atoms"]]).T
        z=z.astype(int);r=np.array(point["positions_A"])
        g=Geometry(z,r,(0,1))
        locks.append(design(g,True)[2])
        rhs.extend(-initial.predict(z,r)[1].ravel())
        nu,modes=vibrations(initial,z,r,mass,.001)
        for v,refnu,n in zip(modes,point["frequencies_cm1"],nu,strict=True):
            up=design(Geometry(z,r+.001*v,(0,1)),True)[2]
            dn=design(Geometry(z,r-.001*v,(0,1)),True)[2]
            locks.append((.25*v.ravel()@(dn-up)/.002)[None,:])
            rhs.append(.25*(refnu**2-np.sign(n)*n**2)/TO_CM1**2)
    lock=np.vstack(locks);rhs=np.array(rhs)
    if residual:
        # These geometries are stationary at their source method. Penalize
        # additional force from the chemistry head there. This is a derivative
        # regularizer, not a claim that the archived CC energy has a zero
        # gradient on a geometry optimized with a different method.
        gradient_prior = []
        for rec in sets["BH9_TRIP50_train"] + sets["RDB7_train"][:128]:
            g = gs[rec.ts]
            _,dh = chemistry.design(g.z,g.r,gradient=True)
            if dh.any():
                row = np.zeros((g.r.size,width))
                row[:,original_width:] = np.column_stack((dh,.5*(g.state[1]==2)*dh,.5*(g.state[1]==3)*dh))
                gradient_prior.append(row/np.sqrt(g.r.size))
        lock = np.vstack([lock,*gradient_prior])
        rhs = np.r_[rhs,np.zeros(sum(len(row) for row in gradient_prior))]
    constrained=perf_counter()
    xa,pa,ya=matrices["BH9_TRIP50_train"]
    xb,pb,yb=matrices["RDB7_train"]
    def metrics(label,w):
        x,p,y=matrices[label]
        err=(p+x@w*EV_TO_KCAL-y).reshape(-1,3)
        return dict(reaction=score(err[:,2]),barrier=score(err[:,:2].ravel()))
    def solve(n, row_scale, *, include_chemistry=True, ridge=.3, chemical_limit=None):
        """Scaling a residual by sqrt(weight) scales its squared loss by weight."""
        w = np.zeros(width)
        if n:
            a = np.vstack((xa, row_scale*xb[:3*n], lock))
            b = np.r_[(ya-pa)/EV_TO_KCAL, row_scale*(yb[:3*n]-pb[:3*n])/EV_TO_KCAL, rhs]
            active = np.any(a!=0., axis=0)
            if not include_chemistry:
                active[original_width:] = False
            if chemical_limit is not None:
                active[original_width:].reshape(3,chemistry.size)[:,chemical_limit:] = False
            if residual:
                active[:original_width] = False
            if not active.any():
                return w
            aa = a[:, active]
            gram = aa.T@aa
            gram.flat[::len(gram)+1] += ridge
            w[active] = cho_solve(cho_factor(gram), aa.T@b)
        return w

    def checkpoint(w):
        model = ForceField.load(OUT/parent, dispersion="d3")
        if added_centers:
            # Give the trial its own coefficients. In-place updates to a shared
            # array here would accumulate weights across development trials.
            model.matern = copy(kernel)
            model.matern.weights = {state:v.copy() for state,v in kernel.weights.items()}
        if not residual:
            model.electronic.weights[0,1] += w[:feature.size]
            model.matern.weights[0,1] += w[feature.size:original_width]
        if chemical:
            common,doublet,triplet = w[original_width:].reshape(3,chemistry.size)
            model.chemical = ChemicalEnvironment({
                state:common + .5*(state[1]==2)*doublet + .5*(state[1]==3)*triplet
                for state in model.supported_states},schema=chemistry.schema)
        model.electronic.target = "BH9/TRIP50 and RDB7 electronic reaction and barrier differences, calibrated with D3(BJ)"
        return model

    curve = []
    solve_seconds = 0.
    if balanced:
        ethanol = json.loads(POINTS.read_text())["ethanol"]
        z, mass = np.array([MASSES[a] for a in ethanol["atoms"]]).T
        n = len(sets["RDB7_train"])
        n_bh = len(sets["BH9_TRIP50_train"])
        solutions = []
        # Ratios refer to the TOTAL loss of each reaction collection. Include
        # the existing checkpoint and the equal-per-row fit as endpoint controls.
        configs = [(ratio,.3) for ratio in (0., .1, .3, 1., 3., n/n_bh)]
        if residual:
            fixed_ratio = json.loads((OUT/parent_report).read_text())
            fixed_ratio = fixed_ratio["curve"][fixed_ratio["selected_trial"]]["total_RDB7_to_BH9_weight"]
            configs = [(0.,.3)] + [(fixed_ratio,ridge) for ridge in (.3,3.,30.,300.,3000.)]
        for ratio,ridge in configs:
            tick = perf_counter()
            w = solve(n, np.sqrt(ratio*n_bh/n), ridge=ridge) if ratio else np.zeros(width)
            solve_seconds += perf_counter() - tick
            solutions.append(w)
            dev = {k:metrics(k,w) for k in ("BH9_TRIP50_development","RDB7_dev")}
            error = sum(vals["mae_kcal_mol"] for group in dev.values() for vals in group.values())
            try:
                eq = measure(checkpoint(w), z.astype(int), mass, np.array(ethanol["positions_A"]),
                             np.array(ethanol["frequencies_cm1"]))
                geom = {k:eq[k] for k in ("bond_mae_A","angle_mae_deg","mae_cm1")}
                # Give reaction energies, barriers, geometry and frequencies one
                # term each. Each collection contributes equally to energy MAEs.
                loss = (error/20 + .5*(eq["bond_mae_A"]/.03+eq["angle_mae_deg"]/3)
                        + eq["mae_cm1"]/100)/4
            except RuntimeError as exc:
                geom, loss = {"failure":str(exc)}, None
            curve.append(dict(total_RDB7_to_BH9_weight=ratio, ridge=ridge, development=dev,
                              ethanol=geom, selection_loss=loss))
            print("development loss",ratio,loss,flush=True)
        valid = [i for i,row in enumerate(curve) if row["selection_loss"] is not None]
        selected = min(valid, key=lambda i:curve[i]["selection_loss"])
        w = solutions[selected]
        extra = dict(selected_trial=selected, test={k:metrics(k,w) for k in
                     ("RDB7_test","BH9_TRIP50_test")})
        if chemical:
            # Ablate only the new columns with identical data, kernel and loss
            # weights. Group labels are geometric diagnostics and may overlap.
            ratio = curve[selected]["total_RDB7_to_BH9_weight"]
            chosen_ridge = curve[selected]["ridge"]
            control = solve(n,np.sqrt(ratio*n_bh/n),include_chemistry=False,ridge=chosen_ridge) if ratio else np.zeros(width)
            extra["matched_control"] = {k:metrics(k,control) for k in extra["test"]}
            if schema!=1:
                twelve = solve(n,np.sqrt(ratio*n_bh/n),chemical_limit=ChemicalEnvironment().size,ridge=chosen_ridge) if ratio else control
                extra["matched_twelve_groups"] = {k:metrics(k,twelve) for k in extra["test"]}
            if schema==4:
                oxygen = solve(n,np.sqrt(ratio*n_bh/n),chemical_limit=chemistry.oxygen_size,ridge=chosen_ridge) if ratio else control
                extra["matched_oxygen_groups"] = {k:metrics(k,oxygen) for k in extra["test"]}
            extra["by_environment"] = {}
            extra["training_coverage"] = {}
            for split in extra["test"]:
                x,p,y = matrices[split]
                err = np.stack([(p+x@coef*EV_TO_KCAL-y).reshape(-1,3) for coef in (control,w)])
                # The first feature for a motif is its continuous count. Flag
                # a participating geometry with occupancy >=0.5 for reporting;
                # the energy/force calculation never uses this threshold.
                occupancy_groups = {key:sl for key,sl in chemistry.slices.items() if not key.startswith("torsion_")}
                indexes = [original_width+sl.start for sl in occupancy_groups.values()]
                present = []
                changed = []
                for reaction in sets[split]:
                    def counts_for(k):
                        return molecular(gs[k])[0][indexes]
                    side = np.array([sum(counts_for(k) for k in reaction.reactants),
                                     counts_for(reaction.ts),
                                     sum(counts_for(k) for k in reaction.products)])
                    present.append(side.max(axis=0)>=.5)
                    changed.append(np.ptp(side,axis=0)>=.5)
                result = {}
                for tag, masks in (("present",present),("changes",changed)):
                    for name,mask in zip(occupancy_groups,np.array(masks).T,strict=True):
                        rows = err[:,mask,:]
                        result[name+"_"+tag] = {key:dict(reaction=score(a[:,2]),barrier=score(a[:,:2].ravel()))
                            for key,a in zip(("control","chemical"),rows,strict=True)} if mask.any() else {"n":0}
                extra["by_environment"][split] = result
            for split in ("BH9_TRIP50_train","RDB7_train"):
                # Count reactions where the occupancy actually changes, by TS
                # multiplicity. Contacts present only in spectators cannot
                # constrain the corresponding reaction energy coefficient.
                indices = [original_width+sl.start for sl in occupancy_groups.values()]
                coverage = {}
                for rec in sets[split]:
                    def count(k):
                        return molecular(gs[k])[0][indices]
                    occupancy = np.array([sum(count(k) for k in rec.reactants),count(rec.ts),
                                          sum(count(k) for k in rec.products)])
                    spin = str(gs[rec.ts].state[1])
                    counts_for_spin = coverage.setdefault(spin,np.zeros(len(indices),dtype=int))
                    counts_for_spin += np.ptp(occupancy,axis=0)>=.5
                extra["training_coverage"][split] = {s:dict(zip(occupancy_groups,counts.tolist(),strict=True))
                                                      for s,counts in coverage.items()}
    else:
        for n in (0,*counts):
            n=min(n,len(sets["RDB7_train"]))
            tick = perf_counter()
            w = solve(n, 1.)
            solve_seconds += perf_counter() - tick
            scores={s:metrics(s,w) for s in matrices}
            curve.append(dict(n_RDB7_reactions=n,metrics=scores))
            print(n,"RDB7 test",scores["RDB7_test"],"BH9/TRIP50",scores["BH9_TRIP50_test"],flush=True)
        extra = {}
    model=checkpoint(w)
    buf=BytesIO();model.save(buf);data=buf.getvalue()
    name="rdb7-"+sha256(data).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(data)
    deployed=ForceField.load(OUT/name,dispersion="d3")
    for i in np.linspace(0,len(sets["RDB7_test"])-1,12,dtype=int):
        rec=sets["RDB7_test"][i]
        for j,eq in enumerate(rec.equations()):
            direct=sum(c*deployed.energy(gs[k].z,gs[k].r,*gs[k].state) for k,c in eq)*EV_TO_KCAL
            x,p,_=matrices["RDB7_test"];ix=3*i+j
            np.testing.assert_allclose(direct,p[ix]+x[ix]@w*EV_TO_KCAL,atol=2e-6)
    report=dict(checkpoint=name,backbone=parent,source="10.1038/s41597-022-01529-6",
                excluded=excluded,available={k:len(v) for k,v in groups.items()},
                sample={k:len(v) for k,v in sets.items()},curve=curve,seconds=perf_counter()-start,
                test_ids=[r.name for r in sets["RDB7_test"]],fixed_singlet_panel=assess(deployed),
                stage_seconds=dict(prepare=prepared-start, assemble_reactions=assembled-prepared,
                                   assemble_derivatives=constrained-assembled, linear_solves=solve_seconds),
                **extra)
    filename = "rdb7_balanced.json" if balanced else "rdb7_learning_curve.json"
    if added_centers:
        filename = filename.replace(".json",f"_centers{added_centers}.json")
        report["augmented_basis"] = dict(training_reactions=len(records_for_centers),
                                         extra_centers_per_element_limit=added_centers,
                                         resulting_centers={str(s):len(c) for s,c in kernel.centers.items()})
    if chemical:
        filename = filename.replace(".json","_chemical.json")
        report["chemical_basis"] = dict(schema=chemistry.schema, features=chemistry.size,
                                        groups=list(chemistry.slices))
    if residual:
        filename = "rdb7_chemical_residual.json"
        report["frozen_control"] = parent
    if schema!=1:
        filename = filename.replace(".json",f"_v{schema}.json")
    (OUT/filename).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", action="store_true", help="Select the relative collection weight on development data")
    parser.add_argument("--added-centers", type=int, default=0,
                        help="Maximum extra sparse centers per element from 400 training reactions")
    parser.add_argument("--chemical",action="store_true",help="Fit smooth functional groups with spin-dependent terms")
    parser.add_argument("--chemical-residual",action="store_true",help="Freeze the expanded control and fit only functional groups")
    parser.add_argument("--chemical-schema",type=int,choices=(1,2,3,4),default=1,
                        help="1–3: oxygen models; 4: broader organic chemistry including rings, nitrogen, sulfur, phosphorus and halogens")
    args = parser.parse_args()
    if args.added_centers < 0 or (args.added_centers and not args.balanced):
        parser.error("Positive --added-centers requires --balanced")
    if args.chemical and not args.balanced:
        parser.error("--chemical requires --balanced for the matched control")
    if args.chemical_residual and (not args.balanced or args.added_centers):
        parser.error("--chemical-residual needs --balanced and reuses the existing expanded centers")
    if args.chemical_schema!=1 and not (args.chemical or args.chemical_residual):
        parser.error("--chemical-schema requires --chemical or --chemical-residual")
    run(balanced=args.balanced,added_centers=args.added_centers,
        chemical=args.chemical or args.chemical_residual,residual=args.chemical_residual,schema=args.chemical_schema)
