"""Add conservative local kernels at changing sites of training reactions.

Each reaction class contributes saddle environments. Its tangent points toward
the nearest corresponding element on an endpoint. An atom map is unnecessary,
and the resulting single potential only needs coordinates, charge and spin.
Tune regularization, source balance and barrier-family balance on development
reactions. Retain independent stationary-geometry derivative constraints.
"""
from collections import Counter,defaultdict
from copy import copy
from hashlib import sha256
from io import BytesIO
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist

from experiments.reactivity.barrier_curve import public_splits, OUT, POINTS, MASSES, EV_TO_KCAL, score
from experiments.reactivity.reaction_classes import lookup,family
from src.forcefield import ForceField
from src.matern import LocalMatern


def center_basis(parent,rows,gs,labels,pooled_targets=None):
    classes=defaultdict(list)
    for row in rows:
        classes[labels[row.name] if pooled_targets is None else 'all'].append(row)
    sites=defaultdict(list)
    for label,group in sorted(classes.items()):
        # Include rare chemistry even if the RDB7 unassigned bin is enormous.
        pool=sorted(group,key=lambda r:sha256(r.name.encode()).digest())
        for row in (pool[:64] if pooled_targets is None else pool):
            def project(k):
                g=gs[k]
                return parent.project(g.z,g.r,gradient=False)
            ts=project(row.ts)
            ends=[project(k) for k in row.reactants+row.products]
            for s,(values,_) in ts.items():
                all_ends=np.vstack([d[s][0] for d in ends if s in d])
                scale=np.ones(values.shape[1]);scale[parent.radial_dims[s]:]=parent.angular
                dist=cdist(values*scale,all_ends*scale)
                j=dist.min(axis=1).argmax();k=dist[j].argmin()
                offset=all_ends[k]-values[j];norm=np.linalg.norm(offset)
                if norm>1e-5:
                    sites[s,label].append((values[j],offset/norm))
    centers={s:[] for s in parent.species};directions={s:[] for s in parent.species}
    for (s,label),items in sorted(sites.items()):
        x=np.array([item[0] for item in items]);scale=np.ones(x.shape[1]);scale[parent.radial_dims[s]:]=parent.angular
        prior=np.vstack([parent.centers[s],*centers[s]])
        nearest=cdist(x*scale,prior*scale).min(axis=1)
        limit=8 if pooled_targets is None else pooled_targets[str(s)]
        for _ in range(min(limit,len(items))):
            i=nearest.argmax()
            if nearest[i]<1e-4:break
            centers[s].append(x[i]);directions[s].append(items[i][1])
            nearest=np.minimum(nearest,np.linalg.norm((x-x[i])*scale,axis=1))
    for s in centers:
        d=parent.projections[s].shape[1]
        centers[s]=np.array(centers[s]).reshape(-1,d)
        directions[s]=np.array(directions[s]).reshape(-1,d)
    if pooled_targets is not None and any(len(c)!=pooled_targets[str(s)] for s,c in centers.items()):
        raise ValueError('Class-blind center count differs from the class-aware model')
    return LocalMatern(parent.species,parent.projections,centers,directions,parent.radial_dims,parent.length,parent.angular,family=parent.family)


def combine(parent,extra,states):
    """Concatenate value and tangent blocks in exactly the order used by inference."""
    centers={s:np.vstack((parent.centers[s],extra.centers[s])) for s in parent.species}
    tangents={s:np.vstack((parent.tangents[s],extra.tangents[s])) for s in parent.species}
    joined=LocalMatern(parent.species,parent.projections,centers,tangents,parent.radial_dims,parent.length,parent.angular,family=parent.family)
    for state,new in states.items():
        joined.weights[state]=np.concatenate([np.column_stack((
            parent.weights[state][parent.slices[s]].reshape(2,-1),
            new[extra.slices[s]].reshape(2,-1))).ravel() for s in parent.species])
    return joined


def by_state(v,m):
    return np.concatenate((v,v*.5*(m==2),v*.5*(m==3)),axis=-1)


def run(preserve_geometry=False,pooled=False):
    info=json.loads((OUT/'rdb7_chemical_residual_v4.json').read_text())
    ref=ForceField.load(OUT/info['checkpoint'],dispersion='d3')
    gs,sets,_,_,_=public_splits(ref)
    assert info['test_ids']==[r.name for r in sets['RDB7_test']]
    registry=lookup();labels={r.name:family(r,gs,registry) for rows in sets.values() for r in rows}
    target_counts=json.loads((OUT/'reaction_centers.json').read_text())['centers'] if pooled else None
    extra=center_basis(ref.matern,sets['BH9_TRIP50_train']+sets['RDB7_train'],gs,labels,target_counts)
    print('additional centers',{s:len(v) for s,v in extra.centers.items()},flush=True)
    def design(z,r,state,deriv=False):
        v,f=extra.design(extra.project(z,r,gradient=deriv),r.size if deriv else 0)
        return by_state(v,state[1]),by_state(f,state[1]) if deriv else None
    cache={};matrices={}
    for name,rows in sets.items():
        a=[];b=[];families=[]
        for row in rows:
            for eq in row.equations():
                for k,c in eq:
                    if k not in cache:
                        g=gs[k];cache[k]=design(g.z,g.r,g.state)[0],ref.energy(g.z,g.r,*g.state)
                a.append(sum(c*cache[k][0] for k,c in eq))
                b.append(sum(c*cache[k][1] for k,c in eq))
            families.append(labels[row.name])
        y=np.array([r.reference for r in rows]).ravel()/EV_TO_KCAL-np.array(b)
        matrices[name]=(np.array(a),y,np.array(families))
        print('assembled',name,len(rows),flush=True)
    lock=[]
    for item in json.loads(POINTS.read_text()).values():
        if item['split']!='train':continue
        z=np.array([MASSES[a][0] for a in item['atoms']],int);r=np.array(item['positions_A'])
        lock.append(design(z,r,(0,1),True)[1])
        # Preserve parent curvature along Cartesian axes at the training minima.
        for i in range(r.size):
            delta=np.zeros(r.size);delta[i]=.001;delta=delta.reshape(r.shape)
            deriv=(design(z,r+delta,(0,1),True)[1]-design(z,r-delta,(0,1),True)[1])/.002
            lock.append(.12*deriv/np.sqrt(r.size))
    for row in sets['BH9_TRIP50_train']+sets['RDB7_train'][:128]:
        g=gs[row.ts];lock.append(.7*design(g.z,g.r,g.state,True)[1]/np.sqrt(g.r.size))
    anchor_count=0
    if preserve_geometry:
        # Broad shape anchors prevent a sparsely supervised reaction correction
        # from changing the forces and curvature of ordinary bonded environments.
        # Keep all public test formulas and the PBE0 frequency panel out.
        import h5py
        from experiments.reactivity.augmented import QM
        from experiments.reactivity.barrier_curve import references as saved_minima, ELEMENTS
        protected={tuple(sorted(gs[k].z)) for split in ('RDB7_test','BH9_TRIP50_test') for row in sets[split]
                   for k in (row.ts,*row.reactants,*row.products)}
        protected.update(tuple(sorted(ELEMENTS[a] for a in point['atoms'])) for point in saved_minima().values())
        with h5py.File(QM) as source:
            for mol,group in source.items():
                k=next(k for k in sorted(group) if k.endswith('-opt'))
                z=group[k]['atNUM'][:]
                if tuple(sorted(z)) in protected:continue
                opt=group[k]['atXYZ'][:]
                siblings=[name for name in group if name.startswith(k[:-3]+'d')]
                if not siblings:continue
                sibling=min(siblings,key=lambda name:sha256(name.encode()).digest())
                disp=group[sibling]['atXYZ'][:]
                jo=design(z,opt,(0,1),True)[1];jd=design(z,disp,(0,1),True)[1]
                scale=1/np.sqrt(opt.size)
                lock.extend([2*scale*jo,2*scale*jd,.4*scale*(jd-jo)/max(.03,np.sqrt(np.mean((disp-opt)**2)))])
                anchor_count+=1
        print('QM7-X parent geometry anchors',anchor_count,flush=True)
    locks=np.vstack(lock);penalty=locks.T@locks
    def metrics(name,w,per_family=False):
        a,y,fam=matrices[name];errors=((a@w-y)*EV_TO_KCAL).reshape(-1,3)
        def stats(e):return dict(reaction=score(e[:,2]),barrier=score(e[:,:2].ravel()))
        return {s:stats(errors[fam==s]) for s in np.unique(fam)} if per_family else stats(errors)
    def loss(w):
        return sum(metric['mae_kcal_mol'] for s in ('BH9_TRIP50_development','RDB7_dev') for metric in metrics(s,w).values())
    candidates=[];options=[];zero=np.zeros(penalty.shape[0]);best=(loss(zero),zero,{'frozen_parent':True});grouped={}
    for balance in (False,True):
        systems=[]
        for name in ('BH9_TRIP50_train','RDB7_train'):
            a,y,fam=matrices[name]
            weights=np.ones((len(fam),3))
            if balance:
                counts=Counter(fam)
                g=np.array([1/np.sqrt(max(5,counts[k])) for k in fam]);weights[:,:2]=np.sqrt(g/g.mean())[:,None]
            x=a*weights.reshape(-1,1);rhs=y*weights.ravel()
            systems.append((x.T@x,x.T@rhs,len(fam)))
        local_best=None
        for ratio in (.1,.5,2.):
            ga,ya,na=systems[0];gb,yb,nb=systems[1]
            wgt=ratio*na/nb
            for ridge in (.3,3.,30.):
                gram=ga+wgt*gb+penalty
                gram.flat[::len(gram)+1]+=ridge
                w=cho_solve(cho_factor(gram),ya+wgt*yb)
                trial=dict(class_balanced=balance,ratio=ratio,ridge=ridge,dev_loss=loss(w))
                options.append(trial)
                candidates.append((trial['dev_loss'],w,trial))
                if trial['dev_loss']<best[0]:best=(trial['dev_loss'],w,trial)
                if local_best is None or trial['dev_loss']<local_best[0]:local_best=(trial['dev_loss'],w)
        grouped[str(balance)]={s:metrics(s,local_best[1]) for s in ('RDB7_test','BH9_TRIP50_test')}
    def deploy(w):
        common,doublet,triplet=w.reshape(3,-1)
        model=copy(ref)
        model.matern=combine(ref.matern,extra,{state:common+.5*(state[1]==2)*doublet+.5*(state[1]==3)*triplet for state in ref.supported_states})
        return model
    geometry_dev=[]
    if preserve_geometry:
        from experiments.reactivity.frequency_validation import measure
        dev=json.loads(POINTS.read_text())['ethanol']
        z,mass=np.array([MASSES[a] for a in dev['atoms']]).T
        # A common preset scale interpolates between the parent and a fitted
        # correction. Use only development energies and the development minimum.
        choices=[(zero,{'frozen_parent':True})]
        for _,coef,trial in sorted(candidates,key=lambda option:option[0])[:3]:
            choices.extend([(scale*coef,{**trial,'scale':scale}) for scale in (.25,.5,1.)])
        best_geometry=None
        for coef,trial in choices:
            try:
                minimum=measure(deploy(coef),z.astype(int),mass,np.array(dev['positions_A']),np.array(dev['frequencies_cm1']))
            except RuntimeError:
                continue
            val=loss(coef)/20 + minimum['bond_mae_A']/.03 + minimum['angle_mae_deg']/3 + minimum['mae_cm1']/100
            geom={k:minimum[k] for k in ('bond_mae_A','angle_mae_deg','mae_cm1')}
            geometry_dev.append(dict(config=trial,geometry=geom,selection_loss=val))
            if best_geometry is None or val<best_geometry[0]:best_geometry=(val,coef,trial)
        best=best_geometry
    _,w,chosen=best
    model=deploy(w)
    model.electronic=copy(ref.electronic);model.electronic.target+='; reaction-center local kernels on KinBot and publisher families'
    buf=BytesIO();model.save(buf);blob=buf.getvalue();ck='reaction-centers-'+sha256(blob).hexdigest()[:16]+'.npz';(OUT/ck).write_bytes(blob)
    restored=ForceField.load(OUT/ck,dispersion='d3')
    for k in list(cache)[::max(1,len(cache)//50)]:
        g=gs[k];v,e=cache[k]
        np.testing.assert_allclose(restored.energy(g.z,g.r,*g.state),e+v@w,atol=1e-7,rtol=0)
    comparison=next(w for _,w,t in candidates if t['class_balanced'] is False and t['ratio']==.5 and t['ridge']==.3)
    report=dict(pooled_centers=pooled,matched_configuration={s:metrics(s,comparison) for s in ('RDB7_test','BH9_TRIP50_test')},parent=info['checkpoint'],checkpoint=ck,QM7X_geometry_anchors=anchor_count,geometry_development=geometry_dev,centers={str(s):len(v) for s,v in extra.centers.items()},
                coverage={s:dict(Counter(labels[r.name] for r in rows)) for s,rows in sets.items()},trials=options,chosen=chosen,balanced_ablation=grouped,
                test={s:metrics(s,w) for s in ('RDB7_test','BH9_TRIP50_test')},by_family={s:metrics(s,w,True) for s in ('RDB7_test','BH9_TRIP50_test')})
    (OUT/('reaction_centers_pooled.json' if pooled else 'reaction_centers_shape.json' if preserve_geometry else 'reaction_centers.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(chosen,report['test'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preserve-geometry',action='store_true')
    parser.add_argument('--pooled-centers',action='store_true')
    args=parser.parse_args()
    run(args.preserve_geometry,args.pooled_centers)
