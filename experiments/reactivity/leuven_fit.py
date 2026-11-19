"""Calibrate private gas-phase energy surfaces with archived CC and DFT data.

Fit separate CC and DFT energy corrections. Converged DFT structures constrain
the derivative of both surfaces (an explicit approximation for CC). Use the
measured gradient where matched. Preserve the entire C5H9O3 doublet H-shift
composition for external transfer. Independent constitutional graphs are held
out together, across conformers, calculations, methods and archive copies.
"""
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
import sqlite3
import numpy as np
from scipy.linalg import cho_factor,cho_solve

from experiments.reactivity.leuven_endpoints import connectivity,key
from experiments.reactivity.leuven_hshift import CACHE,DIFF,OUT,EV_TO_KCAL,ForceField,HARTREE_EV,references,score
from experiments.reactivity.isoprene_irc_fit import fingerprint


def run():
    d=sqlite3.connect(CACHE/'labels.sqlite')
    pname=json.loads((OUT/'rdb7_chemical_residual_v4.json').read_text())['checkpoint']
    parent=ForceField.load(OUT/pname,dispersion='d3')
    prepared={};seen=set();cls=Counter();rows=defaultdict(list);deriv=defaultdict(list)
    state_count=Counter()
    # Train both derivatives and energies in eV and Å. Projection onto the
    # most common local channels complements the explicit functional groups.
    kernel=parent.matern;inds=[]
    for s in (1,6,7,8):
        n=len(kernel.centers[s]); ix=np.linspace(0,n-1,min(n,32),dtype=int)
        inds.extend((kernel.slices[s].start+ix).tolist())
        inds.extend((kernel.slices[s].start+n+ix).tolist())
    inds=np.array(inds,int);nchem=parent.chemical.size; width=nchem+len(inds)
    def design(z,r,gradient):
        h,j=parent.chemical.design(z,r,gradient=gradient)
        sites=kernel.project(z,r,gradient=gradient);k,f=kernel.design(sites,r.size if gradient else 0)
        return np.r_[h,.25*k[inds]], np.column_stack((j,.25*f[:,inds])) if gradient else None
    query="""SELECT l.kind,l.level,l.q,l.m,l.z,l.r,l.energy,t.force,l.hash,l.stage
             FROM labels l LEFT JOIN derivatives t USING(hash,stage) ORDER BY l.hash,l.stage"""
    for i,(kind,level,q,m,zb,rb,energy,fb,log_id,stage) in enumerate(d.execute(query)):
        z=np.frombuffer(zb,np.uint8).astype(int);r=np.frombuffer(rb,'<f8').reshape(-1,3);state=(q,m)
        if not level.endswith('/gas') or state not in parent.supported_states:
            cls['solution_or_unsupported_state']+=1;continue
        formula=tuple(sorted(z.tolist()))
        if formula==tuple([1]*9+[6]*5+[8]*3) and state==(0,2):
            cls['entire_external_H_shift_composition']+=1;continue
        cc='ccsd' in level
        if not cc and kind!='optimized':
            cls['nonCC_nonoptimized']+=1;continue
        fp=fingerprint(z,r).round(4).tobytes()
        uid=(formula,state,fp,level)
        if uid in seen:
            cls['same_geometry_method_state']+=1;continue
        seen.add(uid)
        graph=key(connectivity(z,r))
        bucket=int(sha256(('Leuven-graph-v1/'+graph).encode()).hexdigest()[:8],16)%10
        split='test' if bucket<2 else 'development' if bucket<4 else 'train'
        objective='CC' if cc else 'DFT'
        group=(level,formula,state);ident=(zb,rb,state)
        grad=not cc
        if ident not in prepared or (grad and prepared[ident][2] is None):
            x,j=design(z,r,grad)
            if grad:
                v,f=parent.predict(z,r,*state)
            else:
                v,f=parent.energy(z,r,*state),None
            # Keep energies/features only long-term. Jacobians for thousands
            # of structures are immediately compressed into normal equations.
            prepared[ident]=(x,v,grad if grad else None)
        else:
            x,v,_=prepared[ident]
            if grad:
                x,j=design(z,r,True);_,f=parent.predict(z,r,*state)
        rows[objective,split,state].append((group,graph,x,energy*HARTREE_EV-v))
        if grad:
            # A normalized deterministic random projection is an unbiased
            # low-memory sketch of the Cartesian force equations.
            rng=np.random.default_rng(int(log_id[:16],16))
            u=rng.normal(size=(4,r.size))/np.sqrt(4*r.size)
            force=np.frombuffer(fb,'<f8') if fb else np.zeros(r.size)
            deriv[split,state].append((u@j,u@(force-f.ravel()),graph,fb is not None))
        cls[objective+'_'+split]+=1;state_count[str(state)]+=1
        if i%1000==0:print('read archive records',i,dict(cls),flush=True)
    def centered(objective,state):
        # Intercepts differ by level, spin and composition. Their training-only
        # means set an energy origin. The deployed model learns differences.
        collectors=defaultdict(list)
        for g,graph,x,y in rows[objective,'train',state]:collectors[g].append((x,y))
        means={g:(np.mean([x for x,_ in pairs],axis=0),np.mean([y for _,y in pairs])) for g,pairs in collectors.items()}
        splits={};skips={}
        for split in ('train','development','test'):
            arr=[];b=[];groups=[]
            for g,graph,x,y in rows[objective,split,state]:
                if g not in means:continue
                mx,my=means[g];arr.append(x-mx);b.append(y-my);groups.append(graph)
            skips[split]=len(rows[objective,split,state])-len(arr)
            splits[split]=(np.array(arr).reshape(-1,width),np.array(b),groups)
        return splits,skips
    summary={'inventory':dict(cls),'model_parent':pname,'states':dict(state_count),'fits':{}}
    for objective in ('CC','DFT'):
        model=ForceField.load(OUT/pname,dispersion='d3')
        stats={}
        for state in model.supported_states:
            data,skips=centered(objective,state)
            a,y,groups=data['train'];va,vy,_=data['development'];te,ty,_=data['test']
            if not len(groups):continue
            weights=Counter(groups);w=np.array([weights[g]**-.25 for g in groups]);w/=np.sqrt(np.mean(w*w))
            a=a*w[:,None];y=y*w
            gs=deriv['train',state]
            j=np.vstack([p[0] for p in gs]) if gs else np.empty((0,width))
            v=np.concatenate([p[1] for p in gs]) if gs else np.empty(0)
            gram=a.T@a+.25*j.T@j;rhs=a.T@y+.25*j.T@v
            candidates=[(np.mean(abs(vy)),None,np.zeros(width))] if len(vy) else []
            for ridge in (.03,.3,3.,30.,300.):
                param=cho_solve(cho_factor(gram+ridge*np.eye(width)),rhs)
                if len(vy) or ridge==3.:
                    candidates.append((np.mean(abs(va@param-vy)) if len(vy) else 0.,ridge,param))
            _,ridge,param=min(candidates,key=lambda p:p[0])
            model.chemical.weights[state]+=param[:nchem];model.matern.weights[state][inds]+=.25*param[nchem:]
            stats[str(state)]=dict(ridge=ridge,training_records=len(y),development=len(vy),test=len(ty),test_graphs=len(set(data['test'][2])),gradient_geometries=len(gs),measured_gradients=sum(p[3] for p in gs),unseen_reference_origin_excluded=skips,
                                   parent=score(-ty*EV_TO_KCAL) if len(ty) else None,fit=score((te@param-ty)*EV_TO_KCAL) if len(ty) else None)
        model.electronic.target+='; private gas '+objective+' within-composition relative energies with DFT stationary constraints'
        buf=BytesIO();model.save(buf);blob=buf.getvalue();filename='leuven-'+objective.lower()+'-'+sha256(blob).hexdigest()[:12]+'.npz';(CACHE/filename).write_bytes(blob)
        restored=ForceField.load(CACHE/filename,dispersion='d3');external=[]
        for tag,role,points,ccref in references()[0]:
            if role=='test':
                external.append(DIFF@(np.array([restored.energy(p.z,p.r,0,2) for p in points])-ccref)*EV_TO_KCAL)
        external=np.array(external)
        summary['fits'][objective]=dict(checkpoint=filename,states=stats,external_endpoint_gap=score(external[:,2]),external_saddle_to_ends=score(external[:,:2].ravel()))
        print(objective,stats,summary['fits'][objective]['external_endpoint_gap'],flush=True)
    (CACHE/'global_fit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
