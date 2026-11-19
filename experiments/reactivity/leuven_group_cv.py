"""Leave out whole conformer groups when learning a private radical H-shift.

Every retained path has a CC start, saddle and end. Constraints include both
within-path gaps and cross-conformer electronic energies from the same CC
table column. This directly trains the relative stability of endpoints.
Each outer test cluster is omitted from model selection and training.
"""
from hashlib import sha256
from io import BytesIO
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from experiments.reactivity.leuven_hshift import CACHE,DIFF,EV_TO_KCAL,ForceField,references,score


def run():
    data,sets=references()
    grouped=[[i for i,row in enumerate(data) if row[0] in cluster] for cluster in sets]
    groups=[v for v in grouped if v]
    src=json.loads((CACHE/'global_fit.json').read_text())['fits']['CC']['checkpoint']
    base=ForceField.load(CACHE/src,dispersion='d3')
    x=[];y=[];g=[];dy=[]
    for _,_,pts,cc in data:
        hs=[];es=[];force_x=[];force_y=[]
        for p in pts:
            h,j=base.chemical.design(p.z,p.r,gradient=True)
            val,force=base.predict(p.z,p.r,0,2)
            hs.append(h);es.append(val)
            if p.force is not None:
                force_x.extend(j/np.sqrt(p.r.size))
                force_y.extend((p.force.ravel()-force.ravel())/np.sqrt(p.r.size))
        x.append(hs);y.append(cc-es);g.append(np.array(force_x).reshape(-1,len(h)));dy.append(np.array(force_y))
    x=np.array(x);y=np.array(y)
    slices=base.chemical.slices
    masks=[np.arange(base.chemical.size),np.r_[slices['carbon_H_peroxy'],slices['carbon_H_oxygen'],slices['oxygen_oxygen'],slices['organic_hydroperoxide'],slices['carbon_peroxy']]]
    trials=[(mi,ridge) for mi in range(len(masks)) for ridge in (.03,.3,3.,30.)]+[(-1,0.)]
    def train(keys,setting):
        cols,ridge=setting;w=np.zeros(x.shape[-1])
        if cols==-1:return w
        ids=np.concatenate([groups[i] for i in keys])
        weights=np.concatenate([np.full(len(groups[i]),len(groups[i])**-.5) for i in keys])
        h=x[ids];t=y[ids]
        # Remove the arbitrary CC absolute zero once across training geometries.
        # Centering each path separately would discard inter-conformer gaps.
        av=np.average(h.mean(axis=1),axis=0,weights=weights**2)
        zero=np.average(t.mean(axis=1),weights=weights**2)
        rows=[((h-av)*weights[:,None,None]).reshape(-1,len(w))]
        targets=[((t-zero)*weights[:,None]).ravel()]
        for i,weight in zip(ids,weights):
            rows.extend([weight*DIFF@x[i],.4*weight*g[i]])
            targets.extend([weight*DIFF@y[i],.4*weight*dy[i]])
        a=np.vstack(rows)[:,masks[cols]];b=np.concatenate(targets);active=(abs(a).max(axis=0)>1e-12)
        a=a[:,active];gram=a.T@a;gram.flat[::len(gram)+1]+=ridge
        w[masks[cols][active]]=cho_solve(cho_factor(gram),a.T@b)
        return w
    def error(ids,w):
        return (x[ids]@w-y[ids])@DIFF.T*EV_TO_KCAL
    def choose(keys):
        def loss(setting):
            folds=[]
            for i in keys:
                err=error(groups[i],train([j for j in keys if j!=i],setting))
                folds.append(abs(err[:,2]).mean()+abs(err[:,:2]).mean())
            return np.mean(folds)
        return min(trials,key=loss)
    def stats(arr):
        return dict(endpoint_gap=score(arr[:,2]),saddle_to_ends=score(arr[:,:2].ravel()),saddle_minus_start=score(arr[:,0]),saddle_minus_end=score(arr[:,1]))
    ids=list(range(len(groups)));results=[];all_err=[];before=[]
    for i in ids:
        fit=[j for j in ids if j!=i];setting=choose(fit);w=train(fit,setting)
        err=error(groups[i],w);old=error(groups[i],np.zeros(len(w)))
        results.append(dict(test=[data[k][0] for k in groups[i]],setting=setting,after=stats(err),before=stats(old)))
        all_err.extend(err);before.extend(old)
    selected=choose(ids);w=train(ids,selected)
    base.chemical.weights[(0,2)]+=w
    base.electronic.target+='; within- and between-conformer CC energy calibration on 16 private H-shift paths'
    buf=BytesIO();base.save(buf);blob=buf.getvalue();name='leuven-hshift-all-'+sha256(blob).hexdigest()[:12]+'.npz';(CACHE/name).write_bytes(blob)
    result=dict(parent=src,checkpoint=name,groups=results,selected_full=selected,before=stats(np.array(before)),after=stats(np.array(all_err)),
                macro_after={k:float(np.mean([row['after'][k]['mae_kcal_mol'] for row in results])) for k in ('endpoint_gap','saddle_to_ends')})
    (CACHE/'hshift_group_cv.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('four-group nested validation',result['before'],result['after'],result['macro_after'],flush=True)

if __name__=='__main__':run()
