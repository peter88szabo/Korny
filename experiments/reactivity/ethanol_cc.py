"""Molecule-specific ethanol CCSD(T)/cc-pVTZ fit on the authors' published split."""
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from experiments.reactivity.leuven_endpoints import connectivity
from experiments.reactivity.spin_data import DATA
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.molecular_kernel import MolecularKernel

def run():
    root=DATA/'sGDML'
    with np.load(root/'ethanol_ccsd_t-train.npz') as d:
        z,r,f,e=d['z'],d['R'],d['F']/EV_TO_KCAL,d['E'].ravel()/EV_TO_KCAL
    graph=connectivity(z,r[0])
    perms=np.array(graph.GetSubstructMatches(graph,uniquify=False,maxMatches=10000))
    rng=np.random.default_rng(97);shuffled=rng.permutation(len(r));tr,dev=shuffled[120:],shuffled[:120]
    # Pick centers from training only, greedily maximizing the minimum distance
    # to a selected center after symmetry alignment.
    x=MolecularKernel.descriptors(r[tr]);basis=MolecularKernel(z,x[:1],perms,1.)
    centers=[];ds=np.full(len(tr),np.inf);idx=0
    for _ in range(320):
        c=x[idx];centers.append(c)
        symmetry_images=MolecularKernel(z,c[None,:],perms,1.).images[0]
        ds=np.minimum(ds,((x[:,None,:]-symmetry_images)**2).sum(axis=2).min(axis=1));idx=ds.argmax()
    trials=[]
    def learn(v,j,which,ridge):
        a=np.vstack((v[which]-v[which].mean(axis=0),.12*j[which].reshape(-1,v.shape[1])))
        y=np.r_[e[which]-e[which].mean(),.12*f[which].ravel()]
        gram=a.T@a;gram.flat[::len(gram)+1]+=ridge
        w=cho_solve(cho_factor(gram),a.T@y)
        return w,np.mean(e[which]-v[which]@w)
    for width in (.12,.28,.60):
        model=MolecularKernel(z,centers,perms,width)
        designs=[model.design(p) for p in r];v,j=map(np.array,zip(*designs))
        for ridge in (1e-9,1e-6):
            w,offset=learn(v,j,tr,ridge)
            de=np.abs(v[dev]@w+offset-e[dev]).mean()*EV_TO_KCAL
            df=np.abs(j[dev]@w-f[dev].reshape(len(dev),-1)).mean()*EV_TO_KCAL
            trials.append((float(de+.12*df),width,ridge))
        print('development width',width,trials[-2:],flush=True)
    _,width,ridge=min(trials)
    model=MolecularKernel(z,centers,perms,width)
    v,j=map(np.array,zip(*(model.design(p) for p in r)))
    model.weights,model.offset=learn(v,j,np.arange(len(r)),ridge)
    model.save(root/'korny_ethanol_cc.npz')
    model=MolecularKernel.load(root/'korny_ethanol_cc.npz')
    with np.load(root/'ethanol_ccsd_t-test.npz') as d:
        np.testing.assert_array_equal(z,d['z'])
        p,forces=map(np.array,zip(*(model.predict(z,g) for g in d['R'])))
        err=p*EV_TO_KCAL-d['E'].ravel();df=forces*EV_TO_KCAL-d['F']
    stats=dict(n_train=len(r),n_test=len(err),n_centers=len(centers),n_permutations=len(perms),width=width,ridge=ridge,development=trials,
               energy_mae_kcal_mol=float(abs(err).mean()),energy_rmse_kcal_mol=float(np.mean(err**2)**.5),force_mae_kcal_mol_A=float(abs(df).mean()),force_rmse_kcal_mol_A=float(np.mean(df**2)**.5))
    (root/'ethanol_cc.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats,flush=True)

if __name__=='__main__':run()
