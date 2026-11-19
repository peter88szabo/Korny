"""Probe a molecule-specific sGDML residual using disjoint QM7-X displacements.

Run with the sGDML package on PYTHONPATH. The universal Korny backbone has
never seen these molecule IDs. For each probe molecule, specialize on 32
displacements, select the Matérn width on 12, and score the remaining ones.
This tests interpolation within a known topology. No new electronic structure
calculations are required and none of these local models are distributed as a
universal thermochemical correction.
"""
import json
import numpy as np
import h5py
from sgdml.train import GDMLTrain
from sgdml.predict import GDMLPredict

from experiments.reactivity.augmented import OUT, QM
from src.forcefield import ForceField


def main():
    rng = np.random.default_rng(1954)
    checkpoint = OUT/json.loads((OUT/"stationary_fit.json").read_text())["checkpoint"]
    base = ForceField.load(checkpoint)
    trainer = GDMLTrain(max_memory=1, max_processes=1, use_torch=False)
    split = json.loads((OUT/"results.json").read_text())
    cases = {}
    with h5py.File(QM) as archive:
        for mol in sorted(split["heldout_QM7X_ids"])[:2]:
            group = archive[mol]
            parent = next(key for key in sorted(group) if key.endswith("-opt")).rsplit("-",1)[0]
            keys = [str(k) for k in rng.permutation(sorted(k for k in group if k.startswith(parent+"-d")))]
            r = np.array([group[k]["atXYZ"][:] for k in keys])
            z = group[keys[0]]["atNUM"][:]
            f = np.array([group[k]["pbe0FOR"][:] for k in keys])
            e = np.array([group[k]["ePBE0"][0] for k in keys])
            pred = [base.predict(z,x) for x in r]
            residual_f = f-np.array([p[1] for p in pred])
            residual_e = e-np.array([p[0] for p in pred])
            # Supply explicit disjoint datasets to the stock training routine.
            def subset(section):
                return dict(name=np.array(mol),theory=np.array("PBE0 residual"),z=z,
                            R=r[section],F=residual_f[section],E=residual_e[section])
            n, v = 32,12
            train,dev = subset(slice(n)), subset(slice(n,n+v))
            test = slice(n+v,None)
            candidates = []
            for width in (.2,.8,2.):
                task = trainer.create_task(train,n,dev,v,sig=width,lam=1e-8,use_E=False)
                model = trainer.train(task)
                predictor = GDMLPredict(model,max_processes=1,use_torch=False)
                _,fd = predictor.predict(r[n:n+v].reshape(v,-1))
                loss = float(np.sqrt(np.mean((fd.reshape(v,-1,3)-residual_f[n:n+v])**2)))
                candidates.append((loss,width,predictor,len(task["perms"])))
            loss,width,predictor,nperms = min(candidates,key=lambda c:c[0])
            # Learn the energy-integration constant using only the 32 training energies.
            est_train,_ = predictor.predict(r[:n].reshape(n,-1))
            offset = np.mean(residual_e[:n]-est_train.ravel())
            et,ft = predictor.predict(r[test].reshape(len(r[test]),-1))
            before = residual_f[test]
            after = before-ft.reshape(before.shape)
            report = dict(parent=parent,train=keys[:n],development=keys[n:n+v],test=keys[test],
                          chosen_width=width,symmetries=nperms,dev_force_rmse_eV_A=loss,
                          before_force_rmse_eV_A=float(np.sqrt(np.mean(before**2))),
                          after_force_rmse_eV_A=float(np.sqrt(np.mean(after**2))),
                          before_relative_energy_mae_eV=float(np.mean(abs(residual_e[test]-residual_e[:n].mean()))),
                          after_relative_energy_mae_eV=float(np.mean(abs(residual_e[test]-offset-et.ravel()))))
            cases[mol] = report
            print(mol, len(after),nperms,report["before_force_rmse_eV_A"],report["after_force_rmse_eV_A"],flush=True)
    (OUT/"gradient_domain_probe.json").write_text(json.dumps(cases,indent=2)+"\n")


if __name__ == "__main__":
    main()
