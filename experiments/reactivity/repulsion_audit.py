"""Install the tabulated NLH term and test the full ML surface on independent MP2 curves.

The NLH coefficients are fitted to DMol for every Korny pair. The independent
MP2 screening functions in the same archive provide a check at 30–1000 eV,
inside the repulsive regime. Interaction energies use the separated-atom zero.
"""
import hashlib
import argparse
from io import BytesIO
from itertools import combinations_with_replacement
import json
import tarfile

import numpy as np

from experiments.reactivity.augmented import OUT
from src.forcefield import ForceField
from src.repulsion import NuclearRepulsion

ARCHIVE = OUT.parents[1]/"dataset_for_training/repulsion/nlh_potentials_opendata.tar.gz"
ROOT = "nlh_potentials_opendata"


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--cluster-fit", action="store_true", help="Use the QCell equilibrium and frequency refit")
    args = options.parse_args()
    prefix = "cluster_" if args.cluster_fit else ""
    fit_file = "cluster_fit.json" if args.cluster_fit else "stationary_fit.json"
    original = OUT/json.loads((OUT/fit_file).read_text())["checkpoint"]
    base, candidate = ForceField.load(original), ForceField.load(original)
    pairs = list(combinations_with_replacement(sorted(base.pair.species),2))
    rows, stats = [], {}
    with tarfile.open(ARCHIVE) as archive:
        source = np.loadtxt(archive.extractfile(ROOT+"/nlh/nlh_coeffs.dat"),usecols=range(8))
        table = {(int(row[0]),int(row[1])):row[2:] for row in source}
        core = NuclearRepulsion(base.pair.species, [table[p] for p in pairs])
        candidate.repulsion = core
        for a,b in pairs:
            z = np.array([a,b])
            screening = np.loadtxt(archive.extractfile(f"{ROOT}/mp2/screening_mp2_{a}_{b}.dat"))
            separated = np.array([[0.,0.,0.],[12.,0.,0.]])
            zero = base.energy(z,separated)
            for r,phi in screening:
                if r == 0:
                    continue
                ref = core.coulomb*a*b*phi/r
                if not 30 <= ref <= 1000:
                    continue
                xyz = np.array([[0.,0.,0.],[r,0.,0.]])
                e,f = base.predict(z,xyz)
                de,df = core.predict(z,xyz)
                rows.append([int(a),int(b),r,ref,e-zero,e-zero+de,f[1,0],f[1,0]+df[1,0]])
    data = np.asarray(rows)
    for name,col in (("original",4),("with_core",5)):
        err = data[:,col]-data[:,3]
        stats[name] = dict(mae_eV=float(np.mean(abs(err))),median_absolute_percent=float(
            100*np.median(np.abs(err)/data[:,3])),inward_force_points=int(sum(data[:,col+2] < 0)))
    # Check the Cartesian force at full strength, throughout the join and at both boundaries.
    force_errors = []
    core_only_errors = []
    for (a,b) in pairs:
        z = np.array([a,b])
        for fraction in (.45,.60,.68,.77,.82,.88):
            r = fraction*(core.radii_A[a]+core.radii_A[b])
            xyz = np.array([[0.,0.,0.],[r,0.,0.]])
            delta = np.array([[0.,0.,0.],[1e-5,0.,0.]])
            fd = (candidate.energy(z,xyz-delta)-candidate.energy(z,xyz+delta))/2e-5
            force_errors.append(abs(fd-candidate.predict(z,xyz)[1][1,0]))
            fd_core = (core.predict(z,xyz-delta)[0]-core.predict(z,xyz+delta)[0])/2e-5
            core_only_errors.append(abs(fd_core-core.predict(z,xyz)[1][1,0]))
    stream = BytesIO()
    candidate.save(stream)
    content = stream.getvalue()
    name = "pbe0-core-"+hashlib.sha256(content).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(content)
    summary = dict(checkpoint=name,backbone=original.name,reference="MP2 screening, 30–1000 eV",
                   source="https://zenodo.org/records/17302337",n_element_pairs=len(set(map(tuple,data[:,:2]))),
                   n_points=len(data),metrics=stats,worst_force_fd_eV_A=max(force_errors),
                   worst_core_force_fd_eV_A=max(core_only_errors))
    (OUT/(prefix+"repulsion_audit.json")).write_text(json.dumps(summary,indent=2)+"\n")
    np.savetxt(OUT/(prefix+"repulsion_curves.csv"),data,fmt="%.9g",delimiter=",",
               header="Z1,Z2,r_A,MP2_eV,previous_eV,with_core_eV,previous_radial_force_eV_A,with_core_radial_force_eV_A",comments="")
    print(summary,flush=True)


if __name__ == "__main__":
    main()
