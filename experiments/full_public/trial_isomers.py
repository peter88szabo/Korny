"""Use the electronic energies already archived for individual stationary structures.

Reaction triples leave one undetermined common energy per reaction. Relative
energies between isomers of the same formula, charge, spin and reference method
supply additional constraints on that common energy with no arbitrary atom
energy reference. Exclude every geometry from held-out equations before centering.
"""

from collections import defaultdict
import pickle
import re
import tarfile
import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.references import RAD
from experiments.reactivity import rdb7
from src.units import HARTREE_EV


def prepare():
    geoms, rows, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        ids, names = pickle.load(f)
    energies = {}
    datasets = (("RDB7", rdb7.ARCHIVE), ("Grambow", RAD))
    for src, path in datasets:
        with tarfile.open(path, "r|gz") as arc:
            for member in arc:
                if src == "RDB7":
                    match = rdb7.NAME.match(member.name)
                    if not match:
                        continue
                    key = "RDB7/" + match[1] + "/" + match[2]
                    if key not in ids:
                        continue
                    log = arc.extractfile(member).read().decode(errors="replace")
                    values = re.findall(
                        r"^\s*!(?:U|R)?CCSD\(T\)-F12a total energy\s+([-\d.Ee+]+)",
                        log,
                        re.M,
                    )
                    assert len(values) == 1
                    assert (
                        "Molpro calculation terminated" in log
                        or rdb7.complete_f12a_table(log, values[0])
                    )
                else:
                    match = re.search(r"/rxn(\d+)/(ts|p|r)\d+\.log$", member.name)
                    if not match:
                        continue
                    key = "Grambow/" + match[1] + "/" + match[2]
                    if key not in ids:
                        continue
                    log = arc.extractfile(member).read().decode(errors="replace")
                    tail = log.rsplit("Standard Nuclear Orientation (Angstroms)", 1)[-1]
                    values = re.findall(
                        r"Total energy in the final basis set\s*=\s*([-\d.]+)",
                        tail,
                        re.I,
                    )
                energies[key] = float(values[-1]) * HARTREE_EV
    for _, src, _, _, terms, ref, _ in rows:
        if src in dict(datasets):
            np.testing.assert_allclose(
                sum(c * energies[k] for k, c in terms), ref, atol=3e-10, rtol=1e-9
            )
    partition = defaultdict(set)
    for target in kt.TARGETS:
        d = kt.equation_data(target)
        matrix, _, sources, fold, _, _ = d
        for i in range(matrix.shape[0]):
            for j in matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]]:
                partition[j].add(int(fold[i]))
    obs = defaultdict(list)
    for key, e in energies.items():
        idx = ids[key]
        g = geoms[key]
        group = tuple(sorted(int(z) for z in g.z)), g.state
        obs[key.split("/")[0], group, idx].append(e)
    refdata = defaultdict(lambda: defaultdict(list))
    dropped = defaultdict(int)
    for (src, group, idx), es in obs.items():
        # Reused equilibrium fragments and near-identical archive copies receive
        # one label each. Inconsistent electronic roots at identical coordinates
        # are unsuitable for a geometry-only single-surface model.
        if -1 in partition[idx] or np.ptp(es) > 0.001:
            dropped[src] += 1
            continue
        refdata[src][group].append((idx, float(np.mean(es)), partition[idx]))
    refdata = {s: dict(v) for s, v in refdata.items()}
    counts = {}
    for s, groups in refdata.items():
        kept = [len(v) for v in groups.values() if len(v) >= 2]
        counts[s] = dict(
            geometries=sum(kept),
            groups=len(kept),
            independent_centered_equations=sum(kept) - len(kept),
            excluded_held_or_conflicting=dropped[s],
        )
    with (kt.WORK / "isomer-observations.pkl").open("wb") as f:
        pickle.dump(refdata, f)
    kt.write_json(kt.WORK / "isomer-observations.json", counts)
    print("Verified archived absolute labels", counts, flush=True)


def gram(target, h, x, base):
    with (kt.WORK / "isomer-observations.pkl").open("rb") as f:
        groups = pickle.load(f)["RDB7" if target == "high" else "Grambow"]
    gram = np.zeros((x.shape[1] + 1,) * 2)
    for entries in groups.values():
        rows = [(j, e) for j, e, split in entries if h not in split]
        if len(rows) < 2:
            continue
        ids, energy = zip(*rows)
        ix = np.asarray(ids)
        a = np.column_stack((x[ix], np.array(energy) - base[ix]))
        a -= np.mean(a, axis=0)
        gram += a.T @ a
    return gram


if __name__ == "__main__":
    prepare()
