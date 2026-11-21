"""Transfer local changes along radical paths to the singlet and doublet surfaces.

A source electronic energy compresses a path into one scalar. Here each target
Hamiltonian learns its own response along the local environment changes seen
on training IRCs. The leading eigenvectors of the uncentered matrix of changes
retain an additive single-geometry representation with analytic forces.
"""

import json
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_multiscale import baseline
from experiments.full_public.trial_path_scales import OPTIONS, ROOT
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.periodic_kernel import PeriodicKernel

WORK = kt.WORK
STAGE = "p57-public"
PARENT = "p49-public"


def learn_directions():
    records = json.loads((ROOT / "snapshot.json").read_text())
    ids = [r["rxn"] for r in records if not r["heldout"]]
    cov = None
    for lo in range(0, len(ids), 64):
        blocks = []
        for rx in ids[lo : lo + 64]:
            with np.load(ROOT / f"{rx}.npz") as f:
                blocks.append(f["rows"][:, :-1])
        x = np.vstack(blocks).astype(float)
        batch = x.T @ x
        cov = batch if cov is None else cov + batch
    eig, vec = np.linalg.eigh(cov / len(ids))
    # Predeclared budget of 96 modes; a floor limits amplification of rare
    # geometry changes. Neither cutoff depends on evaluation energies.
    order = np.argsort(eig)[::-1]
    idx = order[:96]
    with np.load(ROOT / "native.npz") as f:
        heads = f["weights"]
    projection = np.column_stack([vec[:, idx] / np.sqrt(eig[idx] + 0.02), heads])
    return projection, {
        "reactions": len(ids),
        "directions": len(idx),
        "captured": float(eig[idx].sum() / eig.sum()),
        "native_heads": heads.shape[1],
    }


def compact_banks(projection, stride=3):
    """Keep selected atomic centers in their original order in each kernel."""
    basis, selected = load_basis()
    selected = selected[::stride]
    starts = np.cumsum([0] + [len(v) for v in basis.centers.values()])
    centers = {}
    for group, lo, hi in zip(basis.centers, starts[:-1], starts[1:]):
        take = selected[(selected >= lo) & (selected < hi)] - lo
        if len(take):
            centers[group] = basis.centers[group][take]
    banks = []
    for (length, degree), p in zip(
        OPTIONS, np.split(projection, len(OPTIONS)), strict=True
    ):
        b = PeriodicKernel(
            basis.groups,
            basis.radii,
            basis.projections,
            centers,
            basis.shells,
            basis.cutoff,
            length,
            degree,
            np.zeros(len(selected)),
        )
        banks.append((b, p))
    return banks, selected


def project_matrix(side, banks, selected):
    """Stream the two memory-mapped banks to stay within the shared 8 GB cap."""
    result = None
    for b, proj in banks:
        mat = np.load(WORK / f"periodic-{b.length:g}-{side}.npy", mmap_mode="r")
        if result is None:
            result = np.zeros((len(mat), proj.shape[1]))
        for lo in range(0, len(mat), 256):
            sl = slice(lo, lo + 256)
            result[sl] += mat[sl][:, selected] @ proj
    return result


def run():
    projection, info = learn_directions()
    banks, selected = compact_banks(projection)
    main = project_matrix("main", banks, selected)
    rad = project_matrix("rad", banks, selected).reshape(-1, 7, projection.shape[1])
    fit_surfaces(banks, main, rad, info, STAGE, "directions57")


def fit_surfaces(
    banks, main, rad, info, stage, key, hdf_shape=None, source_constraint=None,
    huber_kcal=None, parent_stage=PARENT, ridge_penalty=None, shared_dft_weight=0.0,
    balance_families=False, parent_force_cache=None, relax_spectra=True,
    reference_constraints=None,
):
    """Fit both target references from p49 with the same derivative constraints."""
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [x[1][0].split("/")[-1].startswith("ts") for x in pickle.load(f)]
        )
    meta = np.load(WORK / "rad-meta.npy")
    pub = main[: len(names)]
    # The public features are followed by all central PBE0 anchor geometries
    # and two opposite displacements used to protect forces and curvature.
    sources = np.load(WORK / "hdf-sources.npy")
    info["hdf_anchor_geometries"] = len(sources)
    shape = hdf_shape
    if shape is None:
        frames = main[len(names) :].reshape(-1, 3, main.shape[1])
        assert len(frames) == len(sources)
        shape = np.zeros((main.shape[1],) * 2)
        for source in np.unique(sources):
            idx = np.flatnonzero(sources == source)
            for lo in range(0, len(idx), 512):
                center, plus, minus = frames[idx[lo : lo + 512]].transpose(1, 0, 2)
                force = (plus - minus) / 0.002
                curvature = ((plus - center) + (minus - center)) / 0.001**2
                shape += (4500 / len(idx)) * (
                    0.05 * (center.T @ center + 16 * force.T @ force)
                    + 0.012 * curvature.T @ curvature
                )
    print("Prepared reference-specific path directions", info, flush=True)
    geom, _, _ = kt.inputs()

    def derivative(z, r):
        if banks[0][1] is None:
            return np.column_stack([b.design(z, r, True)[1] for b, _ in banks])
        return sum(b.design(z, r, True)[1] @ proj for b, proj in banks)

    for target in kt.TARGETS:
        model = ForceField.load(WORK / f"{parent_stage}-{target}.npz", dispersion="d3")
        data = kt.equation_data(target)
        s, y, src, fold, kind, case = data
        parent = baseline(target, -1, "parent", parent_stage)
        x = s @ pub
        family_weights = None
        if balance_families:
            from experiments.full_public.reaction_family_weights import training_weights
            family_weights = training_weights(data)
            info["balance_training_families"] = True
        groups = kt.grouped_moments(data, x, y - s @ parent, fold >= 0, family_weights)
        eq = (
            groups["other"]
            + 96 * groups["mid"]
            + 36 * groups["gap"]
            + isomers(target, -1, pub, parent)
        )
        del x, groups
        if reference_constraints is not None:
            # The two target surfaces can receive different native electronic
            # references, with each source's discrepancy profiled separately.
            eq += reference_constraints[target]
        eq[:-1, :-1] += shape
        cache = None if parent_force_cache is None else parent_force_cache.setdefault((parent_stage,target), {})
        nist = nist_rows(derivative, model, force_cache=cache)
        eq += 24 * nist.T @ nist
        if target == "wb97xd3":
            if shared_dft_weight:
                from experiments.full_public.trial_fidelity import rdb_dft
                # Paired RDB7 DFT labels have their own electronic reference.
                # Supervise the DFT branch at its native geometries, retaining
                # the original coupled-cluster equations on the high branch.
                dft = rdb_dft(kt.equation_data("high"))
                ds, dy, _, dfold, dkind, _ = dft
                inc = kt.grouped_moments(dft, ds @ pub, dy - ds @ parent, dfold >= 0)
                eq += shared_dft_weight * (96 * inc["mid"] + 36 * inc["gap"])
                info["additional_dft_training_reactions"] = int(np.sum((dfold >= 0) & (dkind == "reaction")))
                info["additional_dft_weight"] = shared_dft_weight
            if source_constraint is not None:
                eq += source_constraint
            bm = baseline(target, -1, "modes", parent_stage)
            a, v = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
            eq += trial_fit.reduce_gram(a, v, np.repeat(meta[:, 0] >= 0, 3))
        ridge = 6 * np.eye(len(eq) - 1) if ridge_penalty is None else ridge_penalty
        factor = cho_factor(eq[:-1, :-1] + ridge, lower=huber_kcal is not None)
        w = cho_solve(factor, eq[:-1, -1])
        if huber_kcal is not None:
            from experiments.full_public.trial_robust import robust_reactions

            w, fit_info = robust_reactions(data, s @ pub, y - s @ parent, factor, w, huber_kcal)
            info.setdefault("robust_optimization", {})[target] = fit_info
            print("Optimized robust reaction objective", target, fit_info, flush=True)
        offset = 0
        for basis, proj in banks:
            basis.weights = (
                w[offset : offset + basis.size] if proj is None else proj @ w
            )
            offset += basis.size
            model.additional_bases.append(basis)
        path = WORK / f"{stage}-{target}.npz"
        model.save_with_base(kt.original.BASE, path)
        restored = ForceField.load(path, dispersion="d3")
        pred = parent + pub @ w
        for mult in (1, 2):
            i = next(
                i
                for i, n in enumerate(names)
                if geom[n].state[1] == mult and 4 <= len(geom[n].z) <= 12
            )
            g = geom[names[i]]
            energy, forces = restored.predict(g.z, g.r, *g.state)
            np.testing.assert_allclose(energy, pred[i], atol=5e-6, rtol=0)
            v = np.random.default_rng(i).normal(size=g.r.shape)
            v /= np.linalg.norm(v)
            fd = (
                restored.energy(g.z, g.r - 2e-5 * v, *g.state)
                - restored.energy(g.z, g.r + 2e-5 * v, *g.state)
            ) / 4e-5
            np.testing.assert_allclose(fd, (forces * v).sum(), atol=7e-5, rtol=0)
        held = fold < 0
        np.save(WORK / f"{key}-{target}-parent--1.npy", pred)
        np.savez_compressed(
            WORK / f"{stage}-{target}-equations.npz",
            source=src[held],
            case=case[held],
            kind=kind[held],
            error_kcal=((s @ pred - y) * kt.KCAL)[held],
        )
        if target == "wb97xd3":
            modes = bm + rad @ w
            np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
            np.save(WORK / f"{stage}-held-modes.npy", modes[meta[:, 0] < 0])
    kt.write_json(
        WORK / f"{stage}-settings.json",
        dict(
            selected=stage,
            parent=parent_stage,
            branches=dict.fromkeys(kt.TARGETS, key),
            **info,
        ),
    )
    compare(stage)
    validate(stage)
    print("Completed matched reactions and saddle curvatures", stage, flush=True)
    if relax_spectra:
        evaluate_trial(stage)
        print("Completed independent relaxed spectra", stage, flush=True)


if __name__ == "__main__":
    run()
