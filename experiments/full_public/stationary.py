"""Fit PBE0 equilibrium and vibrational constraints and install the NLH core.

The harmonic loss is evaluated on the internal mass-weighted Hessian. Validation
molecules are relaxed on the fitted surface. The independent MP2 screening
curves test the published short-range core for every backbone element pair.
"""

from itertools import combinations_with_replacement
import json
import tarfile
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from experiments.full_public.specified import OUT
from experiments.reactivity.larger_vibrations import references, evaluate
from experiments.reactivity.stationary_fit import (
    POINTS,
    stationary_rows,
    residuals,
    summarize,
)
from experiments.reactivity.repulsion_audit import ARCHIVE, ROOT
from src.forcefield import ForceField
from src.repulsion import NuclearRepulsion
from src.valence import Valence


def run():
    start = monotonic()
    model = ForceField.load(OUT / "pbe0-full.npz")
    head = Valence()
    points = json.loads(POINTS.read_text())
    points.update(references())
    held = {"ethanol", "benzene", "propane"}
    train = {
        name: stationary_rows(model, head, p)
        for name, p in points.items()
        if name not in held
    }
    g = [v[0] for v in train.values()]
    h = [v[1:] for v in train.values()]
    prior = np.r_[
        np.tile([1.5, 8.0], len(head.pairs)),
        np.tile([2.0, 6.0, 2.0, 3.0], len(head.centers)),
    ]
    # No additional force secants are needed here; the starting force field
    # has already been fitted to every eligible HDF5 training geometry.
    zero = np.zeros((1, head.size + 1))

    def loss(w):
        return residuals(w, g, h, zero, prior)

    opt = least_squares(
        lambda w: loss(w)[0], head.weights, jac=lambda w: loss(w)[1], max_nfev=400
    )
    if not opt.success:
        raise RuntimeError(opt.message)
    head.weights = opt.x
    model.valence = head
    pairs = list(combinations_with_replacement(sorted(model.pair.species), 2))
    err = []
    core_err = []
    allpts = 0
    with tarfile.open(ARCHIVE) as tar:
        tab = np.loadtxt(
            tar.extractfile(ROOT + "/nlh/nlh_coeffs.dat"), usecols=range(8)
        )
        lookup = {(int(a), int(b)): v for a, b, *v in tab}
        core = NuclearRepulsion(model.pair.species, [lookup[p] for p in pairs])
        model.repulsion = core
        for a, b in pairs:
            z = np.array([a, b])
            zero_xy = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

            # Evaluate the state-independent repulsive interaction relative to
            # isolated atoms. The pair reference is independent of spin labels.
            def energy(r):
                return (
                    model.pair.predict(z, r)[0]
                    + model.local.energy(z, r)
                    + head.predict(z, r)[0]
                    + core.predict(z, r)[0]
                )

            e0 = energy(zero_xy)
            data = np.loadtxt(tar.extractfile(f"{ROOT}/mp2/screening_mp2_{a}_{b}.dat"))
            for dist, phi in data:
                if dist <= 0:
                    continue
                pos = np.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]])
                reference = core.coulomb * a * b * phi / dist
                delta = energy(pos) - e0 - reference
                err.append(abs(delta))
                if 30 <= reference <= 1000:
                    core_err.append(abs(delta))
                allpts += 1
    model.save(OUT / "pbe0-stationary-core.npz")
    stats = dict(
        training=sorted(train),
        training_scores=summarize(train, opt.x),
        core_pairs=len(pairs),
        screening_points=allpts,
        all_positive_distance_mae_eV=float(np.mean(err)),
        core_domain_points=len(core_err),
        core_domain_mae_eV=float(np.mean(core_err)),
    )
    print(
        "NIST fit",
        len(train),
        "core",
        len(pairs),
        allpts,
        "points; wrote checkpoint",
        flush=True,
    )
    stats["validation"] = evaluate(model, {n: points[n] for n in held})
    stats["seconds"] = monotonic() - start
    (OUT / "stationary.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    run()
