"""Fit screened nuclear repulsion for all element pairs in the downloaded archive.

Keep the published three decay lengths for each element pair. Refit their
amplitudes on every tabulated DMol screening radius subject to phi(0)=1.
MP2 curves remain independent. Reverse element order denotes the same pair.
"""

import json
import re
import tarfile
import numpy as np
from experiments.full_public.specified import OUT
from experiments.reactivity.repulsion_audit import ARCHIVE, ROOT
from src.repulsion import NuclearRepulsion


def run():
    with tarfile.open(ARCHIVE) as tar:
        table = np.loadtxt(
            tar.extractfile(ROOT + "/nlh/nlh_coeffs.dat"), usecols=range(8)
        )
    params = {tuple(sorted((int(a), int(b)))): np.array(v) for a, b, *v in table}
    curves = dict(dmol={}, mp2={})
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for entry in tar:
            m = re.search(r"/(dmol|mp2)/screening_\1_(\d+)_(\d+)\.dat$", entry.name)
            if not m:
                continue
            method, a, b = m.groups()
            pair = tuple(sorted((int(a), int(b))))
            curve = np.loadtxt(tar.extractfile(entry))
            if pair in curves[method]:
                if np.array_equal(curves[method][pair], curve):
                    continue
                curve = np.unique(np.vstack((curves[method][pair], curve)), axis=0)
            curves[method][pair] = curve
    ntrain = 0
    retained = 0
    for pair, curve in curves["dmol"].items():
        r, ref = curve.T
        row = params[pair]
        amp = row[::2].copy()
        decay = row[1::2]
        # Fitting changes in two amplitudes makes their sum exactly one.
        exp = np.exp(-r[:, None] * decay)
        d = exp[:, :2] - exp[:, 2, None]
        ridge = 1e-5 * len(curve)
        lhs = d.T @ d + ridge * np.eye(2)
        sol = np.linalg.solve(lhs, d.T @ (ref - exp[:, 2]) + ridge * amp[:2])
        candidate = np.r_[sol, 1 - sol.sum()]
        # Both constraints apply to the physical compressed part of the curve.
        check = np.linspace(0, min(0.7, float(r.max())), 80)
        grid = np.exp(-check[:, None] * decay)
        if np.all(grid @ candidate >= -1e-6) and np.all(
            grid @ (candidate * decay) >= -1e-6
        ):
            row[::2] = candidate
        else:
            retained += 1
        ntrain += len(r)
    errors = []
    rep = []
    npoints = 0
    for pair, curve in curves["mp2"].items():
        r, ref = curve.T
        row = params[pair]
        p = np.exp(-r[:, None] * row[1::2]) @ row[::2]
        errors.extend(abs(p - ref))
        mask = r > 0
        factor = NuclearRepulsion.coulomb * np.prod(pair) / r[mask]
        energy = factor * ref[mask]
        selection = (energy >= 30) & (energy <= 1000)
        rep.extend(abs(factor * (p[mask] - ref[mask]))[selection])
        npoints += len(ref)
    np.savez_compressed(
        OUT / "short-range-pairs.npz",
        pairs=list(params),
        coefficients=list(params.values()),
    )
    stats = dict(
        training_pairs=len(curves["dmol"]),
        training_points=ntrain,
        original_retained_for_monotonicity=retained,
        validated_mp2_pairs=len(curves["mp2"]),
        validation_points=npoints,
        screening_mae=float(np.mean(errors)),
        compressed_energy_mae_eV=float(np.mean(rep)),
        compressed_points=len(rep),
    )
    (OUT / "core-validation.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats, flush=True)


if __name__ == "__main__":
    run()
