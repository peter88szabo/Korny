"""Use AIMNet-NSE (2021) electronic energies for vertical redox supervision.

Pair the neutral singlet with the anion or cation doublet at exactly the same
coordinates, atom order and conformation ID. The ionic and neutral absolute
energy origins cancel within each pair. Train independent molecular IP and EA
readouts in eV. Each uses both the mean chemical environment and the local
environments weighted by p49's separately learned ionic spin population.
"""

from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_spin_population import CACHE, ROOT, probabilities
from src.spin_local import DIM, GROUPS, SpinLocalKernel

DEST = ROOT / "vertical-gaps"
DIMENSION = 2 * GROUPS * DIM + 2


def design(model, types, local, q):
    """Augment mean site chemistry with geometry-dependent ionic spin density."""
    k = q + 1
    p = probabilities(types, local, model.logits[k])
    pol = np.einsum("nad,nad->na", local, model.polarization[k, types])
    spin = p * (1 + pol - (p * pol).sum(axis=1, keepdims=True))
    n, atoms = types.shape
    x = np.zeros((n, 2, GROUPS, DIM))
    for g in np.unique(types):
        mask = types == g
        x[:, 0, g] = np.einsum("na,nad->nd", mask, local) / atoms
        x[:, 1, g] = np.einsum("na,nad->nd", mask * spin, local)
    # Size contributes weakly to charging a compact molecule. Its inverse
    # radius proxy is fitted for each charge with an independent intercept.
    return np.column_stack(
        (x.reshape(n, -1), np.full(n, atoms ** (-1 / 3)), np.ones(n))
    )


def prepare(model):
    counts = Counter()
    for source in ("ions16", "chembl20"):
        with h5py.File(ROOT / (source + ".h5"), "r") as h:
            for name, g in h.items():
                q, spin, ids, xyz, z, e = (
                    g[n][:]
                    for n in ("charge", "mult", "mol_id", "coord", "numbers", "energy")
                )
                neutral = {}
                for i, n in enumerate(ids):
                    if q[i] == 0 and spin[i] == 1:
                        if n in neutral:
                            raise ValueError("Ambiguous neutral conformation ID")
                        neutral[n] = i
                for lo in range(0, len(q), 480):
                    cache = CACHE / f"{source}-{name}-{lo:06d}.npz"
                    take = np.flatnonzero(spin[lo : lo + 480] == 2) + lo
                    with np.load(cache) as a:
                        if not np.array_equal(a["q"], q[take]):
                            raise ValueError("Spin-label row order differs")
                        xs, ts, part = a["x"], a["types"], a["part"]
                    for charge in (-1, 1):
                        rows = []
                        labels = []
                        for row, idx in enumerate(take):
                            if q[idx] != charge:
                                continue
                            j = neutral.get(ids[idx])
                            if j is None:
                                counts[source, "missing neutral"] += 1
                                continue
                            if not np.array_equal(
                                xyz[idx], xyz[j]
                            ) or not np.array_equal(z[idx], z[j]):
                                raise ValueError(
                                    "Ion and neutral coordinates differ for the same ID"
                                )
                            # IP = E+ - E0 and EA = E0 - E-.
                            rows.append(row)
                            labels.append(charge * (e[idx] - e[j]))
                        if not rows:
                            continue
                        x = design(model, ts[rows], xs[rows], charge)
                        y = np.array(labels)
                        dest = DEST / f"{source}-{name}-{lo:06d}-q{charge}.npz"
                        with dest.with_suffix(".pending").open("wb") as f:
                            np.savez_compressed(
                                f, x=x.astype(np.float32), y=y, part=part[rows]
                            )
                        dest.with_suffix(".pending").replace(dest)
                        counts[source, "paired"] += len(rows)
    return {f"{source} {kind}": v for (source, kind), v in counts.items()}


def train():
    DEST.mkdir(exist_ok=True)
    with np.load(ROOT / "spin-population.npz") as f:
        model = SpinLocalKernel.load(f)
    counts = prepare(model)
    weights = {}
    results = {}
    for q in (-1, 1):
        lhs = np.eye(DIMENSION) * 0.06
        lhs[-1, -1] = 1e-7
        rhs = np.zeros(DIMENSION)
        total = 0.0
        n = 0
        validation = defaultdict(list)
        for f in sorted(DEST.glob(f"*-q{q}.npz")):
            with np.load(f) as a:
                x, y, part = a["x"].astype(float), a["y"], a["part"]
            use = part == 0
            b = x[use]
            target = y[use]
            lhs += b.T @ b
            rhs += b.T @ target
            n += len(target)
            total += target.sum()
            for split in (1, 2):
                use = part == split
                if use.any():
                    validation[split].append((x[use], y[use]))
        w = cho_solve(cho_factor(lhs), rhs)
        weights[str(q)] = w
        name = "IP" if q > 0 else "EA"
        results[name] = {"training_pairs": n, "training_mean_eV": total / n}
        for split, batches in validation.items():
            err = np.concatenate([x @ w - y for x, y in batches])
            base = np.concatenate([total / n - y for _, y in batches])
            results[name]["ions16_formula_holdout" if split == 1 else "chembl20"] = {
                "pairs": len(err),
                "mae_eV": float(abs(err).mean()),
                "rmse_eV": float(np.sqrt(np.mean(err**2))),
                "mean_baseline_mae_eV": float(abs(base).mean()),
            }
    np.savez_compressed(DEST / "weights.npz", **weights)
    kt.write_json(DEST / "validation.json", {"counts": counts, "metrics": results})
    print("Vertical ionization and affinity validation", results, counts, flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        train()
