"""Score soft and stretching curvatures along held-out Grambow reference modes.

Report projected signed wavenumbers at the archived coordinates. A projected
curvature probes a named reference direction; a fully relaxed normal spectrum
requires an independently optimized geometry and a full Hessian.
"""

import json
import pickle
from collections import defaultdict
import numpy as np

from experiments.full_public.specified import OUT
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField


def run():
    with (OUT / "radical-projections.pkl").open("rb") as f:
        probes = pickle.load(f)
    reports = {}
    for checkpoint in ["wb97xd3-reaction.npz", "joint-wb97xd3-reaction.npz"]:
        model = ForceField.load(OUT / checkpoint)
        bins = defaultdict(list)
        for name, i, x, prior, ref, mass in probes:
            calc = prior + x @ model.weights
            label = (
                "saddle" if name.split("/")[-1].startswith("ts") else "endpoint",
                "softest" if i == 0 else "highest",
            )

            def nu(k):
                return np.sign(k) * np.sqrt(abs(k) / mass) * TO_CM1

            bins[label].append((nu(calc) - nu(ref), np.sign(calc) == np.sign(ref)))
        table = {}
        for k, values in bins.items():
            e, s = np.asarray(values).T
            table["/".join(k)] = dict(
                n=len(e),
                mae_cm1=float(np.abs(e).mean()),
                rmse_cm1=float(np.sqrt(e @ e / len(e))),
                correct_curvature_sign_fraction=float(s.mean()),
            )
        reports[checkpoint] = table
        print(checkpoint, table, flush=True)
    (OUT / "projected-curvature-validation.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    run()
