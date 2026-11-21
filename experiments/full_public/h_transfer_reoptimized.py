"""Learn a separate H-transfer feature direction at each reoptimization method.

Use atom-balanced reactions at PM6, HF and r²SCAN-3c. Energies of all fragments
on a side are summed before differencing. Only the training readout direction
is transferred to the precision matrix of the higher-level fits. This lets the
primary coefficients select the magnitude and sign warranted by their labels.
"""

from collections import defaultdict
import sqlite3

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import amk_correction_data as archive
from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_reoptimized import (
    atom_counts,
    complete_sides,
    METHODS,
)
from experiments.full_public.trial_h_transfer import ROOT, ANGULAR, bases, dual_features


def prepare(angular=False):
    directory = ANGULAR if angular else ROOT
    directory.mkdir(exist_ok=True)
    models = bases(angular)
    dirs = {f: [] for f in models}
    counts = {}
    with sqlite3.connect(f"file:{archive.DATABASE}?mode=ro", uri=True) as db:
        for method in METHODS:
            records = db.execute(
                """select case_id,role,split,z,r,published_baseline_eV
                from frames where dataset='RDB7' and method=? and split in ('train','validation')
                and q=0 and m=1 order by member,frame""",
                (method,),
            ).fetchall()
            points = [
                (np.frombuffer(v[3], "u1"), np.frombuffer(v[4], "<f8").reshape(-1, 3))
                for v in records
            ]
            formula = np.stack([atom_counts(z) for z, r in points])
            y = np.array([v[-1] for v in records])
            cases = defaultdict(lambda: defaultdict(list))
            for i, (case, role, split, *_) in enumerate(records):
                cases[case, split][role].append(i)
            parts = [
                dual_features(points[i : i + 128], models)
                for i in range(0, len(points), 128)
            ]
            tally = defaultdict(int)
            for f in models:
                x = np.vstack([p[f] for p in parts])
                train = []
                for (case, split), roles in cases.items():
                    r, t, p = (
                        roles[k] for k in ("reactant", "transition_state", "product")
                    )
                    if len(t) != 1:
                        continue
                    a, b = complete_sides(r, p, formula, t[0])
                    if a or b:
                        if f == next(iter(models)):
                            tally[split] += 1
                        if split == "train":
                            for ok, idx in ((a, r), (b, p)):
                                if ok:
                                    train.append(
                                        np.r_[
                                            x[t[0]] - x[idx].sum(0),
                                            y[t[0]] - y[idx].sum(),
                                        ]
                                    )
                a = np.array(train)
                w = cho_solve(
                    cho_factor(a[:, :-1].T @ a[:, :-1] + 60 * np.eye(x.shape[1])),
                    a[:, :-1].T @ a[:, -1],
                )
                dirs[f].append(w / max(np.linalg.norm(w), 1e-12))
            counts[method] = dict(tally)
            print(
                "Reoptimized training reaction IDs and validation IDs",
                method,
                dict(tally),
                flush=True,
            )
    np.savez(
        directory / "reoptimized-directions.npz",
        **{f: np.stack(w) for f, w in dirs.items()},
    )
    kt.write_json(directory / "reoptimized-counts.json", counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angular", action="store_true")
    prepare(parser.parse_args().angular)
