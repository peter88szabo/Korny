"""Profile bond basis size and unstable-mode training weight on inner development formulas."""

import json
import shutil
from threadpoolctl import threadpool_limits
import numpy as np
from experiments.full_public import trial_fit as t
from experiments.full_public import kernel_trials as kt


def run(stage=9):
    ctx = {
        s: (d, t.selection_mask(d, s)) for s in kt.TARGETS if (d := kt.equation_data(s))
    }
    trials = {}
    if stage == 9:
        options = [
            (n, s, w, 0)
            for n, s, w in (
                (144, 1, 16),
                (144, 4, 16),
                (256, 1, 16),
                (256, 4, 16),
                (256, 12, 16),
            )
        ]
    else:
        options = [
            (144, 1, 16, 0),
            (256, 1, 16, 0),
            (144, 2, 16, 2),
            (256, 2, 16, 2),
            (256, 2, 16, 10),
            (256, 2, 16, 40),
        ]
    for count, soft, weight, isomer in options:
        setting = ["base", 1.65, 0.65, 128, True, count, 1.0, True]
        key = t.fmt(setting) + f"-public-h0-s{weight}-soft{soft}"
        if isomer:
            key += f"-isomer{isomer:g}"
        path = kt.WORK / (key + ".json")
        # A change to the saddle derivative objective only affects the radical branch.
        subset = ctx if soft == 1 or isomer else {"wb97xd3": ctx["wb97xd3"]}
        trials[key] = (
            json.loads(path.read_text())
            if path.exists()
            else t.fit(
                setting,
                subset,
                robust=(0, weight),
                saddle_weight=soft,
                isomer_weight=isomer,
            )
        )
    high = min(
        (k for k, r in trials.items() if "high" in r),
        key=lambda k: trials[k]["high"]["score"],
    )
    best = min(r["wb97xd3"]["score"] for r in trials.values())
    # Within a one-kcal/mol margin of the best inner reaction score, prioritize
    # agreement with the reference imaginary-mode curvature on inner formulas.
    front = [k for k, r in trials.items() if r["wb97xd3"]["score"] <= best + 1.0]
    radical = min(
        front, key=lambda k: np.mean(trials[k]["wb97xd3"]["development_saddle_mae_cm1"])
    )
    selected = {"high": high, "wb97xd3": radical}
    result = {}
    for target, key in selected.items():
        for ext in (".npz", "-equations.npz"):
            shutil.copyfile(
                kt.WORK / (key + "-" + target + ext),
                kt.WORK / (f"p{stage}-public-" + target + ext),
            )
        result[target] = trials[key][target]
    kt.write_json(kt.WORK / f"p{stage}-public.json", result)
    kt.write_json(
        kt.WORK / f"p{stage}-public-settings.json",
        dict(
            selected=f"p{stage}-public",
            branches=selected,
            development={
                k: dict(
                    energy=v["wb97xd3"]["score"],
                    saddle=np.mean(v["wb97xd3"]["development_saddle_mae_cm1"]),
                )
                for k, v in trials.items()
            },
        ),
    )
    print("Completed saddle and reaction fit", selected, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(9, 10), default=9)
    with threadpool_limits(3):
        run(parser.parse_args().stage)
