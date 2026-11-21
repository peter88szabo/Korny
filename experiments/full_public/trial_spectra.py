"""Relax the same four NIST molecules with D3 on and check each fitted PES Hessian."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
from experiments.full_public.check_spectra import case, panels
from experiments.full_public.kernel_trials import WORK, write_json


def evaluate_trial(stage):
    jobs = [
        (f"kernel-trials/{stage}-high.npz", "d3", name, point)
        for name, point in panels().items()
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        result = {name: r for _, name, r in pool.map(case, jobs)}
    write_json(WORK / f"{stage}-spectra.json", result)


if __name__ == "__main__":
    evaluate_trial(sys.argv[1])
