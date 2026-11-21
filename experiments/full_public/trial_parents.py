"""Reconstruct fold-specific parent energies for fitting additional conservative kernels."""
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit as fit


def recover():
    for width, targets, soft, isomer in (
        (144, kt.TARGETS, None, 0),
        (256, ["high"], 2, 10),
        (256, ["wb97xd3"], 2, 2),
    ):
        context = {}
        for t in targets:
            d = kt.equation_data(t)
            context[t] = (d, fit.selection_mask(d, t))
        fit.fit(("base", 1.65, .65, 128, True, width, 1., True), context,
                robust=(0, 16), saddle_weight=soft, isomer_weight=isomer, cache_parent=True)


if __name__ == "__main__":
    with threadpool_limits(3):
        recover()
