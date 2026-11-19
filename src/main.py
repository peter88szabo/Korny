"""Command-line checks and compatibility imports for the original reference script."""
if not __package__:
    # Support running python src/main.py from a source checkout.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .checks import check, time_gaussian
from .dispersion.d3 import d3_energy_gradient_bohr
from .dispersion.mbd import mbd_energy_bohr
from .dispersion.ts import TS_FREE_ATOMS, ts_energy
from .jets import Jet, bilinear, jet, join
from .local import (Molecules, SparseGaussian, TaylorTT, descriptor, descriptor_array,
                       descriptor_layout, eig_factor, gaussian_rows, monomial_layout,
                       monomial_values, multiindices, pivots, polynomial, refine_tt, tensor_sweep)


if __name__ == "__main__":
    import sys
    if "--timing" in sys.argv:
        time_gaussian()
    else:
        check()
