"""Additive dispersion corrections in eV and eV/Å on a bare PBE0 surface."""
from functools import partial


def correction_for(method="off", volumes=None):
    """Bind the chosen model and any external weights once per calculator."""
    method = method.lower()
    if method == "ts":
        from .ts import ts_energy_forces
        from .volumes import HirshfeldVolumes
        if volumes is None:
            raise ValueError("TS requires a separate Hirshfeld volume checkpoint")
        return partial(ts_energy_forces, volume_model=HirshfeldVolumes.load(volumes))
    if method == "off":
        return None
    if method == "d3":
        from .d3 import d3_energy_forces
        return d3_energy_forces
    raise ValueError("Dispersion must be off, ts, or d3")
