"""Training-only stationarity and harmonic curvature rows for an additive correction."""

import json
import numpy as np
from experiments.full_public.data import OUT
from experiments.reactivity.stationary_fit import POINTS, MASSES, internal_transform
from experiments.reactivity.larger_vibrations import references
from experiments.reactivity.frequency_validation import TO_CM1


def nist_rows(design, parent, force_cache=None):
    """Use physical force and curvature residuals at the 14 training minima.

    The parent is loaded with D3. Frequencies refer to the bare PBE0 NIST
    calculation and serve as auxiliary shape targets on the higher-level surface.
    Reference eigenvectors were not tabulated, so project onto parent modes.
    """
    points = json.loads(POINTS.read_text())
    points.update(references())
    result = []
    for name in json.loads((OUT / "stationary.json").read_text())["training"]:
        p = points[name]
        z, mass = np.array([MASSES[a] for a in p["atoms"]]).T
        z = z.astype(int)
        r = np.array(p["positions_A"])
        freq = np.sort(p["frequencies_cm1"])
        t = internal_transform(mass, r)
        step = 0.001

        def matrix(pos):
            j = design(z, pos)
            if force_cache is None:
                bf = parent.predict(z, pos, 0, 1)[1]
            else:
                # A caller comparing multiple bases on the same frozen parent
                # can reuse its forces at the identical displaced geometries.
                key = (z.tobytes(), pos.tobytes())
                if key not in force_cache:
                    force_cache[key] = parent.predict(z, pos, 0, 1)[1]
                bf = force_cache[key]
            return np.column_stack((j, bf.ravel()))

        a = matrix(r)
        a[:, -1] *= -1
        result.append(12 * a / np.sqrt(r.size))
        # Directional Cartesian Hessians projected into the mass-weighted normal modes.
        h = np.stack(
            [
                (matrix(r - step * v) - matrix(r + step * v)) / (2 * step)
                for v in np.eye(r.size).reshape(-1, *r.shape)
            ]
        )
        dyn = np.einsum(
            "ai,abw,bj->ijw", t, 0.5 * (h + h.transpose(1, 0, 2)), t, optimize=True
        )
        _, axes = np.linalg.eigh(dyn[:, :, -1])
        curv = np.einsum("im,ijw,jm->mw", axes, dyn, axes, optimize=True)
        curv[:, -1] = (freq / TO_CM1) ** 2 - curv[:, -1]
        scale = 4 * TO_CM1**2 / (2 * np.maximum(350.0, freq) * 120 * np.sqrt(len(freq)))
        result.append(curv * scale[:, None])
    return np.vstack(result)
