"""Linear energy derivatives at NIST training minima for the final regression.

The references give frequencies without Cartesian normal-mode vectors. Use the
parent's internal modes as directions, with the published squared frequencies
as eigenvalue targets. Also enforce the measured zero gradient in Cartesian
coordinates. Independent relaxation and diagonalization test this approximation.
"""

import json
import numpy as np
from experiments.full_public.specified import OUT
from experiments.reactivity.larger_vibrations import references
from experiments.reactivity.stationary_fit import POINTS, MASSES, internal_transform
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField


def run():
    model = ForceField.load(OUT / "pbe0-reaction.npz")
    points = json.loads(POINTS.read_text())
    points.update(references())
    train = json.loads((OUT / "stationary.json").read_text())["training"]
    dim = len(model.weights) + 1
    gram = np.zeros((dim, dim))
    for name in train:
        info = points[name]
        z, mass = np.array([MASSES[a] for a in info["atoms"]]).T
        z = z.astype(int)
        r = np.array(info["positions_A"])
        freq = np.sort(info["frequencies_cm1"])
        t = internal_transform(mass, r)

        def forces(x):
            v, j = model.design(z, x, 0, 1, True)
            return np.column_stack((j, model.prior(z, x, 0, 1, True)[1].ravel()))

        # Last column stores the target residual, including the frozen prior.
        a = forces(r)
        a[:, -1] *= -1
        a *= 12.0 / np.sqrt(r.size)
        gram += a.T @ a
        axes = np.eye(r.size).reshape(-1, *r.shape)
        step = 0.001
        h = np.array(
            [(forces(r - step * d) - forces(r + step * d)) / (2 * step) for d in axes]
        )
        h = (h + h.transpose(1, 0, 2)) / 2
        dyn = np.einsum("ai,abw,bj->ijw", t, h, t, optimize=True)
        _, u = np.linalg.eigh(dyn[:, :, -1])
        projected = np.einsum("im,ijw,jm->mw", u, dyn, u, optimize=True)
        projected[:, -1] = (freq / TO_CM1) ** 2 - projected[:, -1]
        projected *= (
            4.0 * TO_CM1**2 / (2 * np.maximum(350.0, freq) * 120 * np.sqrt(len(freq)))
        )[:, None]
        gram += projected.T @ projected
    np.save(OUT / "nist-derivative-gram.npy", gram)
    print("NIST geometry and harmonic-frequency targets", len(train), flush=True)


if __name__ == "__main__":
    run()
