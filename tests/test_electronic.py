"""Regression checks for charge/spin selection, conservative forces, and checkpoints."""
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np

from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField

BACKBONE = Path(__file__).resolve().parents[1]/"experiments/reactivity/pbe0-core-9325855e31063235.npz"


class StateTests(unittest.TestCase):
    def setUp(self):
        self.base = ForceField.load(BACKBONE)
        self.z = np.array([6, 1, 1])
        self.r = np.array([[0., 0., 0.], [1.08, .1, 0.], [-.4, .98, 0.]])

    def model(self):
        states = [(0, 1), (0, 3), (-1, 2), (1, 2)]
        features = ElectronicFeatures(self.base.pair.species, environment=True)
        weights = np.random.default_rng(5).normal(scale=.10, size=(len(states), features.size))
        self.base.electronic = ElectronicCorrection(self.base.pair.species, states, weights,
                                                     "PBE0", environment=True)
        return self.base

    def test_state_changes_the_potential_and_its_force(self):
        model = self.model()
        forces = {}
        for state in model.supported_states:
            e, f = model.predict(self.z, self.r, *state)
            self.assertAlmostEqual(e, model.energy(self.z, self.r, *state), places=10)
            d = .00002
            cartesian = np.eye(self.r.size).reshape(-1, *self.r.shape)
            fd = [(model.energy(self.z,self.r-u*d,*state)-model.energy(self.z,self.r+u*d,*state))/(2*d)
                  for u in cartesian]
            np.testing.assert_allclose(f.ravel(), fd, atol=2e-6)
            np.testing.assert_allclose(f.sum(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(np.cross(self.r,f).sum(axis=0), 0, atol=1e-10)
            forces[state] = f
        self.assertFalse(np.allclose(forces[0,1],forces[0,3]))
        self.assertFalse(np.allclose(forces[-1,2],forces[1,2]))

    def test_round_trip_and_invalid_states(self):
        with self.assertRaises(ValueError):
            self.base.predict(self.z, self.r, 0, 3)
        model = self.model()
        buf = BytesIO()
        model.save(buf)
        buf.seek(0)
        loaded = ForceField.load(buf)
        self.assertEqual(loaded.supported_states, model.supported_states)
        self.assertIsNotNone(loaded.repulsion)
        for state in model.supported_states:
            expected = model.predict(self.z,self.r,*state)
            actual = loaded.predict(self.z[::-1],self.r[::-1],*state)
            self.assertAlmostEqual(expected[0],actual[0],places=10)
            np.testing.assert_allclose(expected[1][::-1],actual[1],atol=1e-10)
        for q,m in ((0,2),(0,0),(-1,1),(0,1.5),(True,2),(0,5)):
            with self.subTest(charge=q,multiplicity=m), self.assertRaises(ValueError):
                loaded.energy(self.z,self.r,q,m)
