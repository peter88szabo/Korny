"""Regression tests for same-site functional groups on a reactive surface."""
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np

from src.chemical import ChemicalEnvironment
from src.forcefield import ForceField


class ChemicalTests(unittest.TestCase):
    def test_formyl_requires_neighbors_on_one_carbon(self):
        chem = ChemicalEnvironment()
        # Isolated model fragments suffice for geometric feature tests.
        z = np.array([8,6,1,6])
        r = np.array([[1.22,0.,0.], [0.,0.,0.], [-.55,.94,0.], [-.76,-1.30,0.]])
        row = chem.design(z,r)[0]
        index = chem.slices["carbon_substituted_formyl"].start
        self.assertGreater(row[index], .99)
        # Leave a C–H and a C=O in the structure but on separate carbon atoms.
        r[3] += [7.,0.,0.]
        r[2] = r[3] + [1.09,0.,0.]
        row = chem.design(z,r)[0]
        self.assertEqual(row[index], 0.)
        self.assertEqual(row[chem.slices["formyl"].start], 0.)
        self.assertGreater(row[chem.slices["carbonyl"].start], .99)

    def test_hydroperoxide_invariance_and_conservative_forces(self):
        base = Path(__file__).resolve().parents[1]/"experiments/reactivity/matern-15d11dacaf49a610.npz"
        model = ForceField.load(base)
        chem = ChemicalEnvironment()
        rng = np.random.default_rng(123)
        model.chemical = ChemicalEnvironment({s:rng.normal(0.,.1,chem.size) for s in model.supported_states})
        buf = BytesIO()
        model.save(buf)
        buf.seek(0)
        restored = ForceField.load(buf)
        z = np.array([6,8,8,1])
        r = np.array([[0.,0.,0.],[1.44,0.,0.],[1.87,1.54,.03],[2.81,1.50,.08]])
        row,jac = chem.design(z,r,gradient=True)
        self.assertGreater(row[chem.slices["organic_hydroperoxide"].start],.9)
        for i,u in enumerate(np.eye(r.size).reshape(-1,*r.shape)):
            h = 1e-5
            fd = (chem.design(z,r-h*u)[0]-chem.design(z,r+h*u)[0])/(2*h)
            np.testing.assert_allclose(jac[i],fd,atol=2e-7)
        perm = [2,3,1,0]
        rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        for state in ((0,2),(-1,1)):
            en,force = model.predict(z,r,*state)
            self.assertAlmostEqual(restored.energy(z,r,*state),en,places=9)
            other,rot = restored.predict(z[perm],(r@rotation+2)[perm],*state)
            self.assertAlmostEqual(other,en,places=9)
            np.testing.assert_allclose(rot,(force@rotation)[perm],atol=2e-8)

    def test_oxygen_bond_cutoff_has_continuous_curvature(self):
        chem = ChemicalEnvironment()
        z = np.array([8,8])
        idx = chem.slices["oxygen_oxygen"].start
        def slope(d):
            r = np.array([[0.,0.,0.],[d,0.,0.]])
            return -chem.design(z,r,gradient=True)[1][3,idx]
        h = 1e-5
        for cutoff in (1.55,2.05):
            self.assertAlmostEqual(slope(cutoff),0.,places=10)
            self.assertLess(abs((slope(cutoff+h)-slope(cutoff-h))/(2*h)),1e-5)

    def test_shared_hydrogen_distinguishes_endpoints_from_transfer(self):
        chem = ChemicalEnvironment(schema="smooth-oxygen-groups/2")
        z = np.array([6,1,8,8])
        r = np.array([[0.,0.,0.],[1.25,.05,0.],[2.50,0.,0.],[3.12,1.30,.05]])
        idx = chem.slices["carbon_H_peroxy"]
        x,j = chem.design(z,r,gradient=True)
        self.assertGreater(x[idx.start], .8)
        u=np.array([[0.,0.,0.],[.3,.4,.5],[0.,0.,0.],[0.,0.,0.]])
        fd = (chem.design(z,r-1e-5*u)[0]-chem.design(z,r+1e-5*u)[0])/2e-5
        np.testing.assert_allclose(u.ravel()@j[:,idx],fd[idx],atol=2e-7)
        separated=r.copy()
        separated[2:]+= [2.,0.,0.]
        np.testing.assert_array_equal(chem.design(z,separated)[0][idx],0.)
        # A bond to a DIFFERENT hydrogen must not count as a shared H bridge.
        separated=np.vstack((separated,separated[2]+[.97,0.,0.]))
        self.assertEqual(chem.design(np.r_[z,1],separated)[0][idx.start],0.)

    def test_hydrogen_bond_and_torsion_derivatives(self):
        chem=ChemicalEnvironment(schema="smooth-oxygen-groups/3")
        z=np.array([6,6,8,8,1,8,1])
        r=np.array([[-1.25,.6,.1],[.05,.05,0],[.5,1.34,.15],[1.43,1.65,1.20],
                    [1.50,2.58,1.62],[2.20,4.10,1.94],[2.1,4.55,2.83]])
        x,j=chem.design(z,r,gradient=True)
        self.assertGreater(x[chem.slices["hbond_OH_O"].start],.5)
        self.assertGreater(abs(x[chem.slices["torsion_CCOO"].start]),.1)
        for i,u in enumerate(np.eye(r.size).reshape(-1,*r.shape)):
            fd=(chem.design(z,r-1e-5*u)[0]-chem.design(z,r+1e-5*u)[0])/(2e-5)
            np.testing.assert_allclose(j[i],fd,atol=2e-7)
        # An antipodal reflection preserves the energy, including every
        # dihedral feature and its parity-correct transformed derivative.
        reflected,df=chem.design(z,-r,gradient=True)
        np.testing.assert_allclose(reflected,x,atol=1e-10)
        np.testing.assert_allclose(df,-j,atol=1e-10)
