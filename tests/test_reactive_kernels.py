"""Guard kernel derivatives and connected chemistry on rotating, permuted molecules."""
from io import BytesIO
import unittest
import numpy as np
from src.chemical import ChemicalEnvironment
from src.molecular_kernel import MolecularKernel


class ReactiveKernelTests(unittest.TestCase):
    def test_symmetric_conservative_molecular_kernel(self):
        r=np.array([[-.8,.2,.1],[0.,0.,0.],[.54,.68,.02]])
        z=np.array([1,8,1]); perm=np.array([2,1,0]);rng=np.random.default_rng(28)
        centers=MolecularKernel.descriptors(r[None,:,:]+rng.normal(0,.06,(4,3,3)))
        model=MolecularKernel(z,centers,[np.arange(3),perm],.45,rng.normal(size=4))
        buf=BytesIO();model.save(buf);buf.seek(0);model=MolecularKernel.load(buf)
        e,f=model.predict(z,r);p,fp=model.predict(z[perm],r[perm]+1.25)
        self.assertAlmostEqual(e,p,places=10)
        np.testing.assert_allclose(f[perm],fp,atol=1e-10)
        np.testing.assert_allclose(f.sum(axis=0),0.,atol=1e-10)
        step=1e-5;direction=rng.normal(size=r.shape)
        slope=(model.predict(z,r-step*direction)[0]-model.predict(z,r+step*direction)[0])/(2*step)
        self.assertAlmostEqual(slope,(direction*f).sum(),places=6)
        with self.assertRaises(ValueError):model.predict(z,r,0,3)

    def test_benzene_ring_closure_and_derivative(self):
        chem=ChemicalEnvironment(schema='smooth-organic-groups/1')
        phi=np.arange(6)*np.pi/3
        circle=np.column_stack((np.cos(phi),np.sin(phi),phi*0))
        z=np.array([6]*6+[1]*6);r=np.vstack((1.40*circle,2.48*circle))
        h,j=chem.design(z,r,gradient=True);i=chem.slices['benzene_like_ring'].start
        self.assertGreater(h[i],.96)
        self.assertLessEqual(h[i],1.00001)
        old=ChemicalEnvironment(schema='smooth-oxygen-groups/3')
        np.testing.assert_allclose(h[:old.size],old.design(z,r)[0])
        direction=np.random.default_rng(8).normal(size=r.shape)
        fd=(chem.design(z,r-1e-5*direction)[0]-chem.design(z,r+1e-5*direction)[0])/2e-5
        np.testing.assert_allclose(j.T@direction.ravel(),fd,atol=3e-5)
        perm=np.random.default_rng(17).permutation(len(r))
        np.testing.assert_allclose(chem.design(z[perm],r[perm]+2)[0],h,atol=1e-10)
        r[[0,6],2]+=3.
        self.assertEqual(chem.design(z,r)[0][i],0.)

    def test_nitrile_does_not_need_oxygen(self):
        chem=ChemicalEnvironment(schema='smooth-organic-groups/1')
        z=np.array([6,6,7,1,1,1]);r=np.array([[-1.46,0,0],[0,0,0],[1.16,0,0],[-1.80,1,0],[-1.80,-.5,.87],[-1.80,-.5,-.87]])
        h=chem.design(z,r)[0]
        self.assertGreater(h[chem.slices['nitrile'].start],.8)
        self.assertEqual(h[chem.slices['amide_link'].start],0.)
