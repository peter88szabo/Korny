"""Check the two-hop bond kernel during bond breaking and fragment separation."""
import numpy as np
import pytest
from src.bond_order_kernel import BondOrderKernel, environments


@pytest.mark.parametrize("gated",[False,True])
def test_force_order_and_additivity(gated):
    # The O-H-O bridge samples derivatives of the switching functions and the
    # C-C bond checks averaging over equivalent endpoint labels.
    z = np.array([8, 1, 8, 6, 6, 1, 7])
    r = np.array([[0.,0.,0.],[1.06,.1,.05],[2.28,.1,-.03],
                  [-.6,1.2,.3],[-1.9,1.8,.16],[-.4,1.6,1.2],[-2.7,1.7,1.35]])
    rng = np.random.default_rng(5184)
    samples = {k: x for k,x,*_ in environments(z, r)}
    offsets = [0]
    centers = []
    for k in range(10):
        if k in samples:
            centers.extend(samples[k]+rng.normal(0,.12,samples[k].shape))
        offsets.append(len(centers))
    model = BondOrderKernel(centers, offsets, .83)
    if gated:
        # Exercise both factors in the product rule away from the gate's own
        # width, using a nonsymmetric metric to check the gradient pullback.
        model.transforms = np.eye(25)[None,:,:] + rng.normal(0,.02,(10,25,25))
        model.gate_coefficients = rng.normal(size=len(centers))
        model.gate_bias = rng.normal(size=10)
        model = BondOrderKernel.load(model.fields())
    v,j = model.design(z,r,True)
    order = rng.permutation(len(r))
    rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
    new,df = model.design(z[order],r[order]@rotation+3.,True)
    np.testing.assert_allclose(v,new,atol=1.e-10)
    force = j.reshape(len(r),3,-1)
    np.testing.assert_allclose(df.reshape(force.shape),np.einsum('iab,ac->icb',force[order],rotation),atol=1.e-10)
    np.testing.assert_allclose(force.sum(axis=0),0,atol=1.e-12)
    delta = rng.normal(size=r.shape)
    fd = (model.design(z,r-1.e-6*delta)[0]-model.design(z,r+1.e-6*delta)[0])/2.e-6
    np.testing.assert_allclose(fd,delta.ravel()@j,atol=1.e-7)
    both,forces = model.design(np.tile(z,2),np.vstack((r,r+30.)),True)
    np.testing.assert_allclose(both,2*v,atol=1.e-10)
    np.testing.assert_allclose(forces,np.tile(j,(2,1)),atol=1.e-10)
