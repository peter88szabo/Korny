"""Check conservative graph forces, fragment limits and checkpoint field recovery."""
import numpy as np
import pytest
from src.bond_environment import ELEMENTS
from src.message_kernel import MessageKernel, descriptor


@pytest.mark.parametrize("topology",[False,True])
def test_graph_forces_through_a_changing_ring(topology):
    rng=np.random.default_rng(139)
    z=np.array([6,6,6,8,1])
    r=np.array([[0.,0.,0.],[1.41,.3,.07],[.54,1.65,-.13],[1.76,1.75,.16],[-.63,-.8,.03]])
    seeds=rng.normal(0,.35,(4,3,24))
    features,_,_=descriptor(z,r,seeds,topology=topology)
    projection={s:rng.normal(0,.15,(features.shape[1],7)) for s in ELEMENTS}
    centers={s:rng.normal(0,.2,(3,7)) for s in ELEMENTS}
    model=MessageKernel(seeds,projection,centers,.9,rng.normal(size=12),topology)
    model=MessageKernel.load(model.fields())
    val,j=model.design(z,r,True)
    direction=rng.normal(size=r.shape)
    step=1.e-6
    fd=(model.design(z,r-step*direction)[0]-model.design(z,r+step*direction)[0])/(2*step)
    np.testing.assert_allclose(j.T@direction.ravel(),fd,atol=2.e-8)
    order=rng.permutation(len(z))
    rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
    new,df=model.design(z[order],r[order]@rotation+1.,True)
    np.testing.assert_allclose(new,val,atol=1.e-12)
    np.testing.assert_allclose(df.reshape(len(z),3,-1),np.einsum("nik,ij->njk",j.reshape(len(z),3,-1)[order],rotation),atol=1.e-11)
    twice,forces=model.design(np.tile(z,2),np.vstack([r,r+40]),True)
    np.testing.assert_allclose(twice,2*val,atol=1.e-12)
    np.testing.assert_allclose(forces,np.tile(j,(2,1)),atol=1.e-12)
