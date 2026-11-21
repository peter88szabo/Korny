"""Make weighted sums of compatible Korny checkpoints without duplicating D3.

The models share the PBE0 backbone. Group equivalent kernel feature maps by
their serialized parameters, then combine their readout vectors. Within one
call the prediction computes each distinct local feature map once.
"""
from collections import defaultdict
import hashlib
import numpy as np
from experiments.full_public import kernel_trials as kt
from src.forcefield import ForceField


def signature(basis):
    digest=hashlib.sha256()
    fields=basis.fields()
    for k in sorted(fields):
        if k.endswith("weights"):continue
        val=np.asarray(fields[k])
        digest.update(repr((k,val.dtype.str,val.shape)).encode())
        digest.update(val.tobytes())
    return type(basis),digest.digest()


def molecular_weights(branches,values):
    amounts=defaultdict(float,{12:1.})
    for (child,parent),v in zip(branches,values):
        amounts[child]+=v
        amounts[parent]-=v
    return dict(amounts)


class Components:
    """Shared model pieces for a matrix of full scalar potentials or their derivatives."""
    def __init__(self,target,stages):
        self.stages=list(stages)
        models=[ForceField.load(kt.WORK/f"p{s}-public-{target}.npz",dispersion="d3") for s in self.stages]
        self.model=models[0]
        self.spin=np.column_stack([m.weights for m in models])
        banks={}
        for j,(stage,model) in enumerate(zip(stages,models)):
            # All models use the same underlying electronic/repulsive baseline.
            np.testing.assert_array_equal(self.model.extension_core.coefficients,model.extension_core.coefficients)
            for basis in (model.reactive_basis,*model.additional_bases):
                sig=signature(basis)
                if sig not in banks:banks[sig]=basis,np.zeros((basis.size,len(stages)))
                banks[sig][1][:,j]+=basis.weights
        self.banks=list(banks.values())

    def readout(self,z,r):
        model=self.model
        v=model.design(z,r,0,1,True)[1]@self.spin
        for basis,coeff in self.banks:
            v+=basis.design(z,r,True)[1]@coeff
        # Each column contains a gradient of a correction to the same baseline.
        return v

    def checkpoint(self,amounts,path):
        weights=np.array([amounts.get(s,0.) for s in self.stages])
        np.testing.assert_allclose(weights.sum(),1.,atol=1.e-12)
        m=self.model
        m.weights=self.spin@weights
        m.reactive_basis=None
        m.additional_bases=[]
        for basis,coeff in self.banks:
            out=coeff@weights
            if np.any(out):
                basis.weights=out
                m.additional_bases.append(basis)
        m.save_with_base(kt.original.BASE,path)
