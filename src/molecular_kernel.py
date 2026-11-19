"""Symmetrized inverse-distance Matérn potential for one fixed molecular atom map.

The model sums over the supplied equivalent-atom permutations and predicts a
scalar energy. Analytical differentiation gives conservative Cartesian forces.
Coordinates are in Å; Korny energies and forces are in eV and eV/Å.
"""
import numpy as np


class MolecularKernel:
    def __init__(self,z,centers,perms,length,weights=None,offset=0.,state=(0,1)):
        self.z=np.asarray(z,int);self.centers=np.asarray(centers,float)
        self.perms=np.asarray(perms,int);self.length=float(length)
        self.state=tuple(state);self.offset=float(offset)
        self.weights=np.zeros(len(centers)) if weights is None else np.asarray(weights,float)
        self.i,self.j=np.triu_indices(len(z),1)
        if (not np.isfinite(self.length) or self.length<=0 or self.weights.ndim!=1
                or self.centers.shape!=(len(self.weights),len(self.i)) or self.perms.ndim!=2 or not len(self.perms)
                or not np.isfinite(self.centers).all() or not np.isfinite(self.weights).all() or not np.isfinite(self.offset)):
            raise ValueError('Invalid molecular kernel basis')
        if any(sorted(p)!=list(range(len(z))) or np.any(self.z[p]!=self.z) for p in self.perms):
            raise ValueError('Permutations must preserve the atom identities')
        matrix=np.zeros((len(z),len(z)),int)
        matrix[self.i,self.j]=matrix[self.j,self.i]=np.arange(len(self.i))
        order=np.array([matrix[p[self.i],p[self.j]] for p in self.perms])
        self.images=self.centers[:,order]

    @staticmethod
    def descriptors(r):
        r=np.asarray(r,float)
        i,j=np.triu_indices(r.shape[-2],1)
        return 1/np.linalg.norm(r[...,i,:]-r[...,j,:],axis=-1)

    def design(self,r):
        """Return kernel values and their negative Cartesian derivatives."""
        r=np.asarray(r,float)
        if r.shape!=(len(self.z),3):
            raise ValueError('Supply one Cartesian triplet for every atom in the model')
        bond=r[self.i]-r[self.j]
        dis=np.linalg.norm(bond,axis=1)
        if not np.isfinite(dis).all() or np.any(dis<=0):
            raise ValueError('Coordinates must be finite with distinct atom positions')
        delta=(1/dis)-self.images
        a2=5/self.length**2
        t=np.sqrt(a2*np.einsum('cpi,cpi->cp',delta,delta))
        e=np.exp(-t)
        h=((1+t+t*t/3)*e).mean(axis=1)
        # d(1/r)/dR_i = -(R_i-R_j)/r^3. Opposite forces on j enforce
        # exact translation invariance and conserve total linear momentum.
        dh=-a2/(3*len(self.perms))*np.einsum('cp,cpi->ci',(1+t)*e,delta)
        pair=dh.T[:,None,:]*(bond/dis[:,None]**3)[:,:,None]
        f=np.zeros((len(r),3,len(h)))
        np.add.at(f,self.i,pair);np.add.at(f,self.j,-pair)
        return h,f.reshape(r.size,-1)

    def predict(self,z,r,charge=0,multiplicity=1):
        if not np.array_equal(z,self.z) or (charge,multiplicity)!=self.state:
            raise ValueError('This molecular kernel requires its training atom map and electronic state')
        v,f=self.design(r)
        return float(v@self.weights+self.offset),(f@self.weights).reshape(-1,3)

    def save(self,path):
        np.savez_compressed(path,z=self.z,centers=self.centers,perms=self.perms,length=self.length,weights=self.weights,offset=self.offset,state=self.state)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            return cls(**dict(d))
