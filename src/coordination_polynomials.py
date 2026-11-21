"""Finite polynomial kernels for the energy of bonds in their local environments.

A bond contributes according to its length and the smoothly weighted hydrogen
and heavy-atom coordination of both ends. The end atoms exclude each other from
their counts. A C–O distance in a carbonyl can therefore have a different force
constant from the same distance in an alcohol or a partially dissociated pair.
"""

import numpy as np
from numpy.polynomial import chebyshev

from .four_body import switch


class CoordinationPolynomialKernel:
    """Sum element-resolved bond features with a conservative Cartesian readout."""

    channels = 7
    degree = 4

    def __init__(self, radii, family, coefficients, weights, width=0.55):
        self.radii = np.asarray(radii, float)
        self.family = str(family)
        self.coefficients = np.asarray(coefficients, float)
        self.width = float(width)
        self.groups = np.full(len(self.radii), 5, int)
        for i,zs in enumerate(((1,), (6,), (7,), (8,), (9,17,35,53,85))):
            self.groups[[z for z in zs if z < len(self.radii)]] = i
        self.pairs = np.zeros((6,6),int)
        a,b=np.triu_indices(6)
        self.pairs[a,b]=self.pairs[b,a]=np.arange(len(a))
        self.size=len(a)*(self.degree+1)*self.channels
        self.weights=np.asarray(weights,float)
        if (self.family not in ("chebyshev","krawtchouk") or self.coefficients.shape!=(5,5)
            or self.weights.shape!=(self.size,) or self.width<=0):
            raise ValueError("Invalid bond polynomial coefficients or length scale")

    def _batch(self, z, positions, gradient=False):
        n,natoms,_=positions.shape
        out=np.zeros((n,21,5,self.channels))
        i,j=np.triu_indices(natoms,1)
        scale=self.radii[z[i]]+self.radii[z[j]]
        delta=positions[:,j]-positions[:,i]
        length=np.linalg.norm(delta,axis=-1)
        if np.any(length < 1e-10):
            raise ValueError("Coincident atoms in bond features")
        active=(length < 2.05*scale).any(0)
        i,j,scale=i[active],j[active],scale[active]
        length=length[:,active]
        force=np.zeros((natoms,3,21,5,self.channels)) if gradient else None
        if not len(i):
            return out.reshape(n,self.size), (force.reshape(3*natoms,self.size) if gradient else None)
        typ=self.groups[z]
        key=self.pairs[typ[i],typ[j]]
        orient=np.sign(typ[j]-typ[i])
        x=length/scale
        occ,do=switch(x,0.88,1.78)
        envelope,de=switch(x,1.40,2.05)
        u=np.tanh((x-1.22)/self.width)
        radial=chebyshev.chebvander(u,4)@self.coefficients.T
        values=envelope[:,:,None]*radial
        # Each column counts neighbors of one atom. Removing the mutual
        # contact prevents the same bond length entering twice as coordination.
        edges=np.arange(len(i))
        h=(z==1).astype(float)
        def neighbor_matrix(which):
            mat=np.zeros((len(i),natoms))
            mat[edges,i]=which[j]
            mat[edges,j]=which[i]
            return mat
        mh,mc=neighbor_matrix(h),neighbor_matrix(1-h)
        def counts(mat,which):
            cn=occ@mat
            a=cn[:,i]-occ*which[j]
            b=cn[:,j]-occ*which[i]
            return (a+b)/4,(a-b)/4
        hs,hd=counts(mh,h)
        cs,cd=counts(mc,1-h)
        marks=np.stack((np.ones_like(hs),hs,cs,hd*orient,cd*orient,hs*cs,hd**2+cd**2),axis=-1)
        if gradient:
            if n!=1: raise ValueError("Cartesian derivatives require one geometry")
            t=np.diag(do[0]/scale)
            def differential(mat,which):
                # Row e describes the dependence of bond e's local
                # environment on each distance in the list of active edges.
                a=(mat[:,i].T-np.diag(which[j]))*(do[0]/scale)
                b=(mat[:,j].T-np.diag(which[i]))*(do[0]/scale)
                return (a+b)/4,(a-b)/4
            dh,dh_diff=differential(mh,h)
            dc,dc_diff=differential(mc,1-h)
            marks_j=np.stack((np.zeros_like(t),dh,dc,orient[:,None]*dh_diff,
                              orient[:,None]*dc_diff,cs[0,:,None]*dh+hs[0,:,None]*dc,
                              2*hd[0,:,None]*dh_diff+2*cd[0,:,None]*dc_diff),axis=1)
            slope=chebyshev.chebvander(u[0],3)@chebyshev.chebder(self.coefficients.T)
            slope=(de[0,:,None]*radial[0]+envelope[0,:,None]*slope*(1-u[0,:,None]**2)/self.width)/scale[:,None]
            unit=delta[0,active]/length[0,:,None]
        for k in np.unique(key):
            idx=np.flatnonzero(key==k)
            out[:,k]=np.einsum("bem,bec->bmc",values[:,idx],marks[:,idx],optimize=True)
            if gradient:
                derivative=np.einsum("em,eck->kmc",values[0,idx],marks_j[idx],optimize=True)
                derivative[idx]+=slope[idx,:,None]*marks[0,idx,None,:]
                f=unit[:,:,None,None]*derivative[:,None]
                np.add.at(force[:,:,k],i,f)
                np.add.at(force[:,:,k],j,-f)
        return out.reshape(n,self.size), (force.reshape(3*natoms,self.size) if gradient else None)

    def frames(self,z,r):
        """Evaluate a trajectory with shared Cartesian atom order in bounded chunks."""
        z,r=np.asarray(z,int),np.asarray(r,float)
        return self._batch(z,r)[0]

    def design(self,z,r,forces=False):
        values,j=self._batch(np.asarray(z,int),np.asarray(r,float)[None],forces)
        return values[0],j

    def fields(self):
        return dict(coordpoly_family=self.family,coordpoly_radii=self.radii,coordpoly_coeff=self.coefficients,
                    coordpoly_width=self.width,coordpoly_weights=self.weights)

    @classmethod
    def load(cls,d):
        return cls(d["coordpoly_radii"],str(d["coordpoly_family"]),d["coordpoly_coeff"],d["coordpoly_weights"],d["coordpoly_width"])
