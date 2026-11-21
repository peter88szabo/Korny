"""A normalized polynomial kernel on element-group angular and radial densities.

Hydrogen has its own channel. Other neighbors share a channel by valence
electron count, while pair distances retain the radii of the actual elements.
For example, a P environment can share angular correlations with N without
assuming that P–H and N–H bonds have the same length. A constant coordinate
makes the norm well-defined for isolated atoms and at fragment separation.
"""
import numpy as np
from .local import descriptor_array


class PeriodicKernel:
    def __init__(self, groups, radii, projections, centers, shells, cutoff, length, degree, weights):
        self.groups=np.asarray(groups,dtype=int)
        self.radii=np.asarray(radii,dtype=float)
        self.projections=projections
        self.centers=centers
        self.shells=np.asarray(shells,dtype=float)
        self.cutoff=np.asarray(cutoff,dtype=float)
        self.length=float(length)
        self.degree=int(degree)
        self.weights=np.asarray(weights)
        self.size=sum(map(len,centers.values()))
        if self.size!=len(self.weights) or self.degree<1 or self.length<=0:
            raise ValueError("Invalid periodic kernel parameters")

    def values(self,sites,centers,forces=False):
        """Return a local energy feature and its derivative in the projected coordinates."""
        bias=self.length**2
        xx=bias+np.einsum("ij,ij->i",sites,sites)
        cc=bias+np.einsum("ij,ij->i",centers,centers)
        norm=np.sqrt(xx[:,None]*cc)
        corr=(sites@centers.T+bias)/norm
        # Subtract each kernel's isolated-atom value to fix the atomic energy zero.
        val=corr**self.degree-(self.length/np.sqrt(cc))**self.degree
        grad=None
        if forces:
            grad=centers[None,:,:]/norm[:,:,None]-(corr/xx[:,None])[:,:,None]*sites[:,None,:]
            grad*=self.degree*corr[:,:,None]**(self.degree-1)
        return val,grad

    def design(self,z,r,forces=False):
        """Sum atomic kernels and pull their gradients back to Cartesian forces."""
        types=self.groups[z]
        radii=self.radii[z]
        offset=0
        out=np.zeros(self.size)
        jac=np.zeros((r.size,self.size)) if forces else None
        for key,centers in self.centers.items():
            sl=slice(offset,offset+len(centers))
            offset=sl.stop
            for i in np.flatnonzero(types==key):
                desc,j=descriptor_array(r,types,i,nr=self.shells.shape[1],lmax=2,
                    ron=self.cutoff[0],rc=self.cutoff[1],species=tuple(range(9)),
                    gaussian_shells=self.shells,atom_radii=radii,gradient=forces)
                proj=self.projections[key]
                v,dv=self.values((desc@proj)[None,:],centers,forces)
                out[sl]+=v[0]
                if forces:jac[:,sl]-=(dv[0]@(proj.T@j.reshape(len(desc),-1))).T
        return out,jac

    def fields(self):
        fields=dict(periodic_groups=self.groups,periodic_radii=self.radii,periodic_shells=self.shells,
            periodic_cutoff=self.cutoff,periodic_length=self.length,periodic_degree=self.degree,
            periodic_keys=list(self.centers),periodic_weights=self.weights)
        for k,c in self.centers.items():
            fields[f"periodic_centers_{k}"]=c
            fields[f"periodic_proj_{k}"]=self.projections[k]
        return fields

    @classmethod
    def load(cls,fields):
        keys=list(map(int,fields["periodic_keys"]))
        return cls(fields["periodic_groups"],fields["periodic_radii"],{k:fields[f"periodic_proj_{k}"] for k in keys},
            {k:fields[f"periodic_centers_{k}"] for k in keys},fields["periodic_shells"],fields["periodic_cutoff"],
            fields["periodic_length"],fields["periodic_degree"],fields["periodic_weights"])
