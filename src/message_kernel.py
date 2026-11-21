"""Local graph kernels for molecular energy and Cartesian forces.

A smooth radial histogram at each atom is mapped onto element-specific reference
environments. Two rounds of neighbor summation describe substitution patterns
and conjugated paths. An atom-centered kernel reads the resulting geometry
features; no discrete bond assignment or reaction label is required for MD.
"""
import numpy as np
from .bond_environment import ELEMENTS
from .four_body import RADII, switch

SHELLS=np.array([.84,1.,1.24,1.63,2.03])
WIDTHS=np.array([.095,.115,.16,.26,.35])


def graph(z,r,gradient=False,topology=False):
    """Build typed radial environments and derivatives in pair-distance coordinates."""
    keep=np.flatnonzero(np.isin(z,ELEMENTS))
    zz=z[keep]
    t=np.searchsorted(ELEMENTS,zz)
    n=len(zz)
    a,b=np.triu_indices(n,1)
    dr=r[keep[b]]-r[keep[a]]
    dist=np.linalg.norm(dr,axis=1)
    radii=np.array([RADII[int(s)] for s in zz])
    scales=radii[a]+radii[b]
    inc=dist<2.6*scales
    a,b,dr,dist,scales=(v[inc] for v in (a,b,dr,dist,scales))
    x=dist/scales
    if topology:
        # These bond occupancies are flat through normal covalent lengths.
        # A change of connectivity remains differentiable along a reaction.
        w,dw=switch(x,1.12,1.65)
        lo=np.array([1.06,1.10,1.12,1.16,1.22])
        hi=np.array([1.30,1.43,1.56,1.66,1.78])
        shell,der=switch(x[:,None],lo,hi)
        v,dv=np.column_stack((w,shell)),np.column_stack((dw,der))
    else:
        gate,dg=switch(x,1.8,2.6)
        w,dw=switch(x,1.13,1.83)
        u=(x[:,None]-SHELLS)/WIDTHS
        gauss=np.exp(-.5*u*u)
        v=np.column_stack((w,gate[:,None]*gauss))
        dv=np.column_stack((dw,gauss*(dg[:,None]-gate[:,None]*u/WIDTHS)))
    site=np.zeros((n,4,6))
    np.add.at(site,(a,t[b]),v)
    np.add.at(site,(b,t[a]),v)
    ds=None
    if gradient:
        ds=np.zeros((n,4,6,len(a)))
        e=np.arange(len(a))
        ds[a,t[b],:,e]=dv
        ds[b,t[a],:,e]=dv
        ds=ds.reshape(n,24,len(a))
    return t,site.reshape(n,24),ds,(a,b,w,dw,keep,dr,dist,scales)


def closed_walks(mat,a,b,w,dw,gradient):
    """Return local closed-walk moments and exact derivatives of normalized adjacency.

    Powers three and five respond to odd cycles. The even powers also depend on
    branching and ring closure. Degree normalization bounds long walk counts.
    """
    n=len(mat)
    degree=.5+mat.sum(axis=1)
    norm=1./np.sqrt(degree)
    adj=mat*norm[:,None]*norm
    powers=[np.eye(n),adj]
    for _ in range(2,8):
        powers.append(powers[-1]@adj)
    vals=np.column_stack([p.diagonal() for p in powers[3:]])
    if not gradient:return vals,None
    deriv=np.zeros((n,len(vals.T),len(a)))
    for i,k in enumerate(range(3,8)):
        # Differentiate each factor of B^k. The two negative terms at
        # either endpoint propagate changes in its degree normalization.
        for q in range(k):
            l,r=powers[q],powers[k-1-q]
            direct=(l[:,a]*r[:,b]+l[:,b]*r[:,a])*(norm[a]*norm[b])
            left=(l[:,a]*powers[k-q][:,a]+powers[q+1][:,a]*r[:,a])/degree[a]
            right=(l[:,b]*powers[k-q][:,b]+powers[q+1][:,b]*r[:,b])/degree[b]
            deriv[:,i,:]+=dw*(direct-.5*(left+right))
    return vals,deriv


def descriptor(z,r,seeds,gradient=False,topology=False):
    """Propagate local environment populations up to two bonds from each site."""
    typ,site,ds,geom=graph(z,r,gradient,topology)
    a,b,w,dw,keep,dr,dist,scales=geom
    n=len(typ)
    m=seeds.shape[1]
    ne=len(a)
    embed=np.zeros((n,4*m))
    if gradient:
        de=np.zeros((n,4*m,ne))
    for k in range(4):
        sel=np.flatnonzero(typ==k)
        if not len(sel):continue
        d=site[sel,None]-seeds[k]
        scale=1.25
        val=np.exp(-.5*np.einsum("nkd,nkd->nk",d,d)/scale**2)
        sl=slice(k*m,(k+1)*m)
        embed[sel,sl]=val
        if gradient:
            de[sel,sl]=-(val[:,:,None]/scale**2)*np.einsum("nkd,nde->nke",d,ds[sel])
    mat=np.zeros((n,n))
    mat[a,b]=mat[b,a]=w
    one=mat@embed
    # Remove length-two walks that immediately revisit the central atom.
    bounce=np.sum(mat*mat,axis=1)
    two=mat@one-bounce[:,None]*embed
    values=np.column_stack((site,one,two))
    if topology:
        cycles,dc=closed_walks(mat,a,b,w,dw,gradient)
        values=np.column_stack((values,cycles))
    if not gradient:return values,None,geom
    def propagate(x,dx):
        grad=(mat@dx.reshape(n,-1)).reshape(n,x.shape[1],ne) if n and ne else np.zeros_like(dx)
        edges=np.arange(ne)
        grad[a,:,edges]+=dw[:,None]*x[b]
        grad[b,:,edges]+=dw[:,None]*x[a]
        return grad
    d1=propagate(embed,de)
    d2=propagate(one,d1)-bounce[:,None,None]*de
    e=np.arange(ne)
    d2[a,:,e]-=2*(w*dw)[:,None]*embed[a]
    d2[b,:,e]-=2*(w*dw)[:,None]*embed[b]
    derivatives=[ds,d1,d2]
    if topology:derivatives.append(dc)
    return values,np.concatenate(derivatives,axis=1),geom


class MessageKernel:
    """Gaussian kernels on projected, multi-hop atomic environments."""
    def __init__(self,seeds,projections,centers,length,weights,topology=False):
        self.topology=bool(topology)
        self.seeds=np.asarray(seeds)
        self.projections=projections
        self.centers=centers
        self.length=float(length)
        self.weights=np.asarray(weights)
        self.size=sum(map(len,centers.values()))
        if self.size!=len(self.weights) or self.length<=0:
            raise ValueError("Inconsistent message-kernel basis")

    def design(self,z,r,forces=False):
        feat,j,geom=descriptor(z,r,self.seeds,forces,self.topology)
        a,b,_,_,keep,dr,dist,scales=geom
        typ=z[keep]
        values=np.zeros(self.size)
        jac=np.zeros((len(z),3,self.size)) if forces else None
        k=0
        for s in ELEMENTS:
            c=self.centers[s]
            ix=np.flatnonzero(typ==s)
            sl=slice(k,k+len(c))
            k+=len(c)
            if not len(ix):continue
            p=self.projections[s]
            u=feat[ix]@p
            d=(u[:,None]-c)/self.length
            sim=np.exp(-.5*np.einsum("ncd,ncd->nc",d,d))
            empty=np.exp(-.5*np.sum((c/self.length)**2,axis=1))
            values[sl]=sim.sum(axis=0)-len(ix)*empty
            if forces and len(a):
                du=np.einsum("td,nte->nde",p,j[ix],optimize=True)
                dx=-np.einsum("nc,ncd,nde->ec",sim,d,du,optimize=True)/self.length
                f=dx[:,None,:]*(dr/(dist*scales)[:,None])[:,:,None]
                np.add.at(jac[:,:,sl],keep[a],f)
                np.add.at(jac[:,:,sl],keep[b],-f)
        return values,jac.reshape(r.size,self.size) if forces else None

    def fields(self):
        out=dict(message_seeds=self.seeds,message_length=self.length,message_weights=self.weights,
                 message_topology=self.topology)
        for s in ELEMENTS:
            out[f"message_proj_{s}"]=self.projections[s]
            out[f"message_center_{s}"]=self.centers[s]
        return out

    @classmethod
    def load(cls,d):
        return cls(d["message_seeds"],{s:d[f"message_proj_{s}"] for s in ELEMENTS},
                   {s:d[f"message_center_{s}"] for s in ELEMENTS},d["message_length"],d["message_weights"],d.get("message_topology",False))
