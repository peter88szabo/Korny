"""Smooth functional-group features for conservative reactive force fields.

Each feature sums over element-matched paths or branched groups of distinct
atoms. A group's weight includes the product of its smooth bond weights. Thus a carbonyl and a
C–H bond on different carbons cannot activate the same formyl path. The weights
also multiply bond-stretch and bond-angle features. This gives the fitted
energy independent slopes and curvatures in different chemical environments.

The weights are geometry descriptors. Neither a formal bond order nor a
permanent functional-group assignment is needed during a trajectory.
"""
import numpy as np

from .electronic import state_key
from .organic import OrganicBasis


# (element pair, reference length, start of switching, outer cutoff), all in Å.
# The broad covalent switches overlap the bond-breaking region. The short C–O
# channel has an additional Gaussian envelope to distinguish carbonyl-like C–O.
BONDS = {
    "OH": ((8, 1), .97, 1.05, 1.65),
    "OO": ((8, 8), 1.46, 1.55, 2.05),
    "CO": ((6, 8), 1.43, 1.52, 2.05),
    "short_CO": ((6, 8), 1.22, 1.52, 2.05),
    "CH": ((6, 1), 1.09, 1.18, 1.75),
    "CC": ((6, 6), 1.52, 1.65, 2.15),
    "NH": ((7, 1), 1.02, 1.11, 1.65),
    "H_acceptor_O": ((1, 8), 1.90, 2.05, 3.15),
}

# The on-disk schema fixes this order and the constants above. The C–O–O and
# C–O–O–H paths respond to organic peroxy and hydroperoxy environments, while
# the C–O–H, short O–C–H and short O–C–O paths distinguish other oxygen sites.
PATTERNS = (
    ("hydroxyl", (8,1), ("OH",)),
    ("oxygen_oxygen", (8,8), ("OO",)),
    ("carbon_hydroxyl", (6,8,1), ("CO","OH")),
    ("ether", (6,8,6), ("CO","CO")),
    ("carbon_peroxy", (6,8,8), ("CO","OO")),
    ("hydroperoxy", (1,8,8), ("OH","OO")),
    ("organic_hydroperoxide", (6,8,8,1), ("CO","OO","OH")),
    ("carbonyl", (8,6), ("short_CO",)),
    ("formyl", (8,6,1), ("short_CO","CH")),
    ("carbon_substituted_formyl", (8,6,1,6), ("short_CO","CH","CC")),
    ("carbonyl_carbon", (8,6,6), ("short_CO","CC")),
    ("carbonyl_oxygen", (8,6,8), ("short_CO","CO")),
)
# These paths share the SAME hydrogen across donor and acceptor. Their weight
# peaks during transfer and vanishes when either contact exceeds its cutoff.
TRANSFER_PATTERNS = (
    ("carbon_H_oxygen", (6,1,8), ("CH","OH")),
    ("oxygen_H_oxygen", (8,1,8), ("OH","OH")),
    ("nitrogen_H_oxygen", (7,1,8), ("NH","OH")),
    ("carbon_H_peroxy", (6,1,8,8), ("CH","OH","OO")),
    ("oxygen_H_peroxy", (8,1,8,8), ("OH","OH","OO")),
)
# Distinguish folded conformers of the same functional groups. A hydrogen bond
# has a short covalent donor O–H edge, a distinct longer H···O contact, and a
# smooth preference for a linear donor–H···acceptor angle.
CONFORMER_PATTERNS = (
    ("hbond_OH_O", (8,1,8), ("OH","H_acceptor_O")),
    ("hbond_OH_peroxy", (8,1,8,8), ("OH","H_acceptor_O","OO")),
    ("hbond_OH_carbonyl", (8,1,8,6), ("OH","H_acceptor_O","short_CO")),
    ("torsion_CCOO", (6,6,8,8), ("CC","CO","OO")),
    ("torsion_COOH", (6,8,8,1), ("CO","OO","OH")),
    ("torsion_CCOH", (6,6,8,1), ("CC","CO","OH")),
    ("torsion_CCCO", (6,6,6,8), ("CC","CC","CO")),
)
# A carbon-bound aldehyde has H, C and short-bonded O on one carbon.
# It is a branched four-atom motif: O–C–H, with C attached to that same center.
CARBON_SUBSTITUTED = "carbon_substituted_formyl"


def _layout(name, atoms, bonds):
    """Define a connected tree and the angles whose cosines enter the features."""
    if name == CARBON_SUBSTITUTED:
        parents = (0,1,1)
        angles = [(0,1,2),(0,1,3),(2,1,3)]
    else:
        parents = range(len(bonds))
        angles = [(j,j+1,j+2) for j in range(len(atoms)-2)]
    edges = [(parent,j+1,kind) for j,(parent,kind) in enumerate(zip(parents,bonds,strict=True))]
    return atoms, edges, angles


class ChemicalEnvironment:
    """An additive energy head with separately serializable coefficients per state."""
    def __init__(self, weights=None, *, schema="smooth-oxygen-groups/1"):
        if schema not in ("smooth-oxygen-groups/1","smooth-oxygen-groups/2","smooth-oxygen-groups/3",
                          "smooth-organic-groups/1"):
            raise ValueError("Unknown functional-group basis")
        self.schema = schema
        self.organic = OrganicBasis() if schema=="smooth-organic-groups/1" else None
        self.long_contacts = schema.endswith("/3") or self.organic is not None
        self.patterns = PATTERNS + (TRANSFER_PATTERNS if schema!="smooth-oxygen-groups/1" else ())
        if self.long_contacts:
            self.patterns += CONFORMER_PATTERNS
        self.weights = {} if weights is None else {state_key(*s):np.asarray(w, float) for s,w in weights.items()}
        self.slices = {}
        self.size = 0
        for name,atoms,bonds in self.patterns:
            _,edges,angles = _layout(name,atoms,bonds)
            end = self.size + 1 + 2*(len(edges)+len(angles))
            if name.startswith("torsion_"):
                end = self.size + 2
            self.slices[name] = slice(self.size,end)
            self.size = end
        self.oxygen_size = self.size
        if self.organic is not None:
            for name,cols in self.organic.slices.items():
                self.slices[name] = slice(self.size+cols.start,self.size+cols.stop)
            self.size += self.organic.size
        if any(w.shape != (self.size,) or not np.isfinite(w).all() for w in self.weights.values()):
            raise ValueError("Invalid functional-group coefficients")

    def design(self, z, r, *, gradient=False):
        """Return extensive features and their negative Cartesian derivatives.

        Geometry enters through distances and cosines. Count both orientations
        of a symmetric motif at half weight, making its stretch and bend moments
        invariant to which identical atom appears first in the coordinate file.
        """
        values = np.zeros(self.size)
        jac = np.zeros((len(z),3,self.size)) if gradient else None
        if self.organic is not None:
            features, forces = self.organic.design(z,r,gradient=gradient)
            values[self.oxygen_size:] = features
            if gradient:
                jac[:,:,self.oxygen_size:] = forces.reshape(len(z),3,self.organic.size)
        if 8 not in z:
            return values, jac.reshape(r.size,-1) if gradient else None
        delta = r[:,None,:] - r[None,:,:]
        distance = np.linalg.norm(delta, axis=2)
        unit = delta / np.where(distance>0,distance,1.)[:,:,None]
        bonds = {}
        for kind,((a,b),length,on,off) in BONDS.items():
            if kind == "H_acceptor_O" and not self.long_contacts:
                continue
            if kind == "NH" and self.schema=="smooth-oxygen-groups/1":
                continue
            eligible = ((z[:,None]==a)&(z==b)) | ((z[:,None]==b)&(z==a))
            eligible &= ~np.eye(len(z),dtype=bool)
            t = np.clip((distance-on)/(off-on), 0., 1.)
            weight = eligible*(1-t)**4*(1+4*t+10*t**2+20*t**3)
            slope = eligible*(-140./(off-on))*t**3*(1-t)**3
            if kind == "short_CO":
                x = (distance-length)/.12
                envelope = np.exp(-.5*x*x)
                slope = envelope*(slope-weight*x/.12)
                weight *= envelope
            if kind == "H_acceptor_O":
                # The short-range rising switch excludes a covalently attached
                # acceptor. Both this and the outer switch are C3 continuous.
                t = np.clip((distance-1.40)/.35,0.,1.)
                rising = t**4*(35-84*t+70*t**2-20*t**3)
                slope = rising*slope+weight*(140/.35)*t**3*(1-t)**3
                weight *= rising
            bonds[kind] = weight,slope
        for name,atoms,labels in self.patterns:
            cols = self.slices[name]
            atoms,edges,angles = _layout(name,atoms,labels)
            paths = np.flatnonzero(z==atoms[0])[:,None]
            # Grow through supported bonds only. Every omitted path has exactly
            # zero weight, slope and curvature at and beyond its cutoff.
            for a,b,kind in edges:
                i,j = np.nonzero((bonds[kind][0][paths[:,a]]>0)&(z==atoms[b]))
                valid = ~(paths[i] == j[:,None]).any(axis=1)
                paths = np.column_stack((paths[i[valid]],j[valid]))
                if not len(paths):
                    break
            if not len(paths):
                continue
            symmetric = name != CARBON_SUBSTITUTED and atoms == atoms[::-1] and labels == labels[::-1]
            weights = np.array([bonds[k][0][paths[:,a],paths[:,b]] for a,b,k in edges])
            gate = weights.prod(axis=0) * (.5 if symmetric else 1.)
            bond_gate = gate
            moments = []
            gradients = []
            for a,b,kind in edges:
                i,j = paths[:,a], paths[:,b]
                moments.append((distance[i,j]-BONDS[kind][1])/.20)
                if gradient:
                    deriv = unit[i,j]/.20
                    gradients.append(((i,deriv),(j,-deriv)))
            for a,b,c in angles:
                i,j,k = paths[:,a],paths[:,b],paths[:,c]
                u,v = unit[i,j],unit[k,j]
                cosine = np.einsum("ij,ij->i",u,v)
                moments.append(cosine)
                if gradient:
                    du = (v-cosine[:,None]*u)/distance[i,j,None]
                    dv = (u-cosine[:,None]*v)/distance[k,j,None]
                    gradients.append(((i,du),(j,-du-dv),(k,dv)))
            m = np.array(moments).T
            phi = np.column_stack((np.ones(len(m)),m,.5*m*m))
            angular_weight = 1.
            if name.startswith("hbond_"):
                aidx=len(edges)
                cosine=m[:,aidx]
                angular_weight=.25*(1-cosine)**2
                gate=gate*angular_weight
            if name.startswith("torsion_"):
                # Dot the two outer bonds projected perpendicular to the middle
                # bond. This equals sin(a) sin(b) cos(dihedral), so the Fourier
                # response is permutation invariant under path reversal and
                # remains differentiable even when a bond angle straightens.
                i,j,k,l=paths.T
                u,v,w=unit[i,j],unit[k,j],unit[l,k]
                uv=np.einsum("ij,ij->i",u,v)
                vw=np.einsum("ij,ij->i",v,w)
                torsion=np.einsum("ij,ij->i",u,w)-uv*vw
                phi=np.column_stack((torsion,.5*torsion**2))
                if gradient:
                    def pull(g,e,d):
                        return (g-np.einsum("ij,ij->i",g,e)[:,None]*e)/d[:,None]
                    du=pull(w-vw[:,None]*v,u,distance[i,j])
                    dv=pull(-vw[:,None]*u-uv[:,None]*w,v,distance[k,j])
                    dw=pull(u-uv[:,None]*v,w,distance[l,k])
                    dtor=((i,du),(j,-du-dv),(k,dv-dw),(l,dw))
            values[cols] = gate@phi
            if gradient:
                block = jac[:,:,cols]
                for n,(a,b,kind) in enumerate(edges):
                    i,j = paths[:,a],paths[:,b]
                    # On included paths every factor is positive. Multiplying
                    # the OTHER factors explicitly also works at tiny weights.
                    others = np.prod(np.delete(weights,n,axis=0),axis=0)
                    dw = others*bonds[kind][1][i,j]*(.5 if symmetric else 1.)*angular_weight
                    df = (dw[:,None]*unit[i,j])[:,:,None]*phi[:,None,:]
                    np.add.at(block,i,-df)
                    np.add.at(block,j,df)
                if name.startswith("torsion_"):
                    for idx,deriv in dtor:
                        df=-gate[:,None]*deriv
                        np.add.at(block[:,:,0],idx,df)
                        np.add.at(block[:,:,1],idx,torsion[:,None]*df)
                    continue
                if name.startswith("hbond_"):
                    for idx,deriv in gradients[aidx]:
                        df=(.5*(1-cosine)*bond_gate)[:,None]*deriv
                        np.add.at(block,idx,df[:,:,None]*phi[:,None,:])
                for n,partials in enumerate(gradients):
                    for idx,deriv in partials:
                        df = -gate[:,None]*deriv
                        np.add.at(block[:,:,1+n],idx,df)
                        np.add.at(block[:,:,1+len(moments)+n],idx,m[:,n,None]*df)
        return values, jac.reshape(r.size,-1) if gradient else None

    def evaluate(self, z, r, state, *, forces):
        values, jac = self.design(z,r,gradient=forces)
        w = self.weights[state]
        return float(values@w), (jac@w).reshape(r.shape) if forces else None
