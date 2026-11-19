"""Differentiable functional groups for H, C, N, O, F, P, S and Cl.

A template is a small connected element-labelled graph. Each match contributes
a product of smooth functions of bond distances and local coordination. This
allows the same nucleus to participate in several environments during a reaction.
Closed rings include the last bond, and every vertex must map to a distinct atom.
The basis takes geometry in angstroms and returns negative Cartesian derivatives.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np

from .electronic import ElectronicFeatures


ELEMENT = dict(H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17)
RADII = ElectronicFeatures.radii

# These are radial filter centers in angstroms and Gaussian widths, not formal
# bond-order assignments. Single covalent contacts use the broader outer switch.
WINDOWS = {
    "double": {"C C": (1.34,.070), "C N": (1.29,.065), "C O": (1.22,.075),
               "N N": (1.25,.065), "C S": (1.64,.085), "O S": (1.46,.085),
               "O P": (1.49,.085), "N O": (1.20,.060)},
    "triple": {"C C": (1.20,.055), "C N": (1.16,.055)},
    "conjugated": {"C C": (1.40,.075), "C N": (1.36,.075), "C O": (1.36,.075),
                   "C S": (1.71,.085), "N N": (1.35,.075)},
    "resonance": {"C O": (1.26,.07), "N O": (1.24,.075), "C N": (1.34,.07),
                  "N N": (1.23,.09)},
}
WINDOWS = {(kind, *sorted(ELEMENT[x] for x in pair.split())): val
           for kind, entries in WINDOWS.items() for pair, val in entries.items()}


@dataclass(frozen=True)
class Motif:
    name: str
    atoms: tuple
    edges: tuple
    coordination: tuple
    symmetry: int
    angles: tuple

    @property
    def width(self):
        # Population, first and second bond-stretch moments, and corresponding
        # cosine moments over bonded triples. All sums respect graph symmetries.
        return 1 + 2*bool(self.edges) + 2*bool(self.angles)


def motif(name, atoms, kinds="", *, parents=None, close=None, coord=()):
    """Construct a path, branched tree or ring and count its labelled automorphisms.

    Each atom after the first is connected to an earlier parent. Extra closing
    bonds can be specified separately. A coordination selector is (site,
    neighbor element or '*', target count), using all explicit atoms including H.
    """
    z = tuple(ELEMENT[a] for a in atoms.split())
    kinds = kinds.split()
    parents = range(len(z)-1) if parents is None else parents
    edges = [(i,j,kind) for j,(i,kind) in enumerate(zip(parents,kinds,strict=True),1)]
    if close:
        edges.append(close)
    coord = tuple((i,0 if el=="*" else ELEMENT[el],target) for i,el,target in coord)
    neighbors = [[] for _ in z]
    def edge_key(a,b,k):
        return min(a,b),max(a,b),k
    original = {edge_key(*e) for e in edges}
    for i,j,kind in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
        if kind!="bond":
            WINDOWS[kind,*sorted((z[i],z[j]))]
    # Count only relabellings that preserve elements, coordination constraints
    # and all edge types. Unlabelled benzene has 12 mappings to the same ring.
    classes = {}
    for i,el in enumerate(z):
        label = el,tuple(sorted((a,n) for k,a,n in coord if k==i)),len(neighbors[i])
        classes.setdefault(label,[]).append(i)
    symmetry=0
    for groups in product(*(permutations(ids) for ids in classes.values())):
        perm={i:j for ids,assignment in zip(classes.values(),groups) for i,j in zip(ids,assignment)}
        symmetry += {edge_key(perm[i],perm[j],k) for i,j,k in edges}==original
    angles=tuple((a,j,b) for j,ns in enumerate(neighbors) for a,b in combinations(ns,2))
    return Motif(name,z,tuple(edges),coord,symmetry,angles)


def ring(name, atoms, kind="bond", *, substituent=None, coord=()):
    """Close an actual cycle; optionally attach a substituent to its first atom."""
    size=len(atoms.split())
    kinds=[kind]*(size-1)
    parents=list(range(size-1))
    if substituent:
        atoms+=" "+substituent
        kinds.append("bond")
        parents.append(0)
    return motif(name,atoms," ".join(kinds),parents=parents,close=(size-1,0,kind),coord=coord)


@lru_cache(maxsize=1)
def catalog():
    """Fixed, append-only ordering for the smooth-organic-groups/1 checkpoint."""
    B="bond"
    groups=[
        *(motif(name,"C",coord=((0,"*",4),(0,"C",n))) for n,name in enumerate(
            ("primary_alkyl","secondary_alkyl","tertiary_alkyl","quaternary_carbon"),1)),
        motif("alkene_like","C C","double",coord=((0,"*",3),(1,"*",3))),
        motif("alkyne_like","C C","triple",coord=((0,"*",2),(1,"*",2))),
        motif("terminal_alkyne","H C C","bond triple",coord=((1,"*",2),(2,"*",2))),
        motif("conjugated_diene","C C C C","double bond double"),
        motif("allene_like","C C C","double double",coord=((1,"*",2),)),
        motif("allylic_carbon","C C C","double bond",coord=((2,"*",4),)),
        ring("cyclopropyl_ring","C C C"),ring("cyclobutyl_ring","C C C C"),
        ring("benzene_like_ring","C C C C C C","conjugated"),
        ring("phenolic_site","C C C C C C","conjugated",substituent="O",coord=((6,"H",1),)),
        ring("aniline_like_site","C C C C C C","conjugated",substituent="N",coord=((6,"*",3),)),
        ring("benzylic_carbon","C C C C C C","conjugated",substituent="C",coord=((6,"*",4),)),
        ring("pyridine_like_ring","N C C C C C","conjugated"),
        ring("pyrrole_like_ring","N C C C C","conjugated"),
        ring("imidazole_like_ring","N C N C C","conjugated"),
        ring("furan_like_ring","O C C C C","conjugated"),
        ring("thiophene_like_ring","S C C C C","conjugated"),
        ring("epoxide_ring","O C C"),ring("aziridine_ring","N C C"),ring("thiirane_ring","S C C"),
        *(motif(name,"O C",B,coord=((0,"H",1),(1,"*",4),(1,"C",n))) for n,name in enumerate(
            ("primary_alcohol","secondary_alcohol","tertiary_alcohol"),1)),
        motif("enol_like","C C O H","double bond bond"),
        motif("ketone_like","O C C C","double bond bond",parents=(0,1,1),coord=((1,"*",3),)),
        motif("carboxylic_acid","O C O H","double bond bond",coord=((0,"*",1),(2,"*",2))),
        motif("carboxylate_like","O C O","resonance resonance",coord=((0,"*",1),(2,"*",1),(1,"*",3))),
        motif("carboxylic_ester","O C O C","double bond bond",coord=((0,"*",1),(2,"*",2))),
        motif("acid_anhydride","O C O C O","double bond bond double",coord=((2,"*",2),)),
        motif("carbonate","O C O O","double bond bond",parents=(0,1,1),coord=((1,"*",3),(2,"*",2),(3,"*",2))),
        motif("acetal_like","O C O","bond bond",coord=((1,"*",4),(0,"C",2),(2,"C",2))),
        motif("hemiacetal_like","O C O","bond bond",coord=((1,"*",4),(0,"H",1),(2,"C",2))),
        *(motif(name,"N",coord=((0,"*",3),(0,"C",n),(0,"H",3-n))) for n,name in enumerate(
            ("primary_amine_like","secondary_amine_like","tertiary_amine_like"),1)),
        motif("four_coordinate_nitrogen","N",coord=((0,"*",4),)),
        motif("imine_like","C N","double",coord=((0,"*",3),)),
        motif("enamine_like","C C N","double bond",coord=((2,"*",3),)),
        motif("nitrile","C N","triple",coord=((0,"*",2),(1,"*",1))),
        motif("isonitrile","C N C","triple bond",coord=((0,"*",1),(1,"*",2))),
        motif("amide_link","O C N","double bond",coord=((1,"*",3),(2,"*",3))),
        motif("imide","O C N C O","double bond bond double"),
        motif("urea_like","O C N N","double bond bond",parents=(0,1,1)),
        motif("carbamate","N C O O C","bond double bond bond",parents=(0,1,1,3),coord=((1,"*",3),)),
        motif("amidine_like","N C N","double bond",coord=((1,"*",3),)),
        motif("guanidine_like","N C N N","resonance resonance resonance",parents=(0,1,1),coord=((1,"*",3),)),
        motif("isocyanate","N C O","double double",coord=((1,"*",2),)),
        motif("nitro_like","O N O","resonance resonance",coord=((0,"*",1),(1,"*",3),(2,"*",1))),
        motif("nitroso","C N O","bond double",coord=((1,"*",2),)),
        motif("azo_link","C N N C","bond double bond"),
        motif("azide_like","N N N","resonance resonance",coord=((1,"*",2),)),
        motif("hydrazine_link","N N",B,coord=((0,"*",3),(1,"*",3))),
        motif("oxime","C N O H","double bond bond"),
        motif("hydroxylamine_link","N O","bond",coord=((0,"*",3),(1,"*",2))),
        motif("thiol","C S H","bond bond",coord=((1,"*",2),)),
        motif("thioether","C S C","bond bond",coord=((1,"*",2),)),
        motif("disulfide","C S S C","bond bond bond"),
        motif("thiocarbonyl","C S","double",coord=((0,"*",3),)),
        motif("thioester","O C S C","double bond bond"),
        motif("sulfoxide","O S C C","double bond bond",parents=(0,1,1),coord=((1,"*",3),)),
        motif("sulfone","O S O C C","double double bond bond",parents=(0,1,1,1),coord=((1,"*",4),)),
        motif("sulfonamide","O S O N","double double bond",parents=(0,1,1)),
        motif("sulfonic_acid","O S O O H","double double bond bond",parents=(0,1,1,3)),
        motif("organophosphine","C P",B,coord=((1,"*",3),)),
        motif("phosphoryl_like","P O","double",coord=((1,"*",1),)),
        motif("phosphate_ester_site","P O C","bond bond",coord=((0,"O",4),(1,"*",2))),
        motif("phosphonate_site","C P O","bond double",coord=((1,"O",3),(2,"*",1))),
        motif("phosphine_oxide","O P","double",coord=((0,"*",1),(1,"C",3))),
    ]
    for element in ("F","Cl"):
        groups.extend([motif("alkyl_"+element,"C "+element,B,coord=((0,"*",4),)),
                       motif("vinyl_"+element,"C C "+element,"double bond",coord=((1,"*",3),)),
                       ring("aryl_"+element,"C C C C C C","conjugated",substituent=element),
                       motif("acyl_"+element,"O C "+element,"double bond")])
    return tuple(groups)


def switch(t):
    """Cubic-flat ends give continuous forces and Hessians as neighbors leave."""
    t=np.clip(t,0.,1.)
    return (1-t)**4*(1+4*t+10*t*t+20*t**3), -140*t**3*(1-t)**3


class OrganicBasis:
    """Match smooth connected subgraphs and analytically differentiate each sum."""

    def __init__(self):
        self.slices={}
        self.size=0
        self.groups=catalog()
        self.required=[Counter(g.atoms) for g in self.groups]
        for g in self.groups:
            self.slices[g.name]=slice(self.size,self.size+g.width)
            self.size+=g.width

    def design(self,z,r, *, gradient=False):
        n=len(z)
        value=np.zeros(self.size)
        jac=np.zeros((n,3,self.size)) if gradient else None
        delta=r[:,None]-r[None,:]
        distance=np.linalg.norm(delta,axis=-1)
        unit=delta/np.where(distance>0,distance,1.)[:,:,None] if gradient else None
        radius=np.array([RADII[int(a)] for a in z])
        single=radius[:,None]+radius[None,:]
        width=.34*single
        base,slope=switch((distance-1.08*single)/width)
        np.fill_diagonal(base,0.)
        slope/=width
        np.fill_diagonal(slope,0.)
        channels={"bond":(base,slope,single)}
        counts=Counter(z)
        for (kind,a,b),(center,spread) in WINDOWS.items():
            if a not in counts or b not in counts:
                continue
            if kind not in channels:
                channels[kind]=tuple(np.zeros_like(base) for _ in range(3))
            w,dw,ref=channels[kind]
            i,j=np.nonzero((z[:,None]==a)&(z==b)&(base>0))
            x=(distance[i,j]-center)/spread
            envelope=np.exp(-.5*x*x)
            w[i,j]=w[j,i]=base[i,j]*envelope
            dw[i,j]=dw[j,i]=envelope*(slope[i,j]-base[i,j]*x/spread)
            ref[i,j]=ref[j,i]=center

        selectors={}
        starts={}
        prefixes={}
        # Cartesian force contributions of any pair are equal and opposite.
        def scatter(block,i,j,radial):
            forces=-unit[i,j,:,None]*radial[:,None,:]
            np.add.at(block,i,forces)
            np.add.at(block,j,-forces)

        for group,required in zip(self.groups,self.required):
            if any(counts[a]<k for a,k in required.items()):
                continue
            atoms=group.atoms
            if atoms[0] not in starts:
                starts[atoms[0]]=np.flatnonzero(z==atoms[0])[:,None]
            paths=starts[atoms[0]]
            prefix=(atoms[0],)
            for a,b,kind in group.edges:
                prefix=(*prefix,(a,b,kind,atoms[b]))
                if prefix not in prefixes:
                    edge=channels[kind][0]
                    if b==paths.shape[1]:
                        m,j=np.nonzero((edge[paths[:,a]]>0)&(z==atoms[b]))
                        valid=~(paths[m]==j[:,None]).any(axis=1)
                        prefixes[prefix]=np.column_stack((paths[m[valid]],j[valid]))
                    else:
                        prefixes[prefix]=paths[edge[paths[:,a],paths[:,b]]>0]
                paths=prefixes[prefix]
                if not len(paths):
                    break
            if not len(paths):
                continue
            factors=[]
            for a,b,kind in group.edges:
                factors.append(channels[kind][0][paths[:,a],paths[:,b]])
            for a,el,num in group.coordination:
                if (el,num) not in selectors:
                    total=base.sum(axis=1) if not el else base[:,z==el].sum(axis=1)
                    x=(total-num)/.95
                    inside=np.maximum(0.,1-x*x)
                    selectors[el,num]=(inside**4,-8*x*inside**3/.95)
                factors.append(selectors[el,num][0][paths[:,a]])
            weights=np.array(factors)
            valid=(weights>0).all(axis=0)
            if not valid.any():
                continue
            weights=weights[:,valid]
            paths=paths[valid]
            gate=weights.prod(axis=0)/group.symmetry
            cols=self.slices[group.name]
            phi=np.ones((len(paths),group.width))
            lengths=[]; angles=[]
            for a,b,kind in group.edges:
                i,j=paths[:,a],paths[:,b]
                lengths.append((distance[i,j]-channels[kind][2][i,j])/.2)
            if lengths:
                stretches=np.array(lengths)
                phi[:,1]=stretches.mean(axis=0)
                phi[:,2]=.5*(stretches**2).mean(axis=0)
            for a,b,c in group.angles:
                i,j,k=paths[:,a],paths[:,b],paths[:,c]
                u=delta[i,j]/distance[i,j,None]
                v=delta[k,j]/distance[k,j,None]
                angles.append(np.einsum("ij,ij->i",u,v))
            if angles:
                bends=np.array(angles)
                phi[:,3]=bends.mean(axis=0)
                phi[:,4]=.5*(bends**2).mean(axis=0)
            value[cols]=gate@phi
            if not gradient:
                continue
            block=jac[:,:,cols]
            # All factors are strictly positive here. Prefix/suffix products
            # avoid dividing by a tiny gate in the tail of a radial window.
            left=np.ones(len(paths))/group.symmetry
            right=np.cumprod(weights[::-1],axis=0)[::-1]
            partial=[]
            for t,f in enumerate(weights):
                partial.append(left*right[t+1] if t+1<len(weights) else left)
                left=left*f
            for t,(a,b,kind) in enumerate(group.edges):
                i,j=paths[:,a],paths[:,b]
                radial=(partial[t]*channels[kind][1][i,j])[:,None]*phi
                stretch=gate/(len(lengths)*.2)
                radial[:,1]+=stretch
                radial[:,2]+=stretch*lengths[t]
                scatter(block,i,j,radial)
            for t,(a,el,num) in enumerate(group.coordination,len(lengths)):
                dg=partial[t]*selectors[el,num][1][paths[:,a]]
                if np.any(dg):
                    at_site=np.zeros((n,group.width))
                    np.add.at(at_site,paths[:,a],dg[:,None]*phi)
                    i,j=np.nonzero(np.triu(slope!=0,1))
                    if el:
                        radial=at_site[i]*(z[j]==el)[:,None]+at_site[j]*(z[i]==el)[:,None]
                    else:
                        radial=at_site[i]+at_site[j]
                    scatter(block,i,j,slope[i,j,None]*radial)
            for (a,b,c),cosine in zip(group.angles,angles):
                i,j,k=paths[:,a],paths[:,b],paths[:,c]
                u,v=unit[i,j],unit[k,j]
                scale=-gate[:,None]/len(angles)
                du=scale*(v-cosine[:,None]*u)/distance[i,j,None]
                dv=scale*(u-cosine[:,None]*v)/distance[k,j,None]
                for nodes,f in ((i,du),(j,-du-dv),(k,dv)):
                    np.add.at(block[:,:,3],nodes,f)
                    np.add.at(block[:,:,4],nodes,cosine[:,None]*f)
        return value,jac.reshape(r.size,self.size) if gradient else None
