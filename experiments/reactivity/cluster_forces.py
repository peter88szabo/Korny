"""Project published PBE0 forces in flexible water clusters onto internal motions.

Use neutral clusters containing at least three water molecules. No isolated
water geometry, frequency, or barrier is read. Hold out the entire pentamer
size. Subtract the published dispersion force from the published total at
each geometry so the fitted surface remains bare PBE0.
"""
from collections import Counter
from pathlib import Path
import h5py
import numpy as np


def projected_rows(base,head,path):
    """Return training/development force rows and the source IDs needed to recreate the split."""
    rng = np.random.default_rng(915)
    counts = Counter()
    blocks, names, angles = {'train':[], 'dev':[]}, {'train':[], 'dev':[]}, {'train':[], 'dev':[]}
    with h5py.File(path) as h:
        for key in rng.permutation(sorted(k for k in h if k!='metadata')):
            g=h[key]
            n=g['atomic_numbers'].shape[0]//3
            if n not in (3,4,5,6,7) or counts[n]>=35 or g['source'][()]!=b'water_cluster':
                continue
            z=g['atomic_numbers'][:]
            if np.sum(z==8)!=n or np.sum(z==1)!=2*n or g['charge'][()]!=0:
                continue
            r=g['positions'][:]
            # Each hydrogen is assigned to its nearest oxygen. Skip unusual
            # proton-shared topologies for this small-angle calibration.
            oxy=np.flatnonzero(z==8)
            hyd=np.flatnonzero(z==1)
            distances=np.linalg.norm(r[hyd,None]-r[oxy],axis=2)
            owners=distances.argmin(axis=1)
            if np.any(distances.min(axis=1)>1.22) or not np.all(np.bincount(owners,minlength=n)==2):
                continue
            pref='dev' if n==5 else 'train'
            grad=head.design(z,r)[1]
            error=base.predict(z,r)[1]-(g['total_forces'][:]-g['vdw_forces'][:])
            for site,i in enumerate(oxy):
                j,k=hyd[owners==site]
                u,v=r[[j,k]]-r[i]
                u,v=u/np.linalg.norm(u),v/np.linalg.norm(v)
                cosine=np.clip(u@v,-1,1)
                # Pure angular tangent at fixed O-H distances plus independent
                # bond-stretch directions. A rigid shift of the three atoms
                # removes response to a net translational force on the fragment.
                angles[pref].append(float(np.degrees(np.arccos(cosine))))
                for dj,dk in ((v-cosine*u,u-cosine*v),(u,np.zeros(3)),(np.zeros(3),v)):
                    direction=np.stack([np.zeros(3),dj,dk])
                    direction-=direction.mean(axis=0)
                    direction/=np.linalg.norm(direction)
                    idx=np.array([i,j,k])
                    row=direction.ravel() @ grad.reshape(len(z),3,-1)[idx].reshape(9,-1)
                    target=float(np.sum(direction*error[idx]))
                    blocks[pref].append(np.r_[row,target])
            names[pref].append(str(key))
            counts[n]+=1
            if len(counts)==5 and min(counts.values())==35:
                break
    if any(counts[n]!=35 for n in (3,4,5,6,7)):
        raise ValueError(f'Incomplete cluster split: {counts}')
    return {k:np.array(v) for k,v in blocks.items()},names,{
        k:np.percentile(v,[5,50,95]).tolist() for k,v in angles.items()}
