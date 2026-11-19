import h5py
import numpy as np
from tqdm import tqdm
from ase.io.extxyz import write_extxyz
from ase import Atoms
import sys

# Atomic energies
EPBE0_atom = {
    1: -13.6414041617373,  # H
    6: -1027.60791501338,  # C
    7: -1484.27481908749,  # N
    8: -2039.75030551381,  # O
    9: -2710.54734320680,  # F
    15: -9283.01120605055, # P
    16: -10828.7228945312, # S
    17: -12516.4600457922, # Cl
}

# Get name of the hdf5 file
name = sys.argv[1] 
# name = 'DES15k.hdf5'
# name = 'SPICE_dipeptides.hdf5'
# name = 'TorsionNet500.hdf5'

molecules = []
with h5py.File(name, 'r') as f:
    print(len(f.keys()))
    id_mol = list(f.keys())
    
    for molid in tqdm(id_mol):
        conf = list(f[molid].keys())
        collected_ids = np.array(conf).tolist()

        for ID in collected_ids:
            # Get molecular data
            R = np.asarray(f[molid][ID]['atXYZ'])
            z = np.array(f[molid][ID]['atNUM'])
            mols = Atoms(z, R)
            
            # Get energies
            sum_atomic_energies = sum(EPBE0_atom[atom.number] for atom in mols)
            E = float(f[molid][ID]['ePBE0+MBD'][0]) - sum_atomic_energies 

            # Get charges
            hCHG = np.array(f[molid][ID]['hCHG'])
            charge = round(np.sum(hCHG))

            # Get forces, hirshfeld ratios, and dipoles
            F = np.array(f[molid][ID]['totFOR'])
            hRAT = np.array(f[molid][ID]['hRAT'])
            vDIP = np.array(f[molid][ID]['vDIP'])
            
            # Store properties
            mols.info['energy'] = E
            mols.info['dipole'] = vDIP
            mols.info['charge'] = charge
            mols.arrays['forces'] = F
            mols.arrays['hirsh_ratios'] = hRAT
            
            molecules.append(mols)

output_name = name.split('.')[0] + '.extxyz'
write_extxyz(output_name, molecules)
