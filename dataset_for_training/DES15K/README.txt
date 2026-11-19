====================

   DFT DATASETS

====================

For each molecular conformation in the .hdf5 datasets, you will find the following properties:

-'atNUM': Atomic numbers (N)
-'atXYZ': Atoms coordinates [Ang] (Nx3)

-'ePBE0+MBD': Total PBE0+MBD energy [eV] (1)
-'ePBE0': PBE0 energy [eV] (1)
-'DIP': Total dipole moment [e.Ang] (1)
-'vDIP': Dipole moment components [e.Ang] (3)

-'totFOR': Total PBE0+MBD atomic forces (unitary forces cleaned) [eV/Ang] (Nx3)
-'pbe0FOR': PBE0 atomic forces [eV/Ang] (Nx3)
-'hRAT': Hirshfeld ratios (N)
-'hCHG': Hirshfeld charges [e] (N)
-'hDIP': Hirshfeld dipole moments [e.bohr] (N)
-'hVDIP': Components of Hirshfeld dipole moments [e.bohr] (Nx3)

====================

For a minimal example of how to read the hdf5 data, please see hdf5_to_extxyz.py. 

You can run the script with: 
> python hdf5_to_extxyz.py <hdf5_filename>
