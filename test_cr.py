
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, dft
from pyscf.dft import numint
from rdkit import Chem
from rdkit.Chem import AllChem

def optimize_geometry(formula):
    # Generate a molecule from a chemical formula using RDKit
    mol = Chem.AddHs(Chem.MolFromSmiles(formula))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def mol_to_xyz(mol):
    atoms = mol.GetAtoms()
    positions = mol.GetConformer().GetPositions()
    xyz = []
    for atom, pos in zip(atoms, positions):
        xyz.append(f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}")
    return "\n".join(xyz)

def save_dft_data(envs):
    dft_energies.append(envs['e_tot'])
    dft_densities.append(envs['dm'])

# Input chemical formula (SMILES format)
# chemical_formula = 'O'  # For water,
chemical_formula = 'OCC'  # For ethanol,

# Optimize geometry
rdkit_mol = optimize_geometry(chemical_formula)
xyz_geometry = mol_to_xyz(rdkit_mol)

# Define the molecular geometry for PySCF
mol = gto.Mole()
mol.atom = xyz_geometry
# mol.basis = 'sto-3g'
mol.basis = '6-31G(2df,p)'
mol.charge = 0
mol.spin = 0
mol.build()

# Create lists to store the energies and Shannon entropy at each step
dft_energies = []
dft_densities = []

# Perform a closed-shell DFT calculation using B3LYP functional with a callback
mf_dft = dft.RKS(mol)
mf_dft.xc = 'PBE'
#mf_dft.damp = .9
#mf_dft.diis = None
mf_dft.callback = save_dft_data
#aenergy_dft = mf_dft.kernel()
mf_dft.run(
    init_guess='atom',
    max_cycle=50,
    conv_tol=1e-9,
    diis_start_cycle=0,
    diis_space=2,
)

def compute_observable(mol, dm, operator, grid):
    """
    Compute the expected value of an observable over space using the electron density,
    with a specified integration grid.
    
    Parameters:
    - mol: A PySCF molecule object.
    - dm: The density matrix.
    - operator: A function that takes grid coordinates as input and returns the observable's value at those coordinates.
    - grid: A PySCF grid object for integration.
    
    Returns:
    - The expected value of the observable.
    """
    # No need to set up the grid here, it's passed as an argument
    
    # Evaluate the electron density on the grid
    ao_values = dft.numint.eval_ao(mol, grid.coords)
    density = dft.numint.eval_rho(mol, ao_values, dm)
    
    # Evaluate the observable's operator on the grid
    observable_values = operator(grid.coords)
    
    # Compute the expected value by integrating over the grid
    expected_value = np.dot(density, observable_values * grid.weights)
    
    return expected_value

# grid
grid = dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()


# calculates covariance matrix of the position at each step
covariance_matrices = []
for dm in dft_densities:
    cov = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            r_i  = lambda coords: coords[:, i]
            r_j  = lambda coords: coords[:, j]
            r_ij = lambda coords: coords[:, i] * coords[:, j]
            cov[i, j] = compute_observable(mol, dm, r_ij, grid) - compute_observable(mol, dm, r_i, grid) * compute_observable(mol, dm, r_j, grid)
            cov[j, i] = cov[i, j]
    covariance_matrices.append(cov)

# a number of electrons in ethanol
N = 26


# Diagon rise Cobain's matrices and calculate lower bounds to the energy
lower_bounds = []
for cov in covariance_matrices:
    eigenvalues = np.linalg.eigvalsh(cov)
    lower_bound = N/8 * np.sum(1/eigenvalues)
    lower_bounds.append(lower_bound)

# print the total energy and the lore bounds
print('Total energy:', dft_energies)
print('Lower bounds:', lower_bounds)

# Plot the energies
plt.figure(figsize=(10, 6))
plt.plot(dft_energies, label='DFT Energies')
plt.xlabel('SCF Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('SCF Convergence')
plt.legend()
plt.grid(True)
plt.show()