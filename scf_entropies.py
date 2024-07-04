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

def calculate_shannon_entropy(mol, dm):
    grid = dft.gen_grid.Grids(mol)
    grid.level = 3
    grid.build()
    coords = grid.coords
    weights = grid.weights
    ao_value = numint.eval_ao(mol, coords, deriv=0)
    rho = numint.eval_rho(mol, ao_value, dm)
    rho /= mol.nelectron
    rho = np.maximum(rho, 1e-10)  # Avoid log(0)
    entropy = -np.sum(rho * np.log(rho) * weights)
    return entropy

def calculate_kl_divergence(mol1, dm1, mol2, dm2):
    grid = dft.gen_grid.Grids(mol1)
    grid.level = 3
    grid.build()
    coords = grid.coords
    weights = grid.weights
    ao_value1 = numint.eval_ao(mol1, coords, deriv=0)
    rho1 = numint.eval_rho(mol1, ao_value1, dm1)
    rho1 /= mol1.nelectron
    rho1 = np.maximum(rho1, 1e-10)  # Avoid log(0)
    ao_value2 = numint.eval_ao(mol2, coords, deriv=0)
    rho2 = numint.eval_rho(mol2, ao_value2, dm2)
    rho2 /= mol2.nelectron
    rho2 = np.maximum(rho2, 1e-10)  # Avoid log(0)
    kl_divergence = np.sum(rho1 * np.log(rho1 / rho2) * weights)
    return kl_divergence

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
    diis_space=8,
)
# Calculate the Shannon entropy at each step
dft_entropies = []
for dm in dft_densities:
    entropy = calculate_shannon_entropy(mol, dm)
    dft_entropies.append(entropy)

# culate kl divergencies between current and final densities
dft_kl_divergences = []
for dm in dft_densities:
    kl_divergence = calculate_kl_divergence(mol, dft_densities[-1], mol, dm)
    dft_kl_divergences.append(kl_divergence)

# Plot the energies
plt.figure(figsize=(10, 6))
plt.plot(dft_energies, label='DFT Energies')
plt.xlabel('SCF Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('SCF Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Plot the Shannon entropy
plt.figure(figsize=(10, 6))
plt.plot(dft_entropies, label='DFT Shannon Entropy')
plt.xlabel('SCF Iteration')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Plot the KL divergence
plt.figure(figsize=(10, 6))
plt.plot(dft_kl_divergences, label='DFT KL Divergence')
plt.xlabel('SCF Iteration')
plt.ylabel('KL Divergence')
plt.title('KL Divergence Convergence')
plt.legend()
plt.grid(True)
plt.show()