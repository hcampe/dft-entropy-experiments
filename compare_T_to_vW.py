import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft
from pyscf.dft import numint
import pandas as pd

# Define the ethanol molecule
mol = gto.M(
    atom='''C -0.748 0.501 0.000
            C 0.748 0.501 0.000
            O 1.241 -0.873 0.000
            H -1.206 0.927 0.881
            H -1.206 0.927 -0.881
            H -1.281 -0.543 0.000
            H 1.206 0.927 0.881
            H 1.206 0.927 -0.881
            H 2.148 -0.893 0.000''',
    basis='cc-pvdz',
    charge=0,
    spin=0,
    verbose=3,
)

# Perform SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Get electron density on a grid
grids = dft.gen_grid.Grids(mol)
grids.level = 3  # Increase the level for more accuracy
grids.build()
coords = grids.coords
weights = grids.weights

# Calculate the electron density
ao_value = numint.eval_ao(mol, coords)
rho = numint.eval_rho(mol, ao_value, mf.make_rdm1())

# Calculate the kinetic energy density (von Weizsäcker approximation)
def kinetic_energy_density(rho, coords):
    grad_rho_x = np.gradient(rho, coords[:, 0], axis=0)
    grad_rho_y = np.gradient(rho, coords[:, 1], axis=1)
    grad_rho_z = np.gradient(rho, coords[:, 2], axis=2)
    grad_rho_magnitude_squared = grad_rho_x**2 + grad_rho_y**2 + grad_rho_z**2
    tau_w = 0.5 * grad_rho_magnitude_squared / rho
    return tau_w

tau_w = kinetic_energy_density(rho, coords)

# Compute von Weizsäcker kinetic energy
T_W = (1/8) * np.sum(weights * tau_w)

# Compute the SCF kinetic energy
T_SCF = mf.kinetic_energy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(coords[:, 0], tau_w, label='Von Weizsäcker Kinetic Energy Density')
plt.axhline(y=T_SCF/mol.nelectron, color='r', linestyle='--', label='SCF Kinetic Energy per Electron')
plt.xlabel('Coordinate (a.u.)')
plt.ylabel('Kinetic Energy Density (a.u.)')
plt.legend()
plt.title('Kinetic Energy Density Comparison')
plt.show()

# Display kinetic energy values in a DataFrame
kinetic_energy_df = pd.DataFrame({
    'Kinetic Energy Method': ['Von Weizsäcker', 'SCF'],
    'Kinetic Energy (a.u.)': [T_W, T_SCF]
})

print(kinetic_energy_df)
