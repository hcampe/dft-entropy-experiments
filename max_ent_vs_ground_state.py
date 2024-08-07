import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import eigh

# Function to construct the Laplacian matrix
def laplacian(x_values):
    N = len(x_values)
    dx = x_values[1] - x_values[0]  # Assuming uniform spacing
    
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    
    laplacian_matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    laplacian_matrix /= dx**2
    
    return laplacian_matrix

# Function to compute numerical solution and ground state density for a given potential
def ground_state_density(num_points, potential_func, x_min=-3, x_max=3):
    x_values = np.linspace(x_min, x_max, num_points)
    potential = potential_func(x_values)
    kinetic = -0.5 * laplacian(x_values)
    hamiltonian = kinetic + np.diag(potential)
    energies, eigenstates = eigh(hamiltonian)
    density = eigenstates[:, 0]**2
    normalized_density = density / np.sum(density * (x_values[1] - x_values[0]))
    return energies[0], x_values, normalized_density, potential

# Function to compute the maximum entropy density based on the given formula and constants
def max_entropy_density(x_values, n, E):
    s = (2 * n * E / (n + 1))**(1 / (2 * n))
    C = (2 * s / (2 * n)) * sp.gamma(1 / (2 * n))
    density = (1 / C) * np.exp(- (x_values**(2 * n)) / (s**(2 * n)))
    return density

# Potential functions and labels for each n
n_values = [1, 2, 4, 10]
potential_funcs = {n: (lambda x, n=n: x**(2*n)) for n in n_values}
potential_labels = {n: f'$V(x) = x^{2*n}$' for n in n_values}

# Compute and plot the ground state probability densities and updated entropy-maximized densities for each potential
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, n in enumerate(n_values):
    ax = axs[i // 2, i % 2]
    ground_state_energy, x_values, normalized_density, potential = ground_state_density(500, potential_funcs[n])
    max_entropy_density_updated = max_entropy_density(x_values, n, ground_state_energy)
    
    ax.plot(x_values, normalized_density, label='Ground State Density')
    ax.plot(x_values, potential, label=potential_labels[n], color='black')
    ax.plot(x_values, max_entropy_density_updated, label='Max Entropy Density', color='red')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('Probability Density')
    ax.set_ylim(-0.1, 1)  # Adjusted y-axis limit to better fit the potential and density
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
