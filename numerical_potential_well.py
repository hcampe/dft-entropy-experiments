import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to calculate Shannon entropy
def shannon_entropy(rho):
    # Exclude points where rho is zero
    non_zero_rho = rho[rho > 0]
    # Calculate probabilities
    probabilities = non_zero_rho / np.sum(non_zero_rho)
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

# Define the number of points in the domain
n_points = 100

# Define the domain using a linear space between -π/2 and π/2, including the right endpoint
domain = np.linspace(-np.pi / 2, np.pi / 2, n_points, endpoint=True)

# Initial guess for rho (must satisfy constraints)
initial_rho = np.ones(n_points)
initial_rho[0] = 0
initial_rho[-1] = 0
initial_rho = initial_rho / np.sum(initial_rho)  # Normalize to sum to one

# Define the maximum value of the true derivative
D_max = 1

# Calculate the spacing between points in the domain
delta_x = np.pi / (n_points - 1)

# Define a small threshold ε for continuity based on the maximum derivative
epsilon = D_max * delta_x

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda rho: np.sum(rho) * delta_x - 1},  # Normalization constraint
    {'type': 'ineq', 'fun': lambda rho: rho},  # Non-negativity constraint
    {'type': 'eq', 'fun': lambda rho: rho[0]},  # First value is zero
    {'type': 'eq', 'fun': lambda rho: rho[-1]},  # Last value is zero
]

# Add continuity constraints: |ρ_i+1 - ρ_i| <= ε
for i in range(n_points - 1):
    constraints.append({'type': 'ineq', 'fun': lambda rho, i=i: epsilon - np.abs(rho[i+1] - rho[i])})

# Perform the optimization
result = minimize(lambda rho: -shannon_entropy(rho), initial_rho, constraints=constraints, method='SLSQP')

# Optimized rho
optimized_rho = result.x

# Calculate the Shannon entropy for the optimized rho
optimized_entropy = shannon_entropy(optimized_rho)

# Display the results
print("Optimized Function values (ρ):", optimized_rho)
print("Optimized Shannon entropy:", optimized_entropy)

# Plot the result
plt.plot(domain, optimized_rho, label='Optimized ρ')
plt.xlabel('Domain')
plt.ylabel('Function values (ρ)')
plt.title('Optimized Function values (ρ)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the numerical derivative of ρ
numerical_derivative = np.diff(optimized_rho) / delta_x

# Define the domain for the derivative plot (midpoints of the original domain intervals)
derivative_domain = (domain[:-1] + domain[1:]) / 2

# Plot the numerical derivative
plt.plot(derivative_domain, numerical_derivative, label='Numerical Derivative of ρ')
plt.xlabel('Domain')
plt.ylabel('Derivative of ρ')
plt.title('Numerical Derivative of Optimized Function values (ρ)')
plt.legend()
plt.grid(True)
plt.show()

# Define the reference function 0.5 * cos(x)
reference_function = 2/np.pi * np.cos(domain)**2

# Ensure the reference function meets the constraints
reference_function[0] = 0
reference_function[-1] = 0

# Normalize the reference function to sum to one
reference_function = reference_function / np.sum(reference_function)

# Calculate the Shannon entropy for the reference function
reference_entropy = shannon_entropy(reference_function)

# Display the reference function and its Shannon entropy
print("Reference Function values (0.5 * cos(x)):", reference_function)
print("Reference Shannon entropy:", reference_entropy)

from scipy.integrate import simps

# Function to calculate kinetic energy
def kinetic_energy(psi):
    # Numerical derivative of ψ
    numerical_derivative = np.diff(psi) / delta_x
    # Calculate kinetic energy as the integral of the square of the derivative
    kinetic_energy_value = simps(numerical_derivative**2, derivative_domain)
    return kinetic_energy_value

# Initial guess for ψ (must satisfy constraints)
initial_psi_kinetic = np.ones(n_points)
initial_psi_kinetic[0] = 0
initial_psi_kinetic[-1] = 0
initial_psi_kinetic = initial_psi_kinetic / np.sum(initial_psi_kinetic)  # Normalize to sum to one

# Constraints for kinetic energy optimization (same except positivity constraint)
constraints_kinetic = [
    {'type': 'eq', 'fun': lambda psi: np.sum(psi) * delta_x - 1},  # Normalization constraint
    {'type': 'eq', 'fun': lambda psi: psi[0]},  # First value is zero
    {'type': 'eq', 'fun': lambda psi: psi[-1]}  # Last value is zero
]

# Add continuity constraints: |ψ_i+1 - ψ_i| <= ε
for i in range(n_points - 1):
    constraints_kinetic.append({'type': 'ineq', 'fun': lambda psi, i=i: epsilon - np.abs(psi[i+1] - psi[i])})

# Perform the optimization for kinetic energy
result_kinetic = minimize(kinetic_energy, initial_psi_kinetic, constraints=constraints_kinetic, method='SLSQP')

# Optimized ψ for kinetic energy
optimized_psi_kinetic = result_kinetic.x

# Plot the square of the resulting function
plt.plot(domain, optimized_psi_kinetic**2, label='Square of Optimized ψ (Kinetic Energy)')
plt.plot(domain, reference_function/np.max(reference_function)*np.max(optimized_psi_kinetic**2), label='~cos(x)^2')
plt.xlabel('Domain')
plt.ylabel('Squared Function values (ψ^2)')
plt.title('Square of Optimized Function values (ψ) - Kinetic Energy')
plt.legend()
plt.grid(True)
plt.show()
