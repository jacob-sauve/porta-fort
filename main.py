# Jacob Sauvé, McGill University
# 2025/04/06


import numpy as np
import matplotlib.pyplot as plt

# Constants
n_particles = 3000
dt = 1e-3
radius = 0.005
mass = 1.0
snow_strength = 1.68e6  # Pa (compressive strength of snow)
domain_size = 1.0
neighbor_radius = 3 * radius
damping = 0.98

# Particle data
x = np.zeros((n_particles, 2))  # Particle positions
v = np.zeros((n_particles, 2))  # Particle velocities
f = np.zeros((n_particles, 2))  # Forces on particles
compression_force = np.zeros((n_particles, 2))  # Compression forces
accumulated_force = np.zeros((n_particles, 2))  # Accumulated forces

# Airbag parameters
airbag_center = np.array([0.5, 0.5])  # Airbag center location
airbag_pressure = 1.0e5  # Initial airbag pressure (in Pascals)

# Helper function for calculating distance
def distance(i, j):
    return np.linalg.norm(x[i] - x[j])

# Initialization of particles (a simple grid arrangement)
def initialize():
    grid_x = int(np.sqrt(n_particles))
    for i in range(n_particles):
        x[i] = [(i % grid_x) * radius * 2 + 0.05, (i // grid_x) * radius * 2 + 0.05]
        v[i] = [0.0, 0.0]
        f[i] = [0.0, 0.0]

# External forces (gravity, air pressure)
def calculate_external_forces():
    global f
    # Gravity force (downward)
    gravity = np.array([0.0, -9.81]) * mass
    f += gravity  # Apply gravity to all particles

    # Apply airbag force
    for i in range(n_particles):
        dx = x[i] - airbag_center
        r = np.linalg.norm(dx)
        if r < 0.2:  # Radius of influence of the airbag (can adjust this value)
            pressure_force = dx / r * airbag_pressure / (r + 1e-5)
            f[i] += pressure_force  # Apply the airbag force to the particle

# Compression forces between particles
def compute_snow_forces():
    global compression_force
    for i in range(n_particles):
        compression_force[i] = [0.0, 0.0]  # Reset compression forces

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dist = distance(i, j)
            if dist < neighbor_radius:
                dx = x[j] - x[i]
                dir = dx / dist  # Direction
                overlap = neighbor_radius - dist
                force_magnitude = overlap * 1e5  # Adjust stiffness
                force_magnitude = min(force_magnitude, snow_strength * radius**2)
                compression_force[i] -= dir * force_magnitude
                compression_force[j] += dir * force_magnitude

# Boundary collision handling (simple box bounds)
def apply_boundary_conditions():
    global x, v
    for i in range(n_particles):
        for d in range(2):  # Check each dimension (x, y)
            if x[i, d] < 0:
                x[i, d] = 0
                v[i, d] *= -0.3
            elif x[i, d] > domain_size:
                x[i, d] = domain_size
                v[i, d] *= -0.3

# Iterative solver
def iterative_solver():
    overlap_error = 1.0
    iter_count = 0
    min_iter = 10

    while overlap_error > 0.05 and iter_count < min_iter:
        # Step 1: Calculate external forces
        calculate_external_forces()
        
        # Step 2: Compute snow forces
        compute_snow_forces()

        # Step 3: Update velocities and positions
        for i in range(n_particles):
            # Estimate new velocity
            v_star = v[i] + (f[i] + accumulated_force[i]) / mass * dt
            v_star *= damping
            # Estimate new position
            x_star = x[i] + v_star * dt
            x[i] = x_star  # Update position

        # Step 4: Apply boundary conditions
        apply_boundary_conditions()

        # Step 5: Compute overlap error (simplified as max overlap)
        overlap_error = np.max(np.linalg.norm(compression_force, axis=1) / neighbor_radius)
        
        iter_count += 1

        if iter_count % 10 == 0:
            visualize()

# Visualization using Matplotlib
def visualize():
    plt.figure(figsize=(6, 6))
    plt.xlim(0, domain_size)
    plt.ylim(0, domain_size)
    plt.scatter(x[:, 0], x[:, 1], s=10, c='blue', alpha=0.5)
    plt.show()

# Main simulation loop
initialize()

# Running the simulation for a set number of iterations
for _ in range(100):  # 100 iterations as an example
    iterative_solver()
    # visualize() # commented-out since this occurs within the solver now every 10 iterations

    # Increase airbag pressure over time (simulate inflation)
    airbag_pressure += 5000  # Increment pressure at each step

