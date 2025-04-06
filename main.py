# Jacob Sauvé, McGill University
# 2025/04/06

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Use GPU for performance, fallback to CPU if needed

# Simulation parameters
n_particles = 3000
dt = 1e-3
radius = 0.005
mass = 1.0
snow_strength = 1.68e6  # Pa (compressive strength of snow)
airbag_pressure = ti.field(dtype=ti.f32, shape=())
airbag_center = ti.Vector.field(2, dtype=ti.f32, shape=())
airbag_radius = ti.field(dtype=ti.f32, shape=())

# Particle data
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
compression_force = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
accumulated_force = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
velocity_star = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
position_star = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

# Constants
damping = 0.98
domain_size = 1.0
neighbor_radius = 3 * radius

@ti.kernel
def initialize():
    for i in range(n_particles):
        grid_x = int(ti.sqrt(n_particles))
        x[i] = [
            (i % grid_x) * radius * 2 + 0.05,
            (i // grid_x) * radius * 2 + 0.05
        ]
        v[i] = [0.0, 0.0]
        f[i] = [0.0, 0.0]

@ti.kernel
def apply_external_forces():
    for i in range(n_particles):
        f[i] = [0.0, -9.81 * mass]  # Gravity force (downward)
        dx = x[i] - airbag_center[None]
        r = dx.norm()
        if r < airbag_radius[None]:
            # Apply growing airbag pressure force
            pressure_force = dx.normalized() * airbag_pressure[None] / (r + 1e-5)
            f[i] += pressure_force

@ti.kernel
def find_neighbors():
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < neighbor_radius:
                # Apply snow forces
                dir = dx.normalized()
                overlap = neighbor_radius - dist
                force_magnitude = overlap * 1e5  # Adjust stiffness
                force_magnitude = min(force_magnitude, snow_strength * radius**2)
                compression_force[i] -= dir * force_magnitude
                compression_force[j] += dir * force_magnitude

@ti.kernel
def update_forces():
    for i in range(n_particles):
        accumulated_force[i] = f[i]  # Reset accumulated forces
        
    # Apply snow forces (compression between particles)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < neighbor_radius:
                dir = dx.normalized()
                overlap = neighbor_radius - dist
                force_magnitude = overlap * 1e5  # Adjust stiffness
                force_magnitude = min(force_magnitude, snow_strength * radius**2)
                # Apply forces to both particles
                accumulated_force[i] -= dir * force_magnitude
                accumulated_force[j] += dir * force_magnitude

@ti.kernel
def iterative_solver():
    # Iterative process to refine positions and forces
    overlap_error = 1.0  # Initialize overlap error
    iter_count = 0

    while overlap_error > 0.05 and iter_count < 10:  # Loop with a convergence threshold
        for i in range(n_particles):
            # Estimate new velocity
            velocity_star[i] = v[i] + (accumulated_force[i] / mass) * dt
            
            # Estimate new position
            position_star[i] = x[i] + velocity_star[i] * dt

        # Recompute forces based on estimated positions
        find_neighbors()  # Recalculate neighbors and forces

        # Update compression forces
        update_forces()

        # Calculate the overlap error (or some other stopping criteria)
        overlap_error = compute_overlap_error()

        iter_count += 1

    # Update final velocity and position
    for i in range(n_particles):
        v[i] = velocity_star[i]
        x[i] = position_star[i]

def compute_overlap_error():
    max_overlap_error = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < neighbor_radius:
                overlap = neighbor_radius - dist
                max_overlap_error = max(max_overlap_error, overlap / neighbor_radius)
    return max_overlap_error

@ti.kernel
def integrate():
    for i in range(n_particles):
        v[i] += dt * accumulated_force[i] / mass  # Update velocity
        v[i] *= damping  # Apply damping

        x[i] += dt * v[i]  # Update position

# GUI
initialize()
gui = ti.GUI("Snow Simulation with Growing Airbag", res=600, background_color=0x112F41)

# Airbag initial state
airbag_center[None] = ti.Vector([0.5, 0.2])
airbag_pressure[None] = 0.0
airbag_radius[None] = 0.05  # Initial radius of the airbag

while gui.running:
    # Increase pressure and airbag radius gradually
    airbag_pressure[None] += 1e3  # Increase airbag pressure
    airbag_radius[None] += 0.001  # Gradually increase airbag radius (expand over time)

    # Apply forces, update positions, etc.
    apply_external_forces()
    find_neighbors()  # Find neighbors outside kernel loop
    update_forces()   # Update forces outside kernel loop
    iterative_solver()  # Refine positions and forces
    integrate()  # Finalize the velocity and positions

    # Visualize the particles
    positions = x.to_numpy()
    gui.circles(positions, radius=2, color=0xAAAAFF)
    gui.show()
