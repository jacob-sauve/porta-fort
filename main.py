# Jacob Sauvé, McGill University
# 2025/04/06

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
n_particles_x = 50
n_particles_y = 30
spacing = 0.02
radius = 0.01
mass = 1.0
dt = 0.01
damping = 0.98
snow_strength = 1.68e6  # Pa, compressive strength
airbag_center = np.array([0.5, 0.3])
airbag_pressure = 0.0  # increases over time
airbag_radius = 0.15

# Initialize particles
num_particles = n_particles_x * n_particles_y
positions = np.zeros((num_particles, 2))
velocities = np.zeros_like(positions)
forces = np.zeros_like(positions)

for i in range(n_particles_y):
    for j in range(n_particles_x):
        index = i * n_particles_x + j
        positions[index] = [j * spacing + 0.1, i * spacing + 0.1]

def compute_forces():
    global forces
    forces.fill(0)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dx = positions[j] - positions[i]
            dist = np.linalg.norm(dx)
            if 1e-5 < dist < 3 * radius:
                overlap = 3 * radius - dist
                direction = dx / dist
                force_mag = min(overlap * 1e5, snow_strength * radius**2)
                force = force_mag * direction
                forces[i] -= force
                forces[j] += force

def apply_airbag():
    global forces, airbag_pressure
    for i in range(num_particles):
        dx = positions[i] - airbag_center
        dist = np.linalg.norm(dx)
        if dist < airbag_radius:
            direction = dx / (dist + 1e-5)
            pressure_force = direction * airbag_pressure / (dist + 0.01)
            forces[i] += pressure_force

def update():
    global positions, velocities
    velocities += dt * forces / mass
    velocities *= damping
    positions += dt * velocities

    # Boundary conditions
    for i in range(num_particles):
        for d in range(2):
            if positions[i, d] < 0.0:
                positions[i, d] = 0.0
                velocities[i, d] *= -0.3
            elif positions[i, d] > 1.0:
                positions[i, d] = 1.0
                velocities[i, d] *= -0.3

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter(positions[:, 0], positions[:, 1], s=5, c='skyblue')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

def animate(frame):
    global airbag_pressure
    airbag_pressure += 500  # Increase pressure each frame
    compute_forces()
    apply_airbag()
    update()
    scat.set_offsets(positions)
    return scat,

ani = FuncAnimation(fig, animate, frames=300, interval=20, blit=True)
plt.show()
