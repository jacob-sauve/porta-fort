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

# Particle data
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

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
def compute_forces():
    for i in range(n_particles):
        f[i] = [0.0, 0.0]  # Reset forces

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < neighbor_radius and dist > 1e-5:
                dir = dx.normalized()
                overlap = neighbor_radius - dist
                # Linear repulsive force proportional to overlap
                force_magnitude = overlap * 1e5  # adjust stiffness
                # Clamp to snow compressive strength
                force_magnitude = min(force_magnitude, snow_strength * radius**2)
                f[i] -= dir * force_magnitude
                f[j] += dir * force_magnitude

@ti.kernel
def apply_airbag():
    for i in range(n_particles):
        dx = x[i] - airbag_center[None]
        r = dx.norm()
        if r < 0.2:  # radius of influence
            pressure_force = dx.normalized() * airbag_pressure[None] / (r + 1e-5)
            f[i] += pressure_force

@ti.kernel
def update():
    for i in range(n_particles):
        v[i] += dt * f[i] / mass
        v[i] *= damping
        x[i] += dt * v[i]

        # Simple boundary collision
        for d in ti.static(range(2)):
            if x[i][d] < 0:
                x[i][d] = 0
                v[i][d] *= -0.3
            elif x[i][d] > domain_size:
                x[i][d] = domain_size
                v[i][d] *= -0.3

# GUI
initialize()
gui = ti.GUI("Snow Simulation with Airbag", res=600, background_color=0x112F41)

# Airbag initial state
airbag_center[None] = ti.Vector([0.5, 0.2])
airbag_pressure[None] = 0.0

while gui.running:
    # Increase pressure over time
    airbag_pressure[None] += 1e3

    compute_forces()
    apply_airbag()
    update()

    positions = x.to_numpy()
    gui.circles(positions, radius=2, color=0xAAAAFF)
    gui.show()
