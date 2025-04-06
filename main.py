# Jacob Sauvé, McGill University
# 2025/04/06

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ti.init(arch=ti.gpu)

# Simulation parameters
n_particles = int(input("Number of particles: "))
dt = 1e-3
domain_size = 1.0 # scaled, this represents 4.1 metres
compression_strength = 1.68e6  # Pa (for particle repulsion)
damping = 0.98
EPSILON = 1e-5

# Airbag parameters
max_airbag_radius = 3.6/4.1  # Maximum airbag radius (scaled)
airbag_center = ti.Vector.field(2, dtype=ti.f32, shape=())
airbag_radius = ti.field(dtype=ti.f32, shape=())  # airbag radius, to grow over time

# Particle fields
radius = ti.field(dtype=ti.f32, shape=())  # Particle size
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

mass = 1.0 # arbitrary, mass of each particle

# ---------------------------------------------------
# Kernels
# ---------------------------------------------------

@ti.kernel
def initialize():
    grid_x = int(ti.sqrt(n_particles))
    
    # Adjust the radius based on the number of particles
    global radius
    radius[None] = 0.005 * (1.0 / ti.sqrt(n_particles))  # Increases radius if fewer particles
    
    # Position particles in the bottom two-thirds of the domain
    for i in range(n_particles):
        x[i] = ti.Vector([ (i % grid_x) * radius[None] * 2 + 0.05,
                           (i // grid_x) * radius[None] * 2 + 0.05 ])
        # Restrict particles to the bottom 2/3 of the domain
        x[i][1] = ti.min(x[i][1], 0.66 * domain_size)
        
        v[i] = ti.Vector([0.0, 0.0])
        f[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_external_forces():
    for i in range(n_particles):
        # Gravity force (downward)
        f[i] = ti.Vector([0.0, -9.81 * mass])
        
        # Airbag force: Repulsion if particle is within the expanding airbag radius
        ac = airbag_center[None]
        dx = x[i] - ac
        r = dx.norm()
        
        if r < airbag_radius[None] and r > EPSILON:
            # Repulsive force is proportional to how far inside the airbag's radius the particle is
            force_magnitude = 1e4 * (airbag_radius[None] - r)  # Stiffness of repulsion
            f[i] += dx.normalized() * force_magnitude

@ti.kernel
def compute_compression_forces():
    # Add repulsive forces between particles (simulating snow compression)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < radius[None] * 3 and dist > EPSILON:
                dir = dx.normalized()
                overlap = radius[None] * 3 - dist
                # Linear repulsion (adjust 1e5 as stiffness parameter)
                force_magnitude = overlap * 1e5
                force_magnitude = ti.min(force_magnitude, compression_strength * radius[None] * radius[None])
                f[i] -= dir * force_magnitude
                f[j] += dir * force_magnitude

@ti.kernel
def apply_boundary_conditions():
    for i in range(n_particles):
        for d in ti.static(range(2)):
            if x[i][d] < 0:
                x[i][d] = 0
                v[i][d] *= -0.3
            elif x[i][d] > domain_size:
                x[i][d] = domain_size
                v[i][d] *= -0.3

@ti.kernel
def update_particles():
    for i in range(n_particles):
        v[i] += (f[i] / mass) * dt
        v[i] *= damping
        x[i] += v[i] * dt

# ---------------------------------------------------
# Main simulation loop (runs on CPU, calling Taichi kernels)
# ---------------------------------------------------

# Get the representative airbag expansion rate from the user
expansion_rate = float(input("Enter the airbag expansion rate (m/s): "))
airbag_radius[None] = 0.0  # Start with a small airbag radius
airbag_center[None] = ti.Vector([0.5, 1.0])  # Airbag center, 1.8 meters below the surface

initialize()

# List to store simulation frames (particle positions)
frames = []
save_interval = 20  # Save 1 out of every 20 frames
num_sim_frames = int(input("Maximal simulated frames: "))  # Total simulation steps (adjust as needed)

for frame in range(num_sim_frames):
    # Increase the airbag radius progressively
    airbag_radius[None] += expansion_rate * dt
    
    # Ensure airbag doesn't grow beyond the maximum size
    if airbag_radius[None] > max_airbag_radius:
        airbag_radius[None] = max_airbag_radius

    compute_external_forces()
    compute_compression_forces()
    apply_boundary_conditions()
    update_particles()

    if frame % save_interval == 0:
        frames.append(x.to_numpy())

# ---------------------------------------------------
# Visualization with matplotlib slider
# ---------------------------------------------------
fig, ax = plt.subplots()
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)
scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=2, c='blue')

def update_plot(val):
    idx = int(val)
    scat.set_offsets(frames[idx])
    fig.canvas.draw_idle()

ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(frames)-1, valinit=0, valstep=1)
slider.on_changed(update_plot)

plt.show()

