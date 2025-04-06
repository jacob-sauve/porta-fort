# Jacob Sauvé, McGill University
# 2025/04/06

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize Taichi
ti.init(arch=ti.gpu)

# ------------------ Simulation Parameters ------------------
dt = 1e-4
n_particles = int(input("Number of particles: "))
domain_width = 6.0 # metres
domain_height = 6.0 # metres
particle_radius = float(input("Particle radius (metres): ")) 
mass = 0.02
g = 9.81
contact_stiffness = 1e4
damping = 0.98
max_airbag_radius = 1.8 # metres
initial_airbag_radius = 0.05

# ------------------ Taichi Fields ------------------
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

# airbag starts 1.8 metres (1 radius) deep, as per port-a-fort specifications
airbag_center = ti.Vector([domain_width / 2, domain_height - max_airbag_radius])
airbag_radius = ti.field(dtype=ti.f32, shape=())
airbag_expansion_rate = ti.field(dtype=ti.f32, shape=())

# ------------------ Initialization ------------------
@ti.kernel
def initialize():
    cols = int(domain_width // (2 * particle_radius))
    for i in range(n_particles):
        row = i // cols
        col = i % cols
        x[i] = ti.Vector([
            (col + 0.5) * 2 * particle_radius,
            (row + 0.5) * 2 * particle_radius
        ])
        v[i] = ti.Vector([0.0, 0.0])
        f[i] = ti.Vector([0.0, 0.0])

# ------------------ Physics Kernels ------------------
@ti.kernel
def compute_external_forces():
    for i in range(n_particles):
        f[i] = ti.Vector([0.0, -mass * g])
        rel = x[i] - airbag_center
        dist = rel.norm()
        if dist < airbag_radius[None] and dist > 1e-5:
            f[i] += rel.normalized() * contact_stiffness * (airbag_radius[None] - dist)

@ti.kernel
def compute_contact_forces():
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            min_dist = 2 * particle_radius
            if dist < min_dist and dist > 1e-5:
                dir = dx.normalized()
                force = contact_stiffness * (min_dist - dist) * dir
                f[i] -= force
                f[j] += force

@ti.kernel
def apply_boundary_conditions():
    for i in range(n_particles):
        for d in ti.static(range(2)):
            if x[i][d] < particle_radius:
                x[i][d] = particle_radius
                v[i][d] *= -0.3
            elif (d == 0 and x[i][d] > domain_width - particle_radius) or \
                 (d == 1 and x[i][d] > domain_height - particle_radius):
                x[i][d] = domain_width - particle_radius if d == 0 else domain_height - particle_radius
                v[i][d] *= -0.3

@ti.kernel
def update_particles():
    for i in range(n_particles):
        v[i] += dt * f[i] / mass
        v[i] *= damping
        x[i] += dt * v[i]

@ti.kernel
def update_airbag():
    if airbag_radius[None] < max_airbag_radius:
        airbag_radius[None] += airbag_expansion_rate[None] * dt
        if airbag_radius[None] > max_airbag_radius:
            airbag_radius[None] = max_airbag_radius

# ------------------ Main Loop ------------------
rate = float(input("Enter the airbag expansion rate (m/s): "))
airbag_expansion_rate[None] = rate
airbag_radius[None] = initial_airbag_radius

expand_duration = (max_airbag_radius - initial_airbag_radius) / rate
expand_frame_end = int(expand_duration / dt) + 20

initialize()

# Choose how many frames to sample for visualization
frames_to_sample = 50
sampled_frame_indices = np.linspace(0, expand_frame_end, frames_to_sample, dtype=int)
frames = []

print("Simulating...")
for frame in range(expand_frame_end + 1):
    compute_external_forces()
    compute_contact_forces()
    apply_boundary_conditions()
    update_particles()
    update_airbag()
    if frame in sampled_frame_indices:
        frames.append(x.to_numpy())

print("Simulation complete. Visualizing...")

# ------------------ Slider Visualization ------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=1, c='blue')
ax.set_xlim(0, domain_width)
ax.set_ylim(0, domain_height)
ax.set_title("Snow Displacement Simulation")

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    scat.set_offsets(frames[idx])
    fig.canvas.draw_idle()

slider.on_changed(update)
fig.set_size_inches(6, 6) # lock figure size to square for representative plotting
plt.show()
