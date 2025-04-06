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
radius = 0.005
mass = 1.0
domain_size = 1.0
neighbor_radius = 3 * radius
compression_strength = 1.68e6  # Pa
damping = 0.98
EPSILON = 1e-5

# Airbag parameters
# Airbag diameter is 3.6m (so radius 1.8m), but here we work in scaled units.
airbag_center = ti.Vector.field(2, dtype=ti.f32, shape=())
airbag_pressure = ti.field(dtype=ti.f32, shape=())  # representative parameter

# Particle fields
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

# ---------------------------------------------------
# Kernels
# ---------------------------------------------------

@ti.kernel
def initialize():
    grid_x = int(ti.sqrt(n_particles))
    for i in range(n_particles):
        x[i] = ti.Vector([ (i % grid_x) * radius * 2 + 0.05,
                           (i // grid_x) * radius * 2 + 0.05 ])
        v[i] = ti.Vector([0.0, 0.0])
        f[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_external_forces():
    for i in range(n_particles):
        # Gravity force (downward)
        f[i] = ti.Vector([0.0, -9.81 * mass])
        # Airbag force: if particle is within airbag influence (here r < 0.2 in scaled units)
        ac = airbag_center[None]
        dx = x[i] - ac
        r = dx.norm()
        if r < 0.2 and r > EPSILON:
            # The force is proportional to the input pressure and inversely to distance.
            f[i] += dx.normalized() * airbag_pressure[None] / (r + EPSILON)

@ti.kernel
def compute_compression_forces():
    # Add repulsive forces between particles (simulating snow compression)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dx = x[j] - x[i]
            dist = dx.norm()
            if dist < neighbor_radius and dist > EPSILON:
                dir = dx.normalized()
                overlap = neighbor_radius - dist
                # Linear repulsion (adjust 1e5 as stiffness parameter)
                force_magnitude = overlap * 1e5
                force_magnitude = ti.min(force_magnitude, compression_strength * radius * radius)
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

# Get the representative airbag pressure parameter from the user
pressure_input = float(input("Enter the representative airbag pressure (Pa): "))
# Set the airbag center and pressure
airbag_center[None] = ti.Vector([0.5, 0.2])
airbag_pressure[None] = pressure_input

initialize()

# List to store simulation frames (particle positions)
frames = []
save_interval = 20  # Save 1 out of every 20 frames
num_sim_frames = int(input("Maximal simulated frames: "))  # Total simulation steps (adjust as needed)

for frame in range(num_sim_frames):
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

