# Jacob Sauvé, McGill University
# 2025/04/06

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters for the simulation
n_particles = 3000
radius = 0.005
mass = 1.0
airbag_pressure = 0.0
airbag_center = np.array([0.5, 0.2])
domain_size = 1.0
neighbor_radius = 3 * radius
EPSILON = 1e-5

# Particle data
x = np.random.rand(n_particles, 2) * domain_size  # Random initial positions
v = np.zeros_like(x)  # Initial velocities
f = np.zeros_like(x)  # Forces

# Array to store frames
frames = []

# Airbag pressure update
def update_airbag_pressure(val):
    global airbag_pressure
    airbag_pressure = val
    compute_forces()
    update()

def compute_forces():
    global f
    # Apply airbag force
    for i in range(n_particles):
        dx = x[i] - airbag_center
        r = np.linalg.norm(dx)
        if r < 0.2 and r > EPSILON:
            pressure_force = dx / r * airbag_pressure / (r + EPSILON)
            f[i] += pressure_force  # Apply the airbag force to the particle

def update():
    global x, v
    # Update particle velocities and positions
    for i in range(n_particles):
        v[i] += f[i] * 1e-3 / mass  # Update velocity
        x[i] += v[i] * 1e-3  # Update position

        # Boundary conditions
        for d in range(2):
            if x[i][d] < 0:
                x[i][d] = 0
                v[i][d] *= -0.3
            elif x[i][d] > domain_size:
                x[i][d] = domain_size
                v[i][d] *= -0.3

# Create the plot and sliders
fig, ax = plt.subplots()
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)

# Create the slider for airbag pressure
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Airbag Pressure', 0, 1000, valinit=0)

slider.on_changed(update_airbag_pressure)

# Plot particles
scatter = ax.scatter(x[:, 0], x[:, 1], s=5)

# Main loop to update the simulation and save frames
def simulate():
    for frame in range(1000):  # Simulate for 1000 frames (or adjust as needed)
        update()
        frames.append(np.copy(x))  # Save particle positions at this frame

simulate()

# Function to display frames based on slider value
def update_frame(val):
    frame = int(val)
    scatter.set_offsets(frames[frame])
    plt.draw()

# Create the slider to iterate through frames
ax_frame_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
frame_slider = Slider(ax_frame_slider, 'Frame', 0, len(frames)-1, valinit=0, valstep=1)

frame_slider.on_changed(update_frame)

# Display the initial frame
update_frame(0)

plt.show()

