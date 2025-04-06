# Jacob Sauvé, McGill University
# 2025/04/06

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# Parameters for the simulation
n_particles = 3000
radius = 0.005  # Radius of each snow particle
mass = 1.0  # Mass of each particle
domain_size = 1.0  # Size of the simulation domain
neighbor_radius = 3 * radius  # Neighbor radius for snow interaction
EPSILON = 1e-5
compression_strength = 1.68e6  # Pa (compressive strength of snow)

# Airbag and simulation settings
airbag_diameter = 3.6  # meters
airbag_radius = airbag_diameter / 2
airbag_center = np.array([0.5, 0.2])  # Center of the airbag
airbag_pressure = 0.0  # Initial airbag pressure (input by user)

# Particle data
x = np.random.rand(n_particles, 2) * domain_size  # Random initial positions
v = np.zeros_like(x)  # Initial velocities
f = np.zeros_like(x)  # Forces
compression_forces = np.zeros_like(x)  # Compression forces
accumulated_forces = np.zeros_like(x)  # Accumulated forces

# Array to store frames for slider
frames = []

# Get airbag pressure from user
airbag_pressure_input = float(input("Enter the airbag pressure (Pa): "))

# Calculate the force produced by the airbag based on input pressure
def compute_forces():
    global f, compression_forces, accumulated_forces
    # Reset forces
    f.fill(0)
    compression_forces.fill(0)
    accumulated_forces.fill(0)
    
    # External forces (gravity, airbag)
    for i in range(n_particles):
        # Airbag force (similar to before)
        dx = x[i] - airbag_center
        r = np.linalg.norm(dx)
        if r < airbag_radius:  # Airbag influence
            pressure_force = dx / r * airbag_pressure / (r + EPSILON)
            f[i] += pressure_force  # Apply the airbag force to the particle
        
    # Particle interaction forces (snow compression)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = x[j] - x[i]
            dist = np.linalg.norm(dx)
            if dist < neighbor_radius and dist > EPSILON:
                dir = dx / dist
                overlap = neighbor_radius - dist
                # Linear repulsive force proportional to overlap
                force_magnitude = overlap * 1e5  # Adjust stiffness
                force_magnitude = min(force_magnitude, compression_strength * radius**2)
                compression_forces[i] -= dir * force_magnitude
                compression_forces[j] += dir * force_magnitude

    # Apply boundary forces and snow forces (compression)
    for i in range(n_particles):
        # Apply boundary forces
        for d in range(2):
            if x[i][d] < 0:
                x[i][d] = 0
                v[i][d] *= -0.3
            elif x[i][d] > domain_size:
                x[i][d] = domain_size
                v[i][d] *= -0.3
        
        # Apply compression forces
        accumulated_forces[i] += compression_forces[i]

def update():
    global x, v
    for i in range(n_particles):
        # Estimate new velocity
        v_star = v[i] + (f[i] + accumulated_forces[i]) * 1e-3 / mass
        # Estimate new position
        x_star = x[i] + v_star * 1e-3
        x[i] = x_star
        v[i] = v_star

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)

# Plot particles
scatter = ax.scatter(x[:, 0], x[:, 1], s=5)

# Simulation main loop with frame saving
def simulate():
    global airbag_pressure, frames
    overlap_error = 1.0
    iter_count = 0
    frame_step = 20  # Save 1 out of every 20 frames

    while iter_count < 500:  # Run for 500 iterations or adjust as needed
        airbag_pressure = airbag_pressure_input * (iter_count / 500.0)  # Ramp up pressure
        
        for frame in range(1000):  # Simulate for 1000 frames (or adjust as needed)
            compute_forces()  # Calculate forces
            update()  # Update particle positions
            if frame % frame_step == 0:  # Save every 20th frame
                frames.append(np.copy(x))
        
        iter_count += 1

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
