# The polymer is assumed to be isotropic
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Function to compute e_ij values
def compute_eij(d, E, nu):
    e15 = d[15] * E / (2 * (1 + nu))
    e24 = d[24] * E / (2 * (1 + nu))
    e31 = (d[31] * (1 - nu) + (d[32] + d[33]) * nu) * E / ((1 - 2 * nu) * (1 + nu))
    e32 = (d[31] * nu + (d[32] + d[33]) * (1 - nu)) * E / ((1 - 2 * nu) * (1 + nu))
    e33 = ((d[32] + d[33]) * nu + d[31] * (1 - nu)) * E / ((1 - 2 * nu) * (1 + nu))
    return e15, e24, e31, e32, e33

# Define input parameters
# d_ij coefficients of PVDF in C/N
d_ij = {
    15: -4.67E-11,
    24: -4.67E-11,
    31: 2.3E-11,
    32: 2.3E-11,
    33: -3.3E-11
}
E = 2.7e9  # Pa
nu = 0.4
color_scale = 'Viridis'
matplotlib_color_map = 'viridis'

# Compute e_ij values
e15, e24, e31, e32, e33 = compute_eij(d_ij, E, nu)

# Display the input d_ij matrix without the factor
factor = 1.0E-12
print("\nInput [d_{ij}] Matrix, unit [pC/N]:")
d_ij_matrix = np.array([
    [0, 0, 0, 0, d_ij[15] / factor, 0],
    [0, 0, 0, d_ij[24] / factor, 0, 0],
    [d_ij[31] / factor, d_ij[32] / factor, d_ij[33] / factor, 0, 0, 0]
])
print(d_ij_matrix)

# Display computed e_ij values
print("\nComputed e_ij values (C/m²):")
print(f"e15 = {e15:.3e}")
print(f"e24 = {e24:.3e}")
print(f"e31 = {e31:.3e}")
print(f"e32 = {e32:.3e}")
print(f"e33 = {e33:.3e}")

# Generate spherical coordinates
theta = np.linspace(0, np.pi, 100)  # Polar angle
phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
theta, phi = np.meshgrid(theta, phi)

# Compute spherical harmonics (l = 3, m = 0 for simplicity)
l, m = 3, 0
Y_lm = sph_harm(abs(m), l, phi, theta)  # Spherical Harmonic Y_lm

# Map the spherical harmonic to a scalar field representing e_ij values
r = np.sqrt(e15**2 + e24**2 + e31**2 + e32**2 + e33**2)  # Scalar field magnitude
r_field = np.abs(Y_lm) * r

# Convert spherical coordinates to Cartesian coordinates
x = r_field * np.sin(theta) * np.cos(phi)
y = r_field * np.sin(theta) * np.sin(phi)
z = r_field * np.cos(theta)

# Plotly 3D spherical harmonics
trace = go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale=color_scale
)
layout = go.Layout(
    title=f'Spherical Harmonics for Y_{{{l},{m}}}',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)
fig = go.Figure(data=[trace], layout=layout)
fig.show()

# Matplotlib 3D Visualization of spherical harmonics
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    x, y, z,
    facecolors=plt.cm.get_cmap(matplotlib_color_map)(np.real(Y_lm)),
    rstride=2, cstride=2, alpha=0.6
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Spherical Harmonics for Y_{{{l},{m}}}')
plt.show()

