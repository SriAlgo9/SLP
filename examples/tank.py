import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.boundary_conditions import (
    velocity_bc,
    pressure_bc
)

from src.utils.initial_condition import initialize_lamb_oseen

from src.utils.analytical import (
    sample_horizontal_centerline,
)


from src.solvers.simulation import run_simulation

from src.utils.diagnostics import (
    extract_centerlines,
    compute_vorticity,
    extract_vorticity_centerlines,
)


# ==========================================================
#Domain 
# ==========================================================
Lx = 1.0 #domain length in x
Ly = 1.0 #domain length in y

dx = 0.05 #grid spacing
dy = 0.05

Nx = int(Lx/dx) + 1
Ny = int(Ly/dy) + 1


# ==========================================================
#physical parameters
# ==========================================================
rho = 1.0  #ρ (rho) is the density of the fluid.
mu = 0.001 #μ (mu) is the dynamic viscosity of the fluid.
nu = mu/rho #ν (nu) is the kinematic viscosity of the fluid.
'''It tells you how fast momentum diffuses through the fluid.
   Small ν → vortex remains concentrated for a long time.
   Large ν → vortex spreads out quickly.

   nu = 0.001   → weak diffusion
   nu = 0.01    → moderate diffusion
   nu = 0.1     → strong diffusion'''

Gamma = 1.0
rc = 0.10

# ==========================================================
#time parameters
# ==========================================================
dt = dx * 0.2 # assuming max u = 1, 0.2 as relaxation parameter
'''umax = max(
    np.max(np.abs(ux)),
    np.max(np.abs(uy))
)

CFL = 0.25

dt = CFL * dx / umax'''

t_final = 100

nt = int(t_final/dt)  #number of time steps

print ("Number of time steps =", nt)


# ==========================================================
# Initial condition: Lamb-Oseen vortex
# ==========================================================
ux, uy, p = initialize_lamb_oseen(
    Nx, 
    Ny, 
    dx, 
    dy, 
    rho=rho,    
    Gamma=Gamma, 
    rc=rc,
    )

ux, uy = velocity_bc(ux, uy)
p = pressure_bc(p)

print(f"Initial max |ux| = {np.max(np.abs(ux)):.6f}")
print(f"Initial max |uy| = {np.max(np.abs(uy)):.6f}")

print("Initial condition generated")
print("ux shape =", ux.shape)
print("uy shape =", uy.shape)
print("p shape  =", p.shape)

# ==========================================================
# Simulation
# ==========================================================
(
    ux,
    uy,
    p,
    ux_history,
    uy_history,
    p_history,
    time_history,
) = run_simulation(
    ux=ux,
    uy=uy,
    p=p,
    rho=rho,
    nu=nu,
    dx=dx,
    dy=dy,
    dt=dt,
    nt=nt,
    Nx=Nx,
    Ny=Ny,
)



# ==========================================================
# Validation plots
# ==========================================================

# ------------------------------------------
# velocity evolution
# ------------------------------------------
times = [0, 1, 2, 5, 10, 20, 40, 60, 80, 100]

for t in times:

    frame = int(t / dt)

    omega = compute_vorticity(
        ux_history[frame],
        uy_history[frame],
        dx,
        dy
    )

    x, omega_h, _, _ = extract_vorticity_centerlines(
        omega,
        dx,
        dy
    )

    plt.plot(
        x,
        omega_h,
        linewidth=2,
        label=f"t = {t:.1f} s"
    )

plt.xlabel("x")
plt.ylabel(r"$\omega$")
plt.title("Evolution of Horizontal Centreline Vorticity")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------
# Analytical horizontal centreline
# ------------------------------------------

# Smooth analytical curve (300 points)

x_exact = np.linspace(dx, Lx - dx, 300)

# ------------------------------------------
# Analytical at t = 0
# ------------------------------------------

uy_exact_t0 = sample_horizontal_centerline(
    x_exact,
    Lx,
    Ly,
    Gamma,
    rc,
    nu,
    0.0,
)

# ------------------------------------------
# Analytical at t = 1
# ------------------------------------------

uy_exact_t1 = sample_horizontal_centerline(
    x_exact,
    Lx,
    Ly,
    Gamma,
    rc,
    nu,
    1.0,
)


# ==========================================================
# Extract numerical velocity at t = 0 and t = 1
# ==========================================================

# t = 0 s
frame0 = 0

x0, uy0, y0, ux0 = extract_centerlines(
    ux_history[frame0],
    uy_history[frame0],
    dx,
    dy
)

# t = 1 s
frame1 = int(1.0 / dt)

x1, uy1, y1, ux1 = extract_centerlines(
    ux_history[frame1],
    uy_history[frame1],
    dx,
    dy
)

uy_exact_points_t0 = sample_horizontal_centerline(
    x0,
    Lx,
    Ly,
    Gamma,
    rc,
    nu,
    0.0,
)

uy_exact_points_t1 = sample_horizontal_centerline(
    x1,
    Lx,
    Ly,
    Gamma,
    rc,
    nu,
    1.0,
)



plt.figure(figsize=(8,5))

# Numerical t = 0
plt.plot(
    x0,
    uy0,
    color="blue",
    linewidth=2,
    label="Numerical t = 0 s"
)

# Analytical t = 0 (dots only)
plt.scatter(
    x0,
    uy_exact_points_t0,
    color="blue",
    s=40,
    label="Analytical t = 0 s"
)

# Numerical t = 1
plt.plot(
    x1,
    uy1,
    color="red",
    linewidth=2,
    label="Numerical t = 1 s"
)

# Analytical t = 1 (dots only)
plt.scatter(
    x1,
    uy_exact_points_t1,
    color="red",
    s=40,
    label="Analytical t = 1 s"
)

plt.xlabel("x")
plt.ylabel(r"$u_y$")
plt.title("Horizontal Centreline Velocity Validation")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

