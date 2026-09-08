import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.boundary_conditions import (
    velocity_bc,
    pressure_bc
)

from src.utils.initial_condition import initialize_lamb_oseen

from src.utils.analytical import (
    sample_horizontal_centerline,
    analytical_vorticity,
)


from src.solvers.simulation import run_simulation

from src.utils.diagnostics import (
    extract_centerlines,
    compute_vorticity,
    extract_vorticity_centerlines,
    compute_divergence,
    compute_L2_error,
    compute_RMS_error,
    compute_Linf_error,
)

from src.utils.plots import (
    plot_velocity_validation,
    plot_vorticity_validation,
    plot_global_vs_interior_error,
)


# ==========================================================
#Domain 
# ==========================================================
Lx = 1.0 #domain length in x
Ly = 1.0 #domain length in y

dx = 0.0125 #grid spacing
dy = 0.0125

# number of grid points
Nx = int(Lx/dx) + 1
Ny = int(Ly/dy) + 1


# ==========================================================
#physical parameters
# ==========================================================
rho = 1.0  #ρ (rho) is the density of the fluid. Kg/m**3
mu = 0.001 #μ (mu) is the dynamic viscosity of the fluid. Kg/ms
nu = mu/rho #ν (nu) is the kinematic viscosity of the fluid. μ/ρ
'''It tells you how fast momentum diffuses through the fluid.
   Small ν → vortex remains concentrated for a long time.
   Large ν → vortex spreads out quickly.

   nu = 0.001   → weak diffusion
   nu = 0.01    → moderate diffusion
   nu = 0.1     → strong diffusion'''

Gamma = 1.0 # controls how strongly the vortex rotates.
rc = 0.1 # It determines how wide the vortex core is.

# ==========================================================
#time parameters
# ==========================================================
CFL = 0.1 # Courant number
umax = 1.0 # Maximum expected velocity
dt = (dx * CFL)/ umax # assuming max u = 1, 0.2 as relaxation parameter
# solver advances the solution by 0.01 s every iteration
'''umax = max(
    np.max(np.abs(ux)),
    np.max(np.abs(uy))
)

CFL = 0.25

dt = CFL * dx / umax'''

t_final = 1.0 # simulate until 100 seconds

nt = int(t_final/dt)  # Total number of iterations

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


'''
Why print these?
To verify that
vortex was created correctly
velocities are not zero everywhere
velocities are not absurdly large
'''
print(f"Initial max |ux| = {np.max(np.abs(ux)):.6f}")
print(f"Initial max |uy| = {np.max(np.abs(uy)):.6f}")

print("Initial condition generated")
print("ux shape =", ux.shape)
print("uy shape =", uy.shape)
print("p shape  =", p.shape)
print("rc = ", rc)
print("dx = ", dx)




# ==========================================================
# Simulation
# ==========================================================
# This will run the simulation for Total number of iterations
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
# Divergence history
# ==========================================================

divergence_history = []

for ux_n, uy_n in zip(ux_history, uy_history):

    div = compute_divergence(
        ux_n,
        uy_n,
        dx,
        dy
    )

    divergence_history.append(
        np.max(np.abs(div))
    )

# Plot maximum divergence versus time
plt.figure(figsize=(8, 5))

plt.semilogy(
    time_history,
    divergence_history
)

plt.xlabel("Time (s)")
plt.ylabel(r"Maximum $|\nabla \cdot \mathbf{u}|$")
plt.title(
    rf"Divergence History "
    rf"($r_c={rc}$, $\Delta x={dx}$, CFL={CFL}$)"
)

plt.grid(True)
plt.tight_layout()
plt.show()




# ==========================================================
# Validation plots
# ==========================================================

# Cell-centre coordinates
xp = np.linspace(dx, Lx - dx, Nx - 2)
yp = np.linspace(dy, Ly - dy, Ny - 2)
Xp, Yp = np.meshgrid(xp, yp)

# ------------------------------------------
# Vorticity evolution
# ------------------------------------------

times = np.linspace(0, t_final, 5)
interior_x_min = 0.3
interior_x_max = 0.7

# Storage for validation curves
x_num_all = []
omega_num_all = []
x_exact_all = []
omega_exact_all = []
vorticity_relative_errors = []
vorticity_interior_relative_errors = []
valid_times = []

# Check simulation time range
print("Number of stored frames =", len(time_history))
print("First stored time =", time_history[0])
print("Last stored time  =", time_history[-1])

for t in times:

    # Skip requested times outside simulation range
    if t < time_history[0] or t > time_history[-1]:
        print(
            f"Skipping t = {t:.2f} s "
            f"(simulation range: "
            f"{time_history[0]:.6f} to {time_history[-1]:.6f} s)"
        )
        continue

    # Find the stored frame closest to requested physical time
    frame = np.argmin(
        np.abs(np.asarray(time_history) - t)
    )

    actual_t = time_history[frame]

    print(
        f"Requested t = {t:.2f} s | "
        f"Using stored t = {actual_t:.6f} s | "
        f"frame = {frame}"
    )

    # ------------------------------------------
    # Numerical vorticity
    # ------------------------------------------

    omega = compute_vorticity(
        ux_history[frame],
        uy_history[frame],
        dx,
        dy
    )

    x_num, omega_num, _, _ = extract_vorticity_centerlines(
        omega,
        dx,
        dy
    )

    # ------------------------------------------
    # Analytical vorticity
    # ------------------------------------------

    omega_exact = analytical_vorticity(
        Xp,
        Yp,
        Gamma=Gamma,
        rc=rc,
        nu=nu,
        t=actual_t,
    )

    x_exact, omega_exact_h, _, _ = extract_vorticity_centerlines(
        omega_exact,
        dx,
        dy
    )

    # ------------------------------------------
    # Errors
    # ------------------------------------------

    relative_error = compute_L2_error(
        omega_num,
        omega_exact_h,
    )

    interior_mask = (
        (x_num >= interior_x_min) &
        (x_num <= interior_x_max)
    )

    interior_relative_error = compute_L2_error(
        omega_num[interior_mask],
        omega_exact_h[interior_mask],
    )

    RMS = compute_RMS_error(
        omega_num,
        omega_exact_h,
    )

    Linf = compute_Linf_error(
        omega_num,
        omega_exact_h,
    )

    print(
        f"t = {actual_t:>6.3f} s | "
        f"Relative Error = {relative_error:.6e} | "
        f"RMS = {RMS:.6e} | "
        f"L∞ = {Linf:.6e}"
    )

    # Store validation data
    x_num_all.append(x_num)
    omega_num_all.append(omega_num)
    x_exact_all.append(x_exact)
    omega_exact_all.append(omega_exact_h)
    vorticity_relative_errors.append(relative_error)
    vorticity_interior_relative_errors.append(
        interior_relative_error
    )
    valid_times.append(actual_t)


# ------------------------------------------
# Plot vorticity validation
# ------------------------------------------

plot_vorticity_validation(
    x_num_all,
    omega_num_all,
    x_exact_all,
    omega_exact_all,
    valid_times,
    dx=dx,
    CFL=CFL,
    rc=rc
)


# ==========================================================
# Velocity evolution and errors
# ==========================================================

velocity_x_num_all = []
velocity_num_all = []
velocity_x_exact_all = []
velocity_exact_all = []
velocity_relative_errors = []
velocity_interior_relative_errors = []

for actual_t in valid_times:
    frame = np.argmin(
        np.abs(np.asarray(time_history) - actual_t)
    )

    x_num, uy_num, _, _ = extract_centerlines(
        ux_history[frame],
        uy_history[frame],
        dx,
        dy,
    )

    uy_exact = sample_horizontal_centerline(
        x_num,
        Lx,
        Ly,
        Gamma,
        rc,
        nu,
        actual_t,
    )

    relative_error = compute_L2_error(uy_num, uy_exact)

    interior_mask = (
        (x_num >= interior_x_min) &
        (x_num <= interior_x_max)
    )

    interior_relative_error = compute_L2_error(
        uy_num[interior_mask],
        uy_exact[interior_mask],
    )

    RMS = compute_RMS_error(uy_num, uy_exact)
    Linf = compute_Linf_error(uy_num, uy_exact)

    print(
        f"Velocity t = {actual_t:>6.3f} s | "
        f"Relative Error = {relative_error:.6e} | "
        f"RMS = {RMS:.6e} | "
        f"L∞ = {Linf:.6e}"
    )

    velocity_x_num_all.append(x_num)
    velocity_num_all.append(uy_num)
    velocity_x_exact_all.append(x_num)
    velocity_exact_all.append(uy_exact)
    velocity_relative_errors.append(relative_error)
    velocity_interior_relative_errors.append(
        interior_relative_error
    )

plot_velocity_validation(
    velocity_x_num_all,
    velocity_num_all,
    velocity_x_exact_all,
    velocity_exact_all,
    valid_times,
    dx=dx,
    CFL=CFL,
    rc=rc
)


# ==========================================================
# Relative-error summary
# ==========================================================

print()
print("Relative-error summary")
print(f"{'time':>12} {'vorticity err':>18} {'velocity err':>18}")
print("-" * 50)

for t, vorticity_error, velocity_error in zip(
    valid_times,
    vorticity_relative_errors,
    velocity_relative_errors,
):
    print(
        f"{t:12.6f} "
        f"{vorticity_error:18.6e} "
        f"{velocity_error:18.6e}"
    )


print()
print(
    "Interior relative-error summary "
    f"({interior_x_min:.1f} <= x <= {interior_x_max:.1f})"
)
print(f"{'time':>12} {'vorticity err':>18} {'velocity err':>18}")
print("-" * 50)

for t, vorticity_error, velocity_error in zip(
    valid_times,
    vorticity_interior_relative_errors,
    velocity_interior_relative_errors,
):
    print(
        f"{t:12.6f} "
        f"{vorticity_error:18.6e} "
        f"{velocity_error:18.6e}"
    )



plot_global_vs_interior_error(
    valid_times,
    velocity_relative_errors,
    velocity_interior_relative_errors,
    vorticity_relative_errors,
    vorticity_interior_relative_errors,
    dx,
    CFL,
    rc
)