import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils.boundary_conditions import (
    velocity_bc,
    pressure_bc
)

from scipy.sparse.linalg import factorized


from src.utils.explicit_scheme import (
    compute_ux_star,
    compute_uy_star,
    compute_ux,
    compute_uy
)

from src.utils.initial_condition import initialize_lamb_oseen

from src.solvers.poisson_matrix import (
    poisson_matrix,
    build_rhs
)

#Domain 
Lx = 1.0 #domain length in x
Ly = 1.0 #domain length in y

dx = 0.01 #grid spacing
dy = 0.01

Nx = int(Lx/dx) + 1
Ny = int(Ly/dy) + 1

#physical parameters

rho = 1.0  #ρ (rho) is the density of the fluid.
mu = 0.001 #μ (mu) is the dynamic viscosity of the fluid.

nu = mu/rho #ν (nu) is the kinematic viscosity of the fluid.
'''It tells you how fast momentum diffuses through the fluid.
   Small ν → vortex remains concentrated for a long time.
   Large ν → vortex spreads out quickly.

   nu = 0.001   → weak diffusion
   nu = 0.01    → moderate diffusion
   nu = 0.1     → strong diffusion'''

#time parameters

dt = dx * 0.2 # assuming max u = 1, 0.5 as relaxation parameter
'''umax = max(
    np.max(np.abs(ux)),
    np.max(np.abs(uy))
)

CFL = 0.25

dt = CFL * dx / umax'''

t_final = 1.0

nt = int(t_final/dt)  #number of time steps

print ("Number of time steps =", nt)

'''#initial condition

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)

X,Y = np.meshgrid(x,y)

ux = np.zeros((Ny, Nx),dtype = np.float64)
uy = np.zeros((Ny, Nx),dtype = np.float64)

p = np.cos(np.pi * X) * np.cos(np.pi * Y)'''

# Initial condition: Lamb-Oseen vortex

ux, uy, p = initialize_lamb_oseen(Nx, Ny, dx, dy, rho=rho, Gamma=1.0, rc=0.1)
ux, uy = velocity_bc(ux, uy)
print(np.max(ux))
print(np.max(uy))
p = pressure_bc(p)
print("Initial condition generated")
print("ux shape =", ux.shape)
print("uy shape =", uy.shape)
print("p shape  =", p.shape)

A = poisson_matrix(Nx, Ny, dx)
# Fix pressure gauge

A[0, :] = 0.0
A[0, 0] = 1.0

A = A.tocsc()

# LU factorization

solve_poisson = factorized(A)

ux_history = [ux.copy()]
uy_history = [uy.copy()]

# Time loop
for n in range(nt):

    # Explicit convection step
    ux_star = compute_ux_star(ux, uy, dt, dx, dy, nu)
    uy_star = compute_uy_star(ux, uy, dt, dx, dy, nu)

    ux_star, uy_star = velocity_bc(ux_star, uy_star)

    # Pressure RHS
    b = build_rhs(ux_star, uy_star, rho, dt, dx, dy)

    # Pressure solve
    b[0] = 0.0
    p_vec = solve_poisson(b)
    p = p_vec.reshape((Ny, Nx))

    p = pressure_bc(p)

    # Velocity correction
    ux = compute_ux(ux_star, dt, dx, rho, p)
    uy = compute_uy(uy_star, dt, dy, rho, p)

    ux, uy = velocity_bc(ux, uy)

    # Compute velocity magnitude

    u_center = 0.5*(ux[1:-1,:-1] + ux[1:-1,1:])
    v_center = 0.5*(uy[:-1,1:-1] + uy[1:,1:-1])

    speed = np.sqrt(u_center**2 + v_center**2)

    '''speed = np.sqrt(ux**2 + uy**2)'''

    # Compute divergence

    div = (
    (ux[1:-1,1:] - ux[1:-1,:-1]) / dx
    +
    (uy[1:,1:-1] - uy[:-1,1:-1]) / dy
)



    '''div = (
        (ux[1:-1, 2:] - ux[1:-1, :-2]) / (2 * dx)
        +
        (uy[2:, 1:-1] - uy[:-2, 1:-1]) / (2 * dy)
    )'''

    # Save history
    ux_history.append(ux.copy())
    uy_history.append(uy.copy())

    # Print every 10 iterations
    if n % 10 == 0:
        print(
             f"Step {n:3d} | "
             f"max speed = {np.max(speed):.6f} | "
             f"max divergence = {np.max(np.abs(div)):.6e}"
        )

# FINAL OUTPUT
print(ux)
#print("max |ux| =", np.max(np.abs(ux)))
#print("max |uy| =", np.max(np.abs(uy)))
#print("max |p|  =", np.max(np.abs(p)))

##

'''x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
Xg, Yg = np.meshgrid(x,y)'''

ux1 = ux_history[0]
uy1 = uy_history[0]

fig, ax = plt.subplots()

u_plot = 0.5*(ux1[1:-1,:-1] + ux1[1:-1,1:])

v_plot = 0.5*(uy1[:-1,1:-1] + uy1[1:,1:-1])


xp = np.linspace(dx, Lx-dx, Nx-2)
yp = np.linspace(dy, Ly-dy, Ny-2)

Xp, Yp = np.meshgrid(xp, yp)

q = ax.quiver(Xp, Yp, u_plot, v_plot)

'''q = ax.quiver(Xg, Yg, ux1, uy1)'''

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect("equal")
ax.set_title("Velocity field")


def update(frame):

    ux_n = ux_history[frame]
    uy_n = uy_history[frame]

    u_plot = 0.5*(ux_n[1:-1,:-1] + ux_n[1:-1,1:])
    v_plot = 0.5*(uy_n[:-1,1:-1] + uy_n[1:,1:-1])

    q.set_UVC(u_plot,v_plot)

    ax.set_title(f"Velocity field at t={frame*dt:.3f}")

    return [q]


'''def update(frame):
  ux_n = ux_history[frame]
  uy_n = uy_history[frame]

  q.set_UVC(ux_n, uy_n)
  ax.set_title(f"Velocity field at t = {frame * dt:3f}")

  return [q]'''


ani = FuncAnimation(
  fig,
  update,
  frames = len(ux_history),
  interval = 10,
  blit = False,
)

plt.show()

    

    
    

