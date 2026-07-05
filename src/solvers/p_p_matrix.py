import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from prettytable import PrettyTable
import time

start = time.time()

# ============================================================
# Matrix Assembly (Central Difference Neumann BC)
# ============================================================

def poisson_matrix(Nx, Ny, h):

    N = Nx * Ny

    A = lil_matrix((N, N))

    def idx(j, i):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):

            p = idx(j, i) #p=row

            A[p, p] = -4.0

            # --------------------------------
            # x-direction
            # --------------------------------

            if i == 0:

                # left boundary
                A[p, idx(j, i + 1)] += 2.0

            elif i == Nx - 1:

                # right boundary
                A[p, idx(j, i - 1)] += 2.0

            else:

                A[p, idx(j, i - 1)] += 1.0
                A[p, idx(j, i + 1)] += 1.0

            # --------------------------------
            # y-direction
            # --------------------------------

            if j == 0:

                # bottom boundary
                A[p, idx(j + 1, i)] += 2.0

            elif j == Ny - 1:

                # top boundary
                A[p, idx(j - 1, i)] += 2.0

            else:

                A[p, idx(j - 1, i)] += 1.0
                A[p, idx(j + 1, i)] += 1.0

    return A / h**2


# ============================================================
# Problem Parameters
# ============================================================

dimx = 1.0
dimy = 1.0

hx = 0.01
hy = 0.01

Nx = int(dimx / hx) + 1
Ny = int(dimy / hy) + 1

print("Ndof:", Nx * Ny)

# ============================================================
# Single Solve
# ============================================================

h = hx # considering hx = hy

A = poisson_matrix(Nx, Ny, h)

# Remove nullspace (Neumann gauge condition)

A[0, :] = 0.0



A[0, 0] = 1.0

A = A.tocsc()

solve_poisson = factorized(A)

# Grid

# Pressure locations
xp = np.linspace(dx/2, Lx-dx/2, Nx)
yp = np.linspace(dy/2, Ly-dy/2, Ny)
Xp, Yp = np.meshgrid(xp, yp)

# u-velocity locations
xu = np.linspace(dx, Lx-dx, Nx-1)
yu = yp
Xu, Yu = np.meshgrid(xu, yu)

# v-velocity locations
xv = xp
yv = np.linspace(dy, Ly-dy, Ny-1)
Xv, Yv = np.meshgrid(xv, yv)

'''x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)

X, Y = np.meshgrid(x, y)'''

# RHS

b = np.zeros(Nx * Ny)

for j in range(Ny):
    for i in range(Nx):

        k = j * Nx + i

        b[k] = -2.0 * np.pi**2 * \
               np.cos(np.pi * x[i]) * \
               np.cos(np.pi * y[j])

# Gauge

b[0] = 0.0

# Solve

p_vec = solve_poisson(b)

# Exact solution

p_exact = np.cos(np.pi * X) * np.cos(np.pi * Y)

p_exact_vec = p_exact.reshape(-1)

# Match gauge

p_exact_vec -= p_exact_vec[0]

# Error

error = np.linalg.norm(
    p_vec - p_exact_vec
) / np.linalg.norm(p_exact_vec)

print("\nRelative Error =", error)

# ============================================================
# Error Function
# ============================================================

def error_poisson(h):

    Nx = int(dimx / h) + 1
    Ny = int(dimy / h) + 1

    print("Ndof:", Nx * Ny)

    A = poisson_matrix(Nx, Ny, h)

    A[0, :] = 0.0
    A[0, 0] = 1.0

    A = A.tocsc()

    solve_poisson = factorized(A)

    # Grid

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)

    X, Y = np.meshgrid(x, y)

    # RHS

    b = np.zeros(Nx * Ny)

    for j in range(Ny):
        for i in range(Nx):

            k = j * Nx + i

            b[k] = -2.0 * np.pi**2 * \
                   np.cos(np.pi * x[i]) * \
                   np.cos(np.pi * y[j])

    b[0] = 0.0

    # Solve

    p_vec = solve_poisson(b)

    # Exact

    p_exact = np.cos(np.pi * X) * np.cos(np.pi * Y)

    p_exact_vec = p_exact.reshape(-1)

    p_exact_vec -= p_exact_vec[0]

    error = np.linalg.norm(
        p_vec - p_exact_vec
    ) / np.linalg.norm(p_exact_vec)

    return error


# ============================================================
# Grid Convergence Study
# ============================================================

hs = [0.05, 0.025, 0.0125, 0.00625]

errors = []
times = []
ndofs = []

for h in hs:

    start_h = time.time()

    err = error_poisson(h)

    end_h = time.time()

    errors.append(err)

    times.append(end_h - start_h)

    Nx = int(dimx / h) + 1
    Ny = int(dimy / h) + 1

    ndofs.append(Nx * Ny)

# ============================================================
# Observed Order
# ============================================================

orders = ["-"]

for i in range(1, len(errors)):

    p = np.log(errors[i - 1] / errors[i]) / np.log(2)

    orders.append(f"{p:.4f}")

# ============================================================
# Results Table
# ============================================================

table = PrettyTable()

table.field_names = [
    "h",
    "Ndof",
    "Relative L2 Error",
    "Order p",
    "Time (s)"
]

for i in range(len(hs)):

    table.add_row([
        f"{hs[i]:.5f}",
        f"{ndofs[i]:,}",
        f"{errors[i]:.6e}",
        orders[i],
        f"{times[i]:.4f}"
    ])

print("\nGrid Convergence Results")
print(table)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 6))

plt.loglog(
    hs,
    errors,
    'o-',
    linewidth=2,
    markersize=8
)

plt.xlabel("Grid size h")
plt.ylabel("Relative L2 Error")
plt.title("Grid Convergence Study (Sparse Matrix Solver)")

plt.grid(True, which='both')

plt.tight_layout()

plt.show()

# ============================================================
# Runtime
# ============================================================

end = time.time()

print("\nTotal Execution Time =", end - start, "seconds")



from scipy.sparse.linalg import factorized

def solve_pressure_poisson(b, Nx, Ny, h):

    A = poisson_matrix(Nx, Ny, h)

    # Remove nullspace (pressure gauge)
    A[0, :] = 0.0
    A[0, 0] = 1.0

    A = A.tocsc()

    solve = factorized(A)

    b = b.copy()
    b[0] = 0.0

    p = solve(b)

    return p.reshape((Ny, Nx))