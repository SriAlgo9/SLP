import numpy as np
from scipy.sparse import lil_matrix

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



# RHS for Pressure Poisson Equation

def build_rhs(ux_star, uy_star, rho, dt, dx, dy):

    Ny = ux_star.shape[0]
    Nx = uy_star.shape[1]

    b = np.zeros(Nx * Ny)

    for j in range(Ny):
        for i in range(Nx):

            k = j * Nx + i

            dudx = 0.0
            dvdy = 0.0

            if 0 < i < Nx-1:
                dudx = (
                    ux_star[j, i]
                    - ux_star[j, i-1]
                ) / dx

            if 0 < j < Ny-1:
                dvdy = (
                    uy_star[j, i]
                    - uy_star[j-1, i]
                ) / dy

            b[k] = (rho/dt) * (dudx + dvdy)

    return b



'''def build_rhs(ux_star, uy_star, rho, dt, dx, dy):

    Ny, Nx = ux_star.shape

    b = np.zeros(Nx*Ny)

    for j in range(1, Ny-1):
        for i in range(1, Nx-1):

            k = j*Nx + i

            b[k] = (rho/dt)*(
                (ux_star[j,i+1] - ux_star[j,i-1])/(2*dx)
                +
                (uy_star[j+1,i] - uy_star[j-1,i])/(2*dy)
            )

    return b'''
