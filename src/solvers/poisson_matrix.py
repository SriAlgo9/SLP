import numpy as np
from scipy.sparse import lil_matrix

def poisson_matrix(Nx, Ny, h):

    """
    Assemble the sparse matrix for the two-dimensional Pressure Poisson Equation.

    The Laplacian operator is discretized using the second-order central
    difference method. Homogeneous Neumann boundary conditions are imposed
    by modifying the stencil coefficients along the domain boundaries.

    Parameters
    ----------
    Nx : int
        Number of grid points in the x-direction.

    Ny : int
        Number of grid points in the y-direction.

    h : float
        Uniform grid spacing (dx = dy).

    Returns
    -------
    scipy.sparse.lil_matrix
        Sparse coefficient matrix representing the discrete Laplacian
        operator divided by h².

    Notes
    -----
    The matrix corresponds to the five-point finite difference stencil

            0   1   0
            1  -4   1
            0   1   0

    with Neumann boundary conditions implemented using mirrored stencil
    coefficients.
    """

    

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



