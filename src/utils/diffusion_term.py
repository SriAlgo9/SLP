import numpy as np

def diffusion_ux(ux, dx, dy, nu):
    """
    Compute the viscous diffusion term for the x-momentum equation.

    The diffusion term is evaluated using second-order central
    finite differences:

        D_ux = ν (∂²u/∂x² + ∂²u/∂y²)

    which is discretized as

        D_ux(i,j) = ν [
            (u(i+1,j) - 2u(i,j) + u(i-1,j))/dx²
          + (u(i,j+1) - 2u(i,j) + u(i,j-1))/dy²
        ]

    Parameters
    ----------
    ux : ndarray of shape (Ny, Nx)
        x-velocity field defined on the staggered u-grid.

    dx : float
        Grid spacing in the x-direction.

    dy : float
        Grid spacing in the y-direction.

    nu : float
        Kinematic viscosity [m²/s].

    Returns
    -------
    diff_ux : ndarray of shape (Ny, Nx)
        Diffusion contribution ν∇²u evaluated at all interior
        ux-control volumes.

    Notes
    -----
    - Second-order central difference approximation.
    - Boundary values are not computed here and remain zero.
    - Used in the predictor step of the explicit Euler solver.
    """

    diff_ux = np.zeros_like(ux)

    diff_ux[1:-1, 1:-1] = nu * (
        (ux[1:-1, 2:] - 2*ux[1:-1, 1:-1] + ux[1:-1, :-2]) / dx**2
      + (ux[2:, 1:-1] - 2*ux[1:-1, 1:-1] + ux[:-2, 1:-1]) / dy**2
    )

    return diff_ux

def diffusion_uy(uy, dx, dy, nu):
    """
    Compute the viscous diffusion term for the y-momentum equation.

    The diffusion term corresponds to

        D_uy = ν (∂²v/∂x² + ∂²v/∂y²)

    evaluated using second-order central differences on the
    staggered v-grid.

    Parameters
    ----------
    uy : ndarray of shape (Ny, Nx)
        y-velocity field defined on the staggered v-grid.

    dx : float
        Grid spacing in the x-direction.

    dy : float
        Grid spacing in the y-direction.

    nu : float
        Kinematic viscosity [m²/s].

    Returns
    -------
    diff_uy : ndarray of shape (Ny, Nx)
        Diffusion contribution ν∇²v evaluated at all interior
        uy-control volumes.

    Notes
    -----
    - Second-order accurate in space.
    - Uses a five-point Laplacian stencil.
    - Boundary values are not modified.
    - Used in the predictor step before pressure correction.
    """

    diff_uy = np.zeros_like(uy)

    diff_uy[1:-1, 1:-1] = nu * (
        (uy[1:-1, 2:] - 2*uy[1:-1, 1:-1] + uy[1:-1, :-2]) / dx**2
      + (uy[2:, 1:-1] - 2*uy[1:-1, 1:-1] + uy[:-2, 1:-1]) / dy**2
    )

    return diff_uy