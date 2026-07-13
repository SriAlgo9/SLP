import numpy as np



# ==========================================================
# Cell-centred velocity
# ==========================================================

def compute_velocity_magnitude(ux, uy):
    """
    Convert staggered velocities to cell centres and compute
    velocity magnitude.

    Parameters
    ----------
    ux : (Ny, Nx-1)
        x velocity on vertical faces

    uy : (Ny-1, Nx)
        y velocity on horizontal faces

    Returns
    -------
    u_center : (Ny-2, Nx-2)
    v_center : (Ny-2, Nx-2)
    speed    : (Ny-2, Nx-2)
    """

    u_center = 0.5 * (
        ux[1:-1, :-1] +
        ux[1:-1, 1:]
    )

    v_center = 0.5 * (
        uy[:-1, 1:-1] +
        uy[1:, 1:-1]
    )

    speed = np.sqrt(
        u_center**2 +
        v_center**2
    )

    return u_center, v_center, speed


# ==========================================================
# Divergence
# ==========================================================

def compute_divergence(ux, uy, dx, dy):
    """
    Compute divergence at cell centres.
    """

    div = (
        (ux[1:-1, 1:] - ux[1:-1, :-1]) / dx
        +
        (uy[1:, 1:-1] - uy[:-1, 1:-1]) / dy
    )

    return div


# ==========================================================
# Vorticity
# ==========================================================

def compute_vorticity(ux, uy, dx, dy):
    """
    Compute vorticity

        omega = dv/dx - du/dy

    at cell centres.
    """

    u_center, v_center, _ = compute_velocity_magnitude(ux, uy)

    dv_dx = (
        v_center[:, 2:] -
        v_center[:, :-2]
    ) / (2.0 * dx)

    du_dy = (
        u_center[2:, :] -
        u_center[:-2, :]
    ) / (2.0 * dy)

    omega = (
        dv_dx[1:-1, :]
        -
        du_dy[:, 1:-1]
    )

    return omega

# ==========================================================
# Vorticity centreline
# ==========================================================

def extract_vorticity_centerlines(omega, dx, dy):
    """
    Extract numerical vorticity along the horizontal and
    vertical centre lines.
    """

    Ny, Nx = omega.shape

    centre_y = Ny // 2
    centre_x = Nx // 2

    omega_horizontal = omega[centre_y, :]
    omega_vertical = omega[:, centre_x]

    x = (np.arange(Nx)+ 1) * dx
    y = (np.arange(Ny)+ 1) * dy

    return x, omega_horizontal, y, omega_vertical


# ==========================================================
# Horizontal / Vertical centreline
# ==========================================================

def extract_centerlines(ux, uy, dx, dy):
    """
    Extract numerical velocity along
    horizontal and vertical centre lines.
    """

    Ny, Nxm1 = ux.shape
    Nx = Nxm1 + 1

    # --------------------------------------
    # Horizontal centreline
    # --------------------------------------

    centre_y = Ny // 2

    uy_horizontal = 0.5 * (
        uy[centre_y - 1, 1:-1] +
        uy[centre_y,     1:-1]
    )

    x = np.arange(1, Nx-1) * dx

    # --------------------------------------
    # Vertical centreline
    # --------------------------------------


    centre_x = (Nx - 1) // 2

    ux_vertical = 0.5 * (
        ux[1:-1, centre_x - 1] +
        ux[1:-1, centre_x]
    )

    y = np.arange(1, Ny-1) * dy

    return x, uy_horizontal, y, ux_vertical


# ==========================================================
# Cell-centred grid
# ==========================================================

def create_cell_center_grid(
    Lx,
    Ly,
    dx,
    dy
):
    """
    Create coordinates at cell centres.
    """

    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dy) + 1

    xp = np.linspace(
        dx,
        Lx - dx,
        Nx - 2
    )

    yp = np.linspace(
        dy,
        Ly - dy,
        Ny - 2
    )

    Xp, Yp = np.meshgrid(
        xp,
        yp
    )

    return xp, yp, Xp, Yp

# ==========================================================
# L2 error
# ==========================================================

def compute_L2_error(numerical, analytical):
    """
    Relative L2 error.
    """

    error = numerical - analytical

    denominator = np.sqrt(
        np.sum(analytical**2)
    )

    if denominator < 1e-14:
        return np.nan

    return (
        np.sqrt(np.sum(error**2))
        /
        denominator
    )


# ==========================================================
# RMS error
# ==========================================================

def compute_RMS_error(numerical, analytical):
    """
    Root mean square error.
    """

    error = numerical - analytical

    return np.sqrt(
        np.mean(error**2)
    )


# ==========================================================
# Maximum error
# ==========================================================

def compute_Linf_error(numerical, analytical):
    """
    Maximum absolute error.
    """

    return np.max(
        np.abs(
            numerical -
            analytical
        )
    )