import numpy as np


# ==========================================================
# Lamb-Oseen tangential velocity
# ==========================================================

def analytical_velocity(
    r,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Lamb-Oseen tangential velocity.

    Parameters
    ----------
    r : ndarray
        Radius from vortex centre

    Gamma : float
        Circulation

    rc : float
        Core radius

    Returns
    -------
    u_theta : ndarray
    """

    r = np.asarray(r)

    u_theta = np.zeros_like(r)

    mask = r > 1.0e-12

    rc_t = np.sqrt(rc**2 + 4.0 * nu * t)

    u_theta[mask] = (
        Gamma
        /
        (2.0 * np.pi * r[mask])
        *
        (
            1.0
            -
            np.exp(-(r[mask]**2)/(rc_t**2))
        )
    )

    return u_theta


# ==========================================================
# Cartesian velocity
# ==========================================================

def analytical_cartesian_velocity(
    x,
    y,
    xc,
    yc,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Analytical ux and uy.

    Returns
    -------
    ux
    uy
    """

    rx = x - xc
    ry = y - yc

    r = np.sqrt(rx**2 + ry**2)

    u_theta = analytical_velocity(
    r,
    Gamma,
    rc,
    nu,
    t
)

    ux = np.zeros_like(r)
    uy = np.zeros_like(r)

    mask = r > 1.0e-12

    ux[mask] = (
        -u_theta[mask]
        *
        ry[mask]
        /
        r[mask]
    )

    uy[mask] = (
        u_theta[mask]
        *
        rx[mask]
        /
        r[mask]
    )

    return ux, uy


# ==========================================================
# Analytical vorticity
# ==========================================================

def analytical_vorticity(
    r,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Lamb-Oseen vorticity.

    Returns
    -------
    omega
    """

    rc_t = np.sqrt(rc**2 + 4.0*nu*t)

    omega = (
        Gamma
        /
        (np.pi * rc_t**2)
    ) * np.exp(
        -(r**2)/(rc_t**2)
    )

    return omega


# ==========================================================
# Analytical pressure
# ==========================================================

def analytical_pressure(
    r,
    rho=1.0,
    Gamma=1.0,
    rc=0.1,
    Nr=2000
):
    """
    Pressure obtained from

        dp/dr = rho*u_theta²/r

    Returns pressure interpolated at r.
    """

    rmax = np.max(r) + rc

    rtab = np.linspace(
        1.0e-6,
        rmax,
        Nr
    )

    u_theta = analytical_velocity(
        rtab,
        Gamma,
        rc
    )

    integrand = (
        u_theta**2
        /
        rtab
    )

    dr = rtab[1] - rtab[0]

    pressure = np.zeros_like(rtab)

    for k in range(Nr-2, -1, -1):

        pressure[k] = (
            pressure[k+1]
            -
            rho
            *
            integrand[k]
            *
            dr
        )

    return np.interp(
        r,
        rtab,
        pressure
    )


# ==========================================================
# Horizontal centreline
# ==========================================================

def sample_horizontal_centerline(
    x,
    Lx,
    Ly,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Analytical uy along y = Ly/2.
    """

    xc = Lx/2
    yc = Ly/2

    y = np.full_like(
        x,
        yc
    )

    _, uy = analytical_cartesian_velocity(
        x,
        y,
        xc,
        yc,
        Gamma,
        rc,
        nu,
        t
    )

    return uy


# ==========================================================
# Vertical centreline
# ==========================================================

def sample_vertical_centerline(
    y,
    Lx,
    Ly,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Analytical ux along x = Lx/2.
    """

    xc = Lx/2
    yc = Ly/2

    x = np.full_like(
        y,
        xc
    )

    ux, _ = analytical_cartesian_velocity(
        x,
        y,
        xc,
        yc,
        Gamma,
        rc,
        nu,
        t
    )

    return ux


# ==========================================================
# Velocity profile
# ==========================================================

def velocity_profile(
    rmax,
    npoints=500,
    Gamma=1.0,
    rc=0.1
):
    """
    Smooth analytical profile.

    Returns
    -------
    r
    u_theta
    """

    r = np.linspace(
        0.0,
        rmax,
        npoints
    )

    u = analytical_velocity(
        r,
        Gamma,
        rc
    )

    return r, u

# ==========================================================
# Horizontal vorticity centreline
# ==========================================================

def sample_horizontal_vorticity(
    x,
    Lx,
    Ly,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Analytical vorticity along y = Ly/2.
    """

    xc = Lx / 2
    yc = Ly / 2

    y = np.full_like(x, yc)

    r = np.sqrt((x - xc)**2 + (y - yc)**2)

    omega = analytical_vorticity(
        r,
        Gamma,
        rc,
        nu,
        t
    )

    return omega

# ==========================================================
# Vertical vorticity centreline
# ==========================================================

def sample_vertical_vorticity(
    y,
    Lx,
    Ly,
    Gamma=1.0,
    rc=0.1,
    nu=0.001,
    t=0.0
):
    """
    Analytical vorticity along x = Lx/2.
    """

    xc = Lx / 2
    yc = Ly / 2

    x = np.full_like(y, xc)

    r = np.sqrt((x - xc)**2 + (y - yc)**2)

    omega = analytical_vorticity(
        r,
        Gamma,
        rc,
        nu,
        t
    )

    return omega