
import numpy as np


def initialize_lamb_oseen(
        Nx,
        Ny,
        dx,
        dy,
        rho=1.0,
        Gamma=1.0,
        rc=0.1):

    # ==========================================================
    # Allocate variables (Staggered Grid)
    # ==========================================================

    # Pressure at cell centres
    p = np.zeros((Ny, Nx))

    # u velocity at vertical faces
    ux = np.zeros((Ny, Nx - 1))

    # v velocity at horizontal faces
    uy = np.zeros((Ny - 1, Nx))

    # ==========================================================
    # Domain
    # ==========================================================

    Lx = Nx * dx
    Ly = Ny * dy

    xc = Lx / 2.0
    yc = Ly / 2.0

    # ==========================================================
    # Initialize u velocity
    # Location : (i+1/2 , j)
    # ==========================================================

    for j in range(Ny):

        y = (j + 0.5) * dy

        for i in range(Nx - 1):

            x = (i + 1.0) * dx

            rx = x - xc
            ry = y - yc

            r = np.sqrt(rx**2 + ry**2)

            if r < 1.0e-12:
                continue

            u_theta = (
                Gamma
                / (2.0 * np.pi * r)
                * (1.0 - np.exp(-(r**2) / rc**2))
            )

            ux[j, i] = -u_theta * ry / r

    # ==========================================================
    # Initialize v velocity
    # Location : (i , j+1/2)
    # ==========================================================

    for j in range(Ny - 1):

        y = (j + 1.0) * dy

        for i in range(Nx):

            x = (i + 0.5) * dx

            rx = x - xc
            ry = y - yc

            r = np.sqrt(rx**2 + ry**2)

            if r < 1.0e-12:
                continue

            u_theta = (
                Gamma
                / (2.0 * np.pi * r)
                * (1.0 - np.exp(-(r**2) / rc**2))
            )

            uy[j, i] = u_theta * rx / r

    # ==========================================================
    # Pressure initialization
    # ==========================================================

    Rmax = np.sqrt(Lx**2 + Ly**2)

    Nr = 1000

    rtab = np.linspace(1e-6, Rmax, Nr)

    utheta_tab = (
        Gamma
        / (2.0 * np.pi * rtab)
        * (1.0 - np.exp(-(rtab**2) / rc**2))
    )

    integrand = utheta_tab**2 / rtab

    dr = rtab[1] - rtab[0]

    pressure_tab = np.zeros_like(rtab)

    for k in range(Nr - 2, -1, -1):

        pressure_tab[k] = (
            pressure_tab[k + 1]
            - rho * integrand[k] * dr
        )

    # ==========================================================
    # Pressure at cell centres
    # ==========================================================

    for j in range(Ny):

        y = (j + 0.5) * dy

        for i in range(Nx):

            x = (i + 0.5) * dx

            r = np.sqrt(
                (x - xc)**2 +
                (y - yc)**2
            )

            p[j, i] = np.interp(
                r,
                rtab,
                pressure_tab
            )

    return ux, uy, p













'''import numpy as np

def initialize_lamb_oseen(
        Nx, Ny,
        dx, dy,
        rho=1.0,
        Gamma=1.0,
        rc=0.1):
    


    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    p  = np.zeros((Ny, Nx))


    Lx = (Nx-1)*dx
    Ly = (Ny-1)*dy

    xc = Lx/2.0
    yc = Ly/2.0

    # -------------------------
    # Velocity field
    # -------------------------

    for j in range(Ny):
    for i in range(Nx):

        x = i*dx
        y = j*dy

        rx = x - xc
        ry = y - yc

        r = np.sqrt(rx**2 + ry**2)

        if r < 1e-12:
            continue

        u_theta = (
            Gamma/(2*np.pi*r)
            *
            (1.0 - np.exp(-(r**2)/(rc**2)))
        )

        ux[j,i] = -u_theta*ry/r
        uy[j,i] =  u_theta*rx/r

    # -------------------------
    # Pressure field
    # -------------------------

    Rmax = np.sqrt(Lx**2 + Ly**2)

    Nr = 1000
    rtab = np.linspace(1e-6, Rmax, Nr)

    utheta_tab = (
        Gamma/(2*np.pi*rtab)
        *
        (1.0 - np.exp(-(rtab**2)/(rc**2)))
    )

    integrand = utheta_tab**2 / rtab

    dr = rtab[1] - rtab[0]

    pressure_tab = np.zeros_like(rtab)

    for k in range(Nr-2, -1, -1):
        pressure_tab[k] = (
            pressure_tab[k+1]
            - rho*integrand[k]*dr
        )

    for j in range(Ny):
        for i in range(Nx):

            x = i*dx
            y = j*dy

            r = np.sqrt(
                (x-xc)**2 +
                (y-yc)**2
            )

            p[j,i] = np.interp(
                r,
                rtab,
                pressure_tab
            )

    return ux, uy, p

'''