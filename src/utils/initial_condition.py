
import numpy as np

'''Create the initial velocity and pressure fields 
corresponding to a vortex located at the centre of your computational domain.'''
def initialize_lamb_oseen(
        Nx,
        Ny,
        dx,
        dy,
        rho=1.0,
        Gamma=1.0,
        rc=0.025):

    

    # ============================================================
    # Staggered-grid variables
    # ============================================================

    # Pressure p is stored at grid points / cell centers.
    # Shape = (Ny, Nx)
    # p[j, i] represents pressure at location (x_i, y_j).
    p = np.zeros((Ny, Nx))


    # u-velocity is stored at vertical faces.
    # Shape = (Ny, Nx-1)
    # ux[j, i] represents the x-direction velocity at a
    # vertical face between pressure points i and i+1.
    ux = np.zeros((Ny, Nx - 1))


    # v-velocity is stored at horizontal faces.
    # Shape = (Ny-1, Nx)
    # uy[j, i] represents the y-direction velocity at a
    # horizontal face between pressure points j and j+1.
    uy = np.zeros((Ny - 1, Nx))

                    #  p[j,i+1]
                    #●
                    #|
                    #|  ux[j,i]
                    #→
                    #|
          #p[j,i]    ●

    # ==========================================================
    # Domain
    # ==========================================================

    '''# Calculate the physical domain dimensions.
    Nx and Ny are the number of grid points in the x- and y-directions.
    Since N grid points create N-1 intervals, the domain lengths are:
    Nx = number of grid points in the x-direction
    Ny = number of grid points in the y-direction
    Δx = grid spacing in the x-direction
    Δy = grid spacing in the y-direction
    Lx = physical length of the domain in x
    Ly = physical length of the domain in y
    Number of intervals = Number of points - 1'''
    Lx = (Nx - 1) * dx
    Ly = (Ny - 1) * dy

    xc = Lx / 2.0 # x-coordinate of the center
    yc = Ly / 2.0 # y-coordinate of the center

    # ==========================================================
    # Initialize u velocity
    # Location : (i+1/2 , j)
    # ==========================================================

    for j in range(Ny):

        #y = (j + 0.5) * dy
        y = j * dy

        for i in range(Nx - 1):

            # because u velocity is located halfway between two pressure points.
            x = (i + 0.5) * dx 
            
            #rx and ry tell how far the current velocity location is from the vortex center.
            rx = x - xc
            ry = y - yc

            #It gives the radial distance from the vortex center.
            #we need r because your vortex is radially symmetric.
            #It depends on: How far am I from the vortex centre?
            r = np.sqrt(rx**2 + ry**2)

            '''If:r=0 then mathematically we would encounter 1/0
                Numerically, you don't want your code to calculate that.
                basically means:If I'm essentially at the vortex centre, 
                don't evaluate the 1/r expression.'''
            if r < 1.0e-12:
                continue

            '''Calculate tangential velocity
             This is the main Lamb–Oseen analytical formula:   
             I initialize a regularized Lamb–Oseen-type vortex using a Gaussian core radius rc.
             Γ is the circulation strength of the vortex. controls how strong the vortex is. 
             velocity strength increases with insrease of gamma value.
             rc controls how quickly the velocity transitions from 
             the centre of the vortex to the outer region.
                    Gaussian factor is:

                    1−exp(−r2/rc2)

                    If r≪rc:

                    exp(−r2/rc2)≈1

                    so:
                    
                    1−exp(−r2/rc2)≈0

                    Therefore the velocity doesn't blow up at the centre.

                    That's exactly why this regularization is usefu'''
            u_theta = (
                Gamma
                / (2.0 * np.pi * r)
                * (1.0 - np.exp(-(r**2) / rc**2))
            )

            #taking the tangential velocity of the circular vortex and 
            # resolving it into the Cartesian x-direction.
            # The tangential velocity is then converted into its x-component:
            ux[j, i] = -u_theta * ry / r

    # ==========================================================
    # Initialize v velocity
    # Location : (i , j+1/2)
    # ==========================================================

    for j in range(Ny - 1):

        y = (j + 0.5) * dy

        for i in range(Nx):

            x = i * dx

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

            '''                                     
              
                    y
                    ↑

              ● → → → ●
           ↗             ↘
         ●       ↻         ●
        ↑        vortex      ↓
         ●       center     ●
           ↖             ↙
              ● ← ← ← ●

                    └────────→ x
            
            The velocity is not pointing away from the centre.

            It goes around the centre.

            That's why it is called tangential velocity.

            The radial direction is:

                        er
            
            The tangential direction is:

                        eθ​

            Your code is constructing exactly that circular velocity field.'''

    # ==========================================================
    # Pressure initialization
    # ==========================================================
    # Maximum possible distance from the vortex center to a corner
    # of the rectangular computational domain.
    # It represents the diagonal length of the computational domain.
    Rmax = np.sqrt(Lx**2 + Ly**2)

    Nr = 1000 # means number of radial points.

    # rtab = array containing the radial positions 
    # np.linspace(a, b, N) creates N equally spaced values between a and b.
    # Create 1000 equally spaced radial positions from 10−6 to Rmax​.
    rtab = np.linspace(1e-6, Rmax, Nr)

    # Analytical Lamb-Oseen tangential velocity
    # Calculate the analytical tangential velocity u_theta(r)
    # at every radial location stored in rtab.
    utheta_tab = (
        Gamma
        / (2.0 * np.pi * rtab)
        * (1.0 - np.exp(-(rtab**2) / rc**2))
    )

    # this line takes every value of uθ, squares it, and divides it by the corresponding radius.
    #its called integrand Because later you will integrate this quantity with respect to r:
    integrand = utheta_tab**2 / rtab

    # Radial spacing between consecutive points in rtab.
    # rtab is uniformly spaced because it was created using np.linspace().
    dr = rtab[1] - rtab[0]

    # Array for storing the analytical pressure at each radial location.
    # Initially set to zero; the pressure values are calculated below.
    '''
    ==========================================================
    Analytical pressure profile
    ==========================================================

    Pressure in a circular vortex satisfies:

        dp/dr = rho * u_theta^2 / r

    The pressure is obtained by integrating this equation
    from the outer radius toward the vortex center.

    The outermost pressure is used as the reference:

        p(Rmax) = 0

    The discrete integration is:

        p[k] = p[k+1]
                - rho * (u_theta[k]^2 / r[k]) * dr

    Therefore, pressure is calculated backward from the
    outer radial location toward the vortex center.
    ==========================================================
    '''
    
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

            #I know the radial distance r of this grid point. 
            # What is the analytical pressure at this radius?
            '''Calculate the distance r of the current grid point from the vortex center,
            look up the corresponding pressure in the analytical radial pressure table 
            using linear interpolation, and store that pressure in the 2D pressure array
              at position [j,i].
              
              y(x) = y1 + [x-x1/x2-x1] (y2-y1)'''
            
            p[j, i] = np.interp(
                r, # where do we want the value?
                rtab, # known x-values
                pressure_tab # known y-values
            )

    return ux, uy, p
