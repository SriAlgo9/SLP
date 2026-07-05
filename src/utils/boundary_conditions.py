
def velocity_bc(ux, uy):

    # ---------------------------------
    # ux : (Ny, Nx-1)
    # stored on vertical faces
    # ---------------------------------

    # Left wall
    ux[:,0] = 0.0

    # Right wall
    ux[:,-1] = 0.0

    # Bottom wall
    ux[0,:] = 0.0

    # Top wall
    ux[-1,:] = 0.0


    # ---------------------------------
    # uy : (Ny-1, Nx)
    # stored on horizontal faces
    # ---------------------------------

    # Bottom wall
    uy[0,:] = 0.0

    # Top wall
    uy[-1,:] = 0.0

    # Left wall
    uy[:,0] = 0.0

    # Right wall
    uy[:,-1] = 0.0

    return ux, uy


def pressure_bc(p):

    # ---------------------------------
    # p : (Ny, Nx)
    # stored at cell centres
    # Zero normal gradient
    # ---------------------------------

    p[:,0]  = p[:,1]
    p[:,-1] = p[:,-2]

    p[0,:]  = p[1,:]
    p[-1,:] = p[-2,:]

    return p

'''def velocity_bc(ux, uy):

    ux[0,:] = 0
    ux[-1,:] = 0
    ux[:,0] = 0
    ux[:,-1] = 0

    uy[0,:] = 0
    uy[-1,:] = 0
    uy[:,0] = 0
    uy[:,-1] = 0

    return ux, uy

def pressure_bc(p):

    p[:,0] = p[:,1]
    p[:,-1] = p[:,-2]

    p[0,:] = p[1,:]
    p[-1,:] = p[-2,:]

    return p'''