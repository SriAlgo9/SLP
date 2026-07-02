def velocity_bc(ux, uy):

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

    return p