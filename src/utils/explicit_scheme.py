import numpy as np
from src.utils.diffusion_term import diffusion_ux, diffusion_uy

# intermediate/predicted velocities in Chorin projection method.
def compute_ux_star(ux, uy, dt, dx, dy, nu):

    Ny, Nx = ux.shape 

    '''
    Create an independent copy of the current u-velocity.
    We update ux_star during the time step while keeping
    the original ux unchanged.
    
    IMPORTANT:
    ux_star = ux.copy() creates a new array.
    ux_star = ux would only create another reference to ux.
    '''
    ux_star = ux.copy() 
    diff_x = diffusion_ux(ux, dx, dy, nu)

    for j in range (1, Ny-1):
        for i in range (1, Nx-1):

            '''
            x-direction convection.
        
            Approximates the nonlinear conservative term:
        
                ∂(u²)/∂x
        
            using neighboring u-velocity values.
            '''
            convection_x = (
                (ux[j, i] + ux[j, i+1])**2 
                - (ux[j, i] + ux[j, i-1])**2
            )


            '''
            y-direction convection.
        
            Approximates the nonlinear conservative term:
        
                ∂(uv)/∂y
        
            using both u- and v-velocity values.'''
            convection_y = (
                (ux[j, i] + ux[j+1, i]) 
                * (uy[j, i] + uy[j, i+1])
                - (ux[j ,i] + ux[j-1, i]) 
                * (uy[j-1, i] + uy[j-1, i+1])
            ) 


            '''
            Explicit time update for the intermediate
            x-velocity u*.
        
            Current velocity
                - convection
                + diffusion
        
            The pressure-gradient correction is NOT included
            here; it is applied in the next projection step.'''
            ux_star[j, i] = (
                ux[j, i]
                - dt/(4*dx) * convection_x
                - dt/(4*dy) * convection_y
                + dt*diff_x[j, i]
            )
    return ux_star

def compute_uy_star(ux, uy, dt, dx, dy, nu):
    
    Ny, Nx = uy.shape
    uy_star = uy.copy()
    diff_y = diffusion_uy(uy, dx, dy, nu)

    for j in range (1, Ny-1): # iterating through rows (cols in grid)
        for i in range(1, Nx-1): # iterating through cols (rows in grid)

            convection_y = ( 
                (uy[j, i] + uy[j+1, i])**2 
                - (uy[j, i] + uy[j-1, i])**2
            )

            convection_x = (
                (uy[j, i] + uy[j, i+1]) 
                * (ux[j, i] + ux[j+1, i]) 
                - (uy[j, i] + uy[j, i-1]) 
                * (ux[j, i-1] + ux[j+1, i-1])
            )

            uy_star[j, i] = (
                uy[j, i]
                - dt/(4*dx) * convection_x
                - dt/(4*dy) * convection_y
                + dt*diff_y[j, i]
            )
    return uy_star
