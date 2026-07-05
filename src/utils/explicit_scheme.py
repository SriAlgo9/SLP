import numpy as np
from src.utils.diffusion_term import diffusion_ux, diffusion_uy
#ux = np. zeros ((Ny, Nx), dtype = np.float32)
#uy = np. zeros ((Ny, Nx), dtype = np.float32) 

#ux_star = np. zeros ((Ny, Nx), dtype = np.float32)
#uy_star = np. zeros ((Ny, Nx), dtype = np.float32)

def compute_ux_star(ux, uy, dt, dx, dy, nu):

    Ny, Nx = ux.shape 
    ux_star = ux.copy() # arrays pass reference. Thus creating a copy
    diff_x = diffusion_ux(ux, dx, dy, nu)

    for j in range (1, Ny-1):
        for i in range (1, Nx-1):
            
            convection_x = (
                (ux[j, i] + ux[j, i+1])**2 
                - (ux[j, i] + ux[j, i-1])**2
            )

            convection_y = (
                (ux[j, i] + ux[j+1, i]) 
                * (uy[j, i] + uy[j, i+1])
                - (ux[j ,i] + ux[j-1, i]) 
                * (uy[j-1, i] + uy[j-1, i+1])
            ) 

            ux_star[j, i] = (ux[j, i] - dt/(4*dx) * (convection_x + convection_y)+ dt*diff_x[j, i])

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

            uy_star[j, i] = (uy[j, i] - dt/(4*dy) * (convection_x + convection_y) + dt*diff_y[j, i] )
    return uy_star




#velocity correction
def compute_ux(ux_star, dt, dx, rho, p):
    
    Ny, Nx = ux_star.shape

    velocity_ux = ux_star.copy()
    
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):

         velocity_ux[j,i] = ux_star[j,i] - (dt/rho)*(p[j,i+1] - p[j,i])/(dx)
        #if we use  p[j,i+1] - p[j,i-1] it becomes a collocated grid
    return velocity_ux

def compute_uy(uy_star, dt, dy, rho, p):
   
    Ny,Nx = uy_star.shape

    velocity_uy = uy_star.copy()

    for j in range(1, Ny-1):
        for i in range (1, Nx-1):
            
            velocity_uy[j,i] = uy_star[j,i] - (dt/rho)*(p[j+1,i] - p[j,i])/(dy)
            #if we use p[j+1,i] - p[j-1,i] it becomes a collocated grid

    return velocity_uy





