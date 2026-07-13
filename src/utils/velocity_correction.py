import numpy as np


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
