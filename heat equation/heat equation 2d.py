""" 
@Author: Robert Brown

This program solves the heat equation for given inital conditions using the finite difference method.
This uses Runge-Kutta 3.
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



nx = 20     # space steps
ny = 20
nt = 500   # time steps
dt = 0.00001

dx2, dy2 = (1/n**2 for n in (nx, ny))

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
xx, yy = np.meshgrid(x,y)

u = np.empty([nt,nx,ny])

def gaussian(x, y, x0=0.5, y0=0.5, σ=0.1, A=0.5):
    r2 = (x - x0)**2 + (y - y0)**2
    return A*np.exp(-r2/(2*σ**2))


def d2(f, i):
    return f[i+1] - 2*f[i] + f[i-1]


def laplacian(u, dx2, dy2):
    # returns ∇^2 u
    deriv = np.zeros(u.shape)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            deriv[i,j] = d2(u[:,i],i)/dx2 + d2(u[j,:],j)/dy2
    return deriv
    

u[0] = gaussian(xx, yy) # set initial conditions
k = np.empty([4,nx])

for i in range(nt-1):
    
    u[i + 1] = u[i] + dt*laplacian(u[i], dx2, dy2)
    u[i+1,:,0] = 0
    u[i+1,:,-1] = 0
    u[i+1,0,:] = 0
    u[i+1,-1,:] = 0
    
fig, ax = plt.subplots()
line = ax.matshow(u[0])

def animate(i):
    line.set_array(u[i])  # update the data
    return line, ax

    
ani = animation.FuncAnimation(fig, animate, np.arange(1, nt), interval=25, blit=False)
plt.show()
