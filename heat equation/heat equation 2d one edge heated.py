""" 
@Author: Robert Brown

This program solves the heat equation for given inital conditions using the finite difference method.
This uses naive euler's method.
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



nx = 50     # space steps
ny = 50
nt = 1500
   # time steps
dt = 0.0001

dx2, dy2 = (1/(n - 1)**2  for n in (nx, ny))

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
xx, yy = np.meshgrid(x, y)
u = np.empty([nt,nx,ny])


def gaussian(x, y, x0=0.5, y0=0.5, σ=0.1, A=0.5):
    r2 = (x - x0)**2 + (y - y0)**2
    return A*np.exp(-r2/(2*σ**2))


def d2(f, i):
    return f[i+1] - 2*f[i] + f[i-1]


def d(f, axis=0, dirichlet=True):
    # central difference
    if axis==0:   # x axis
        return f[2:,1:-1] - 2* f[1:-1,1:-1] + f[:-2,1:-1]
    elif axis==1: # y axis
        return f[1:-1,2:] - 2 * f[1:-1,1:-1] + f[1:-1,:-2]


def laplacian(u, dx2, dy2):
    # returns ∇^2 u
    deriv = np.zeros_like(u)
    nx, ny = u.shape
    deriv[1:-1,1:-1] = d(u, axis=0) / dx2 + d(u, axis=1) / dy2
    """
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            deriv[i,j] = d2(u[:,i],i)/dx2 + d2(u[j,:],j)/dy2
    """
    # dirichlet boundary conditions
    deriv[0,:]  = 0
    deriv[-1,:] = 0
    deriv[:,0]  = 0
    deriv[:,-1] = 0
    return deriv
    

u[0] = 0 # set initial conditions
# u[0] = gaussian(xx, yy)

k = np.empty([4,nx])

for i in range(nt-1):
    # dirichlet boundary conditions
    u[i,:,0] = 1
    u[i,:,-1] = 1
    u[i,-1,:] = 0
    u[i,0,:] = 0

    u[i + 1] = u[i] + dt*laplacian(u[i], dx2, dy2)

    
fig, ax = plt.subplots(2)
line = ax[0].matshow(u[0])
cont = ax[1].contour(u[0])

def animate(i):
    line.set_array(u[i])  # update the data
    ax[1].cla()
    ax[1].contour(u[i])
    return line,

ani = animation.FuncAnimation(fig,
                              animate,
                              np.arange(0, nt, 10),
                              interval=25,
                              blit=False
                              )
plt.show()
