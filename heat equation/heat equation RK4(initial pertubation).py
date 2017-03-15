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
nt = 3000   # time steps
dt = 0.001
"""
Try setting dt=0.003 and see what happens.
The boundary for stability in with these particular conditions is between dt = 0.001 and 0.002
RK4 is actually less stable than RK3 in this example!
"""

dx = 1/nx

x = np.linspace(0,1,nx)
u = np.empty([nt,nx])

def initial(x, k = 2):
    return 0

def d2(u):
    # returns ∂²u
    deriv = np.empty(len(u))
    for i in range(1,len(u)-1):
        deriv[i] = u[i + 1] - 2* u[i] + u[i - 1]
    return deriv
    

u[0] = np.vectorize(initial)(x) # set initial conditions
u[0,10] = 1
mu = dt/(dx**2)                 # Δt/Δx^2
k = np.empty([4,nx])

for i in range(0,nt-1):
    k[0] =  mu*d2(u[i])
    k[1] =  mu*d2(u[i] + (k[0]/2))
    k[2] =  mu*d2(u[i] + k[1]/2)
    k[3] =  mu*d2(u[i] + k[2])
    u[i+1] = u[i] + (k[0] + 2*(k[1] + k[2]) + k[3])/6

    
fig, ax = plt.subplots()
x = np.linspace(0,1,nx)
line, = ax.plot(x, np.sin(x), linewidth=3, color='r')
ax.set_xlabel("Position, $x$")
ax.set_ylabel("Temperature,  $u(x, t)$")

def animate(i):
    line.set_ydata(u[i])  # update the data
    return line, ax

def init():
    # runs once
    ax.set_ylim([0,1])
    line.set_ydata(np.ma.array(x, mask=True))

    return line,
    
ani = animation.FuncAnimation(fig, animate, np.arange(0, nt), interval=25, init_func=init, blit=False)
plt.show()