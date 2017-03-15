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
dt = 0.002
"""
Try setting dt=0.003 and see what happens.
The boundary for stability in with these particular conditions is between dt = 0.002 and 0.003
"""

dx = 1/nx

x = np.linspace(0,1,nx)
u = np.empty([nt,nx])

def initial(x, k = 2):
    if x < 0.5:
        x = k*x
    else:
        x = k*(1-x)
    return x

def d2(u):
    # returns ∂²u
    deriv = np.zeros(len(u))
    for i in range(1,len(u)-1):
        deriv[i] = u[i + 1] - 2* u[i] + u[i - 1]
    return deriv
    

u[0] = np.vectorize(initial)(x) # set initial conditions
mu = dt/(dx**2)                 # Δt/Δx^2
k = np.empty([3,nx])

for i in range(0,nt-1):
    k[0] =  mu*d2(u[i])
    k[1] =  mu*d2(u[i] + (k[0]/2))
    k[2] =  mu*d2(u[i] + k[1]/2)
        
    u[i+1] = u[i] + (1/6)*k[0] + (2/3)*k[1] + (1/6)*k[2]
    
fig, ax = plt.subplots()
x = np.linspace(0,1,nx)
line, = ax.plot(x, u[0])
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
    
ani = animation.FuncAnimation(fig, animate, np.arange(1, nt), interval=25, init_func=init, blit=True)
plt.show()