""" 
@Author: Robert Brown

This program solves the heat equation for given inital conditions using the finite difference method.
This uses a naïve eulerian method.
u/t = ²u/x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


nx = 20     # space steps
nt = 3000   # time steps
dt = 0.0012
"""
Try setting dt=0.0013 and see what happens.
The boundary for stability in with these particular conditions is between dt = 0.0012 and 0.0013
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
    # returns ²u
    deriv = np.empty(len(u))
    for i in range(1,len(u)-1):
        deriv[i] = u[i + 1] - 2* u[i] + u[i - 1]
    return deriv
    

u[0] = np.vectorize(initial)(x) 
mu = dt/(dx**2)
for i in range(0,nt-1):
    u[i+1] = u[i] + mu*d2(u[i])
    
fig, ax = plt.subplots()
x = np.linspace(0,1,nx)
line, = ax.plot(x, u[0])
ax.set_ylim([0,1])

def animate(i):
    line.set_ydata(u[i])  # update the data
    return line,
    
ani = animation.FuncAnimation(fig, animate, np.arange(1, nt), interval=25, blit=False)
plt.show()