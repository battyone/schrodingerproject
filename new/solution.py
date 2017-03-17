""" 
@Author: Robert Brown

This program solves the Schrodinger equation for given inital conditions using the finite difference method.
This uses Euler
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import qm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
        
        
def initial(x):
    # return 0.3*np.exp(5j*x*np.pi)
    # return 0.5 * np.sin(5*x*np.pi)
    return np.exp(-(x-0.3)**2/0.1**2)

t_int = 0.000001
foo = qm.Wavefunction(initial, dt=t_int, trange=(0,0.2))
foo.solve()
    
fig, ax = plt.subplots()
time = ax.text(.7, .5, '', fontsize=15)
line1, = ax.plot([], [], linewidth=3, color='b')
line2, = ax.plot([], [], linewidth=3, color='r')


ax.set_xlabel("Position, $x$")
ax.set_ylabel("Wavefunction,  $\\Psi(x, t)$")
ax.set_ylim([-.8,.8])

def init():
    line1.set_data(foo.x,foo.real[0])
    line2.set_data(foo.x,foo.imag[0])
    time.set_text('')
    ax.set_xlim([0,1])
    return line1, line2, time

def animate(i):
    line1.set_ydata(foo.real[i])  # update the data
    line2.set_ydata(foo.imag[i]) 
    time.set_text("t = {0:.4f}".format(t_int*i))
    return line1, line2

    
ani = animation.FuncAnimation(fig,
                              animate,
                              np.arange(0, foo.nt,30),
                              interval=10, 
                              blit=False, 
                              init_func=init)
plt.show()