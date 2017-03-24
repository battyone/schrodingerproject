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
    # return np.exp(2j*x*np.pi)
    # return np.sqrt(2) * np.sin(x*np.pi)
    return np.exp(-(x-0.3)**2/0.1**2)

def exact_d(x):
    # exact derivative of the particle in a box stationary state
    return - (np.pi ** 2) * x

def V(x):
    return np.zeros_like(x) # free particle potential

t_int = 0.00003
t_rag = (0,1)
save = False
foo = qm.Wavefunction(initial, potential=V, dx=0.05, dt=t_int, trange=t_rag, order=2, boundaries=(0,0))
print("done!")
foo.solve()
print("done2!")
    
fig, ax = plt.subplots(2, sharex=False)
time = ax[0].text(.7, .5, '', fontsize=15)
line1, = ax[0].plot([], [], linewidth=3, color='b')
line2, = ax[0].plot([], [], linewidth=3, color='r')
line3, = ax[1].plot([], [], linewidth=2, color='k')

ax[0].set_xlabel("Position, $x$")
ax[0].set_ylabel("Wavefunction,  $\\Psi(x, t)$")
ax[0].set_ylim([-1.5,1.5])

ax[1].set_xlabel("Time, $t$")
ax[1].set_ylabel("$\\langle \\Psi | \Psi \\rangle$")
ax[0].set_xlim([0, 1])
ax[1].set_ylim([0, 2])
ax[1].set_xlim(t_rag)

# line1.set_data(foo.x, 0.007* foo.deriv(foo.real[0]))
# line2.set_data(foo.x, foo.real[0])
# plt.show()

def init():
    line1.set_data(foo.x,foo.psi[0].real)
    line2.set_data(foo.x,foo.psi[0].imag)
    line3.set_data(foo.t[0], foo.I[0])
    time.set_text('')

    return line1, line2, line3, time

def animate(i):
    line1.set_ydata(foo.psi[i].real)  # update the data
    line2.set_ydata(foo.psi[i].imag)
    line3.set_data(foo.t[:i], foo.I[:i])
    time.set_text("t = {0:.4f}".format(i*t_int))
    return line1, line2, line3, time

nstart = int(round((0.015/t_int)))

ani = animation.FuncAnimation(fig,
                              animate,
                              np.arange(0, foo.nt,1),
                              interval=10,
                              blit=True,
                              init_func=init)
if save:
    ani.save("4th order.mp4", writer="ffmpeg", codec='libx264', bitrate=-1, fps=24)

plt.show()