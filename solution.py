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
import potential


def initial(x):
    # return np.sqrt(2)*np.exp(3j*x*2*np.pi/1.02)
    # return np.sqrt(2) * np.sin(5*x*np.pi) # + np.sin(3*x*np.pi))
    # return 0.2 * np.exp(-np.power(x, 2)/0.1**2)  # approximate stationary state for harmonic potential
    # return 2 * np.exp(-np.power(x, 2) / 0.16 ** 2)  * np.exp(4j*x*2*np.pi/1.02)
    return np.exp(-x**2/0.01)

save = False

# foo = qm.Wavefunction(initial, dt=t_int, trange=(0,1), order=4, potential=potential.harmonic)
foo = qm.Wavefunction(initial,
                      dt=0.001,
                      xrange=(-1,1),
                      trange=(0,1),
                      order=2,
                      periodic=False,
                      # potential=potential.harmonic
                      )
# foo.exact_solve()
foo.solve()

# Plot the results
fig, ax = plt.subplots(3)
time = ax[0].text(.7, .5, '', fontsize=15)
line1, = ax[0].plot([], [], linewidth=3, color='b')
line2, = ax[0].plot([], [], linewidth=3, color='r')
linep, = ax[0].plot([], [], linewidth=1, color='k')
line3, = ax[1].plot([], [], linewidth=1, color='k')
line4, = ax[2].plot([], [], linewidth=1, color='k')

# wavefunction
ax[0].set_xlabel("Position, $x$")
ax[0].set_ylabel("Wavefunction,  $\\Psi(x, t)$")
ax[0].set_ylim([-2,2])
ax[0].set_xlim(foo.xrange)

# probability density function
ax[1].set_xlabel("Position, $x$")
ax[1].set_ylabel("Probability density,  $|\\Psi(x, t)|^2$")
ax[1].set_ylim([0,6])
ax[1].set_xlim(foo.xrange)

# integral over space
ax[2].set_xlabel("Time, $t$")
ax[2].set_ylabel("$\\langle \\Psi | \\Psi \\rangle$")
ax[2].set_xlim(foo.trange)
ax[2].set_ylim([0,1.5])

def init():
    line1.set_data(foo.x,foo.psi[0].real)
    line2.set_data(foo.x,foo.psi[0].imag)
    linep.set_data(foo.x, foo.potential)
    line3.set_data(foo.x,foo.prob[0])
    line4.set_data(foo.t[0],foo.I[0])
    time.set_text('')
    return line1, line2, line3, line4, time

def animate(i):
    line1.set_ydata(foo.psi[i].real)  # update the data
    line2.set_ydata(foo.psi[i].imag)
    line3.set_data(foo.x,foo.prob[i])
    line4.set_data(foo.t[:i],foo.I[:i])    
    time.set_text("t = {0:.4f}".format(foo.dt*i))
    return line1, line2, line3, line4, time

nstart = int(round((0.015/foo.dt)))

ani = animation.FuncAnimation(fig,
                              animate,
                              np.arange(0, foo.nt,1),
                              interval=20,
                              blit=True, 
                              init_func=init)
if save:
    ani.save("instability.mp4", writer="ffmpeg", codec='libx264', bitrate=-1, fps=24)
plt.show()