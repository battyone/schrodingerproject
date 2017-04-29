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

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

def initial(x):
    # return np.sqrt(2)*np.exp(3j*x*2*np.pi/2.02)
    return np.sqrt(2) * np.sin(5*x*np.pi) # + np.sin(3*x*np.pi))
    # return 0.2 * np.exp(-np.power(x, 2)/0.1**2)  # approximate stationary state for harmonic potential
    # return 2 * np.exp(-np.power(x, 2) / 0.1 ** 2)  * np.exp(-4j*x*2*np.pi/1.02)
    # return np.exp(-x**2/0.01)

save = True
filename = "qho.mp4"

# foo = qm.Wavefunction(initial, dt=t_int, trange=(0,1), order=4, potential=potential.harmonic)
foo = qm.Wavefunction(initial,
                      dt=0.0005,
                      xrange=(-1,1),
                      trange=(0,0.2),
                      order=2,
                      periodic=False,
                      potential=potential.free
                      )
# foo.exact_solve()
foo.solve()

# Plot the results
fig, ax = plt.subplots(2)
time = ax[0].text(.0, .8, '', fontsize=15)
line1, = ax[0].plot([], [], linewidth=2, color='b', label=r'$\mathrm{Re}(\Psi)$')
line2, = ax[0].plot([], [], linewidth=2, color='r', label=r'$\mathrm{Im}(\Psi)$')
line1p, = ax[0].plot([], [], linewidth=1, color='k')
line2p, = ax[1].plot([], [], linewidth=1, color='k')
line3, = ax[1].plot([], [], linewidth=2, color='k')

# wavefunction
ax[0].set_ylabel("Wavefunction,  $\\Psi(x, t)$")
ax[0].set_ylim([-1.5,1.5])
ax[0].set_xlim(foo.xrange)

# probability density function
ax[1].set_xlabel("Position, $x$")
ax[1].set_ylabel("Probability density,  $|\\Psi(x, t)|^2$")
ax[1].set_ylim([0,2.2])
ax[1].set_xlim(foo.xrange)

line1p.set_data(foo.x, foo.potential)
line2p.set_data(foo.x, foo.potential)

# new range
# [ax[i].set_xlim([-5,5]) for i in range(0,2)]

def init():
    line1.set_data(foo.x,foo.psi[0].real)
    line2.set_data(foo.x,foo.psi[0].imag)
    line3.set_data(foo.x,foo.prob[0])
    time.set_text('')
    return line1, line2, line3, time

def animate(i):
    line1.set_ydata(foo.psi[i].real)  # update the data
    line2.set_ydata(foo.psi[i].imag)
    line3.set_data(foo.x, foo.prob[i])
    time.set_text(r'$t = ${0:.4f}'.format(foo.dt*i))
    return line1, line2, line3, time

nstart = int(round((0.015/foo.dt)))

ani = animation.FuncAnimation(fig,
                              animate,
                              np.arange(0, foo.nt,1),
                              interval=20,
                              blit=True, 
                              init_func=init)
if save:
    ani.save(filename, writer="ffmpeg", codec='libx264', bitrate=-1, fps=24)

ax[0].legend(handles=[line1, line2])
plt.show()