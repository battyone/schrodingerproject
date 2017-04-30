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

from time import perf_counter


def initial(x):
    return np.sqrt(2)*np.exp((3j*2*np.pi/2.02)*x)
    # return np.sqrt(2) * np.sin(5*x*np.pi/2.02) # + np.sin(3*x*np.pi))
    # return 0.2 * np.exp(-np.power(x, 2)/0.1**2)  # approximate stationary state for harmonic potential
    # return 2 * np.exp(-np.power(x+2, 2) / 0.1 ** 2)  * np.exp(-4j*x*2*np.pi/1.02)
    # return np.exp(-x**2/0.01)

from cycler import cycler
color_c = cycler('color', ['k'])
style_c = cycler('linestyle', ['-', '--', ':', '-.'])
markr_c = cycler('marker', [''])
c_cms = color_c * markr_c * style_c
c_csm = color_c * style_c * markr_c

plt.rc('text', usetex=True)
plt.rc('axes', prop_cycle=c_cms)
plt.rc('font', family='serif')
fig = plt.figure()
ax = fig.add_subplot(111)

lines = []
for delta_t in (0.002,0.001,0.0005,0.00001):
    foo = qm.Wavefunction(initial,
                          dt=delta_t,
                          xrange=(-1,1),
                          trange=(0,.2),
                          order=2,
                          periodic=True,
                          # eigenvalue= -(5*np.pi/2.02)**2,
                          eigenvalue= (3j * 2 * np.pi / 2.02)**2,
                          potential=potential.free
                          )

    foo.exact_solve()
    start = perf_counter()
    foo.solve()
    end = perf_counter()
    e = foo.psi - foo.exact_psi
    E = np.amax(np.abs(e), axis=1)
    lines.append(ax.plot(foo.t, E, label="{}".format(delta_t))[0])

    # ...

    print(end - start)

ax.set_xlim([0,0.2])
plt.legend(handles=lines, title=r'$\Delta t$')
plt.ylabel(r'Maximum error $E_n $')
plt.xlabel(r'Time elapsed, $t_n$')
plt.show()