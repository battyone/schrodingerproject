import qm
import numpy as np
import numpy.random as rnd
from scipy import ndimage
import matplotlib.pyplot as plt

def initial(x):
    return np.sin(np.pi * x) # 0.02*np.sin(46*np.pi * x)

def deriv(x):
    return - np.pi ** 2 * np.sin(np.pi * x)

x_range = (0,1)
dx=0.1
foo1 = qm.Wavefunction(initial, dx=dx, xrange = x_range, dt=0.5, order=2)
foo2 = qm.Wavefunction(initial, dx=dx, xrange = x_range, dt=0.5, order=4)
# ggf = - ndimage.gaussian_filter1d(foo1.real[0], sigma=4.5, order=2, mode='wrap') /(dx**2)
x = np.linspace(*x_range, 50)
exact = deriv(x)

fig, ax = plt.subplots(2, sharex=True)
# ax[0].plot(x, initial(x))
# plt.show()
# derivative
o2 = foo1.deriv(foo1.real[0])
o2 = ndimage.gaussian_filter1d(o2, sigma=1, mode='wrap')
o4 = foo2.deriv(foo2.real[0])
o4 = ndimage.gaussian_filter1d(o4, sigma=2, mode='reflect')
ax[0].scatter(foo1.x, o2, c='r')
ax[0].scatter(foo2.x, o4, c='g')
ax[0].plot(x, exact)

# deviation in ordinates
ax[1].scatter(foo1.x, deriv(foo1.x) - o2, c='r')
ax[1].scatter(foo2.x, deriv(foo2.x) - o4, c='g')
ax[1].plot(x, 0*x)
plt.show()