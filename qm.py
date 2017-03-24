""" 
@Author: Robert Brown

This program solves the Schrodinger equation for given inital conditions using the finite difference method.
This uses Euler
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy
from scipy.integrate import simps, odeint

foward_diff = (469/90, -223/10, 879/20, -949/18, 41, -201/10, 1019/180, -7/10)  # 4th order accurate forward difference
backward_diff = foward_diff[::-1]

def prob(psi: int) -> numpy.ndarray:
    return psi.real ** 2 + psi.imag ** 2


def tridiagonal(x, n):
    # returns a tridiagonal matrix with numbers x1, x2, x3 as inputs
    a = (n-1) * [x[0],]
    b = n * [x[1],]
    c = (n-1) * [x[2]]
    return numpy.diag(a, k=-1) + numpy.diag(b, k=0) + numpy.diag(c, k=1)


def c_to_tup(x):
    # given a complex number x, returns a tuple of floats for the real and imag parts of x
    return x.real, x.imag

class Wavefunction:
    def __init__(self, initial,
                 potential=None,
                 dx=0.02,
                 dt=0.0001,
                 xrange=(0,1),
                 trange=(0,1),
                 order=2,
                 boundaries=(0+0j,0+0j),
                 exact = None,
                 periodic = False,
                 smooth = False
                    ):
        # class initializer
        self.dx = dx
        self.dx2 = dx ** 2
        self.dt = dt
        self.mu = 0.5j*dt/(dx**2)
        self.nx = int(round((xrange[1] - xrange[0])/(dx))) + 1  # number of space steps
        self.nt = int(round((trange[1] - trange[0])/(dt))) + 1  # number of time steps
        self.I = numpy.empty(self.nt)
        self.psi = numpy.empty([self.nt, self.nx], dtype=complex) # holds wavefunction for all times
        self.prob = numpy.empty([self.nt, self.nx], dtype=float)  # holds probability density function for all times
        self.shape = self.psi.shape            # shape of wavefunction array
        self.x = numpy.linspace(xrange[0], xrange[1], self.nx)
        self.t = numpy.linspace(trange[0], trange[1], self.nt)
        init = initial(self.x)                  # generate the initial wavefunction
        self.psi[0] = init
        self.periodic = periodic  # periodic boundary condition
        self.order = order  # central difference order
        self.boundaries = boundaries
        self.smooth = smooth

        if potential is not None:
            self.potential = potential(self.x)
            # potential at each point
            # TODO: allow for a time varying potential
        if self.boundaries is not None:
            self._setbcs(0)
        if exact is not None:
            self.deriv = exact

    def __getitem__(self, key):
        return self.real[key] + 1j * self.imag[key]

    def _setbcs(self, key):
        # make wavefunction satisfy BCs
            self.psi[key, 0]  = self.boundaries[0]
            self.psi[key, -1] = self.boundaries[1]
        # self.real[key, 1], self.imag[key, 1]   = 0, 0 # DEBUG
        # self.real[key, -2], self.imag[key, -2] = 0, 0

    def deriv(self, psi):
        # returns the second spatial derivative of psi, ∂²ψ/∂x²
        # TODO: implement periodic boundary condition
        df = numpy.empty_like(psi)
        # df = ndimage.gaussian_filter1d(psi, sigma=4, order=2, mode={True:'wrap', False:'constant'}[self.periodic], cval=0) / self.dx2

        
        # central difference
        if self.order == 2:
            # 2nd order central difference for 2nd derivative
            df[1:-1] = psi[2:] - 2*psi[1:-1] + psi[0:-2]
            if self.periodic:
                df[0] =    psi[1] - 2*psi[0]    + psi[-2]
                df[-1] =   psi[1] - 2*psi[-1]   + psi[-2]

        elif self.order == 4:
            # 4th order central difference for 2nd derivative
            df[2:-2] = (-psi[:-4] + 16*psi[1:-3] - 30 * psi[2:-2] + 16*psi[3:-1] - psi[4:])/12
            if self.periodic:
                df[1] = (-psi[-2] + 16*psi[0] - 30 * psi[1] + 16*psi[2] - psi[3])/12
                df[0] = (-psi[-3] + 16 * psi[-2] - 30 * psi[0] + 16 * psi[1] - psi[2]) / 12
                df[-1] = (-psi[-3] + 16 * psi[-2] - 30 * psi[-1] + 16 * psi[1] - psi[2]) / 12
                df[-2] = (-psi[-4] + 16 * psi[-3] - 30 * psi[-2] + 16 * psi[-1] - psi[0]) / 12
            else:
                df[1] = (10*psi[0] - 15*psi[1] -4*psi[2] + 14*psi[3] - 6*psi[4] + psi[5])/12
                df[-2] = (10 * psi[-1] - 15 * psi[-2] - 4 * psi[-3] + 14 * psi[-4] - 6 * psi[-5] + psi[-6]) / 12

        if self.boundaries is not None:
            # set dirichlet boundary conditions (endpoints are constant -> deriv = 0)
            df[0] = 0
            df[-1] = 0
        elif not self.periodic:
            # try to estimate df at psi[0]. psi[-1]. This is usually quite unstable
            df[0] = -1 * psi[0] + 4 * psi[1] - 5*psi[2] + 2 * psi[3] # first point
            df[-1] = -1 * psi[-1] + 4 * psi[-2] - 5*psi[-3] + 2 * psi[-4] # last point

        return df / self.dx2 + self.potential

    def time_d(self, psi, type='r'):
        # returns the time derivative of the real or imaginary part of a wavefunction
        if type == 'r':
            # real part of wavefunction
            return - 0.5 * self.deriv(psi) + self.potential * psi
        if type =='i':
            # imaginary part of wavefunction
            return   0.5 * self.deriv(psi) - self.potential * psi

    def solve(self):
        #set up triadiagonal matrix
        for i in range(self.nt-1):
            self._setbcs(i)
            A = tridiagonal([-self.mu, 1 + 2*self.mu, -self.mu], self.nx)
            B = numpy.conjugate(A)
            C = numpy.dot(B,self.psi[i])
            self.psi[i+1] = numpy.linalg.solve(A,C)
            self.I[i] = simps(prob(self.psi[i]), dx=self.dx)

        self.prob=prob(self.psi)