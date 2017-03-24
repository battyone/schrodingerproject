""" 
@Author: Robert Brown
This program solves the Schrodinger equation for given inital conditions using the finite difference method.
This uses Euler
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy
from scipy.integrate import simps


def prob(psi: int) -> numpy.ndarray:
    # returns the probability density of an wavefunction array
    return numpy.power(psi.real, 2) + numpy.power(psi.imag, 2)


def tridiagonal(x, n):
    # returns a tridiagonal matrix with numbers x1, x2, x3 as inputs
    a = (n-1) * [x[0],]
    b = n * [x[1],]
    c = (n-1) * [x[2]]
    return numpy.diag(a, k=-1) + numpy.diag(b, k=0) + numpy.diag(c, k=1)


def pentdiagonal(x, n):
    # returns a pent-diagonal matrix with numbers x1, x2, x3 as inputs
    return sum([numpy.diag((n + i) * [x[j], ], k=(-2 + j)) for i, j in [(-2, 0), (-1, 1), (0, 2), (-1, 3), (-2, 4)]])


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
        print("start")
        self.dx = dx
        self.dx2 = dx ** 2
        self.dt = dt
        self.mu = 0.5j*dt/(dx**2)
        self.nx = int(round((xrange[1] - xrange[0])/(dx))) + 1  # number of space steps
        self.nt = int(round((trange[1] - trange[0])/(dt))) + 1  # number of time steps
        self.trange = trange
        self.xrange = xrange
        self.I = numpy.empty(self.nt)
        self.psi  = numpy.empty([self.nt, self.nx], dtype=complex) # holds wavefunction for all times
        self.prob = numpy.empty([self.nt, self.nx], dtype=float)   # holds probability density function for all times
        self.shape = self.psi.shape            # shape of wavefunction array
        self.x = numpy.linspace(xrange[0], xrange[1], self.nx)
        self.t = numpy.linspace(trange[0], trange[1], self.nt)
        self.psi[0] = initial(self.x) # generate the initial wavefunction
        self.periodic = periodic      # periodic boundary condition
        self.order = order            # central difference order
        self.boundaries = boundaries
        print("finish")

        if potential is not None:
            self.potential = potential(self.x)
            # potential at each point
            # TODO: allow for a time varying potential
        else:
            # no potential
            self.potential = None
            
        if self.boundaries is not None and not self.periodic:
            print("setting bcs")
            self._setbcs(0)
        if exact is not None:
            self.deriv = exact

    def __getitem__(self, key):
        return self.real[key] + 1j * self.imag[key]

    def _setbcs(self, key):
        # make wavefunction satisfy BCs
        if not self.periodic:
            self.psi[key, 0]  = self.boundaries[0]
            self.psi[key, -1] = self.boundaries[1]


    def solve(self):
        #set up triadiagonal matrix
        if self.order==2:
            A = tridiagonal([-self.mu, 1 - 2 * self.mu, -self.mu], self.nx)
        if self.order==4:
            A = pentdiagonal([-self.mu/12, 4*self.mu/3, 1 - 5*self.mu/2, 4*self.mu/3, -self.mu/12], self.nx)
        
        if self.periodic:
            # set periodic BCs
            if self.order==2:
                print("periodic!")
                A[-1, 0] = -self.mu
                A[ 0,-1] = -self.mu
                
            if self.order ==4:
                print("periodic!")
                A[-1,0], A[0,-1] = -4*self.mu/3,  -4*self.mu/3
                A[-2,0], A[0,-2] = -1*self.mu/12, -1*self.mu/12
        
        if self.potential is not None:
            A = A + numpy.diag(self.potential)
        else:
            print("no potential!")
        B = numpy.conjugate(A)
        for i in range(self.nt-1):
            # crank-nicholson algorithm
            self._setbcs(i)
            self.psi[i+1] = numpy.linalg.solve(A,numpy.dot(B,self.psi[i]))
            self.I[i] = simps(prob(self.psi[i]), dx=self.dx)

        # calculate probability density
        self.prob=prob(self.psi)