""" 
@Author: Robert Brown, Changkai Zhang
This program solves the Schrodinger equation for given inital conditions using the finite difference method.
This uses Crank-Nicholson
∂u/∂t = ∂²u/∂x²
"""

import numpy
from scipy.integrate import simps
from scipy.linalg import fractional_matrix_power
import solve

def _test(x, n=5):
    # test initial for testing qm.py
    return numpy.sqrt(2) * numpy.sin(n*x*numpy.pi)


def prob(psi):
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
                 eigenvalue = None
                 ):
        # class initializer
        self.dx = dx
        self.dx2 = dx ** 2
        self.dt = dt
        self.mu = 0.5j*dt/(dx**2)
        self.nx = int(round((xrange[1] - xrange[0])/(dx))) + 1  # number of space steps
        self.nt = int(round((trange[1] - trange[0])/(dt))) + 1  # number of time steps
        self.trange = trange
        self.xrange = xrange
        self.I = numpy.empty(self.nt) # <psi|psi>
        self.e = numpy.empty(self.nt) # error
        self.psi  = numpy.empty([self.nt, self.nx], dtype=numpy.complex128) # holds wavefunction for all times
        self.prob = numpy.empty([self.nt, self.nx], dtype=numpy.float)   # holds probability density function for all times
        self.shape = self.psi.shape            # shape of wavefunction array
        self.x = numpy.linspace(xrange[0], xrange[1], self.nx)
        self.t = numpy.linspace(trange[0], trange[1], self.nt)
        self.psi[0] = initial(self.x) # generate the initial wavefunction
        self.periodic = periodic      # periodic boundary condition
        self.eigenvalue = eigenvalue
        self.order = order            # central difference order
        self.boundaries = boundaries

        if potential is not None:
            self.potential = potential(self.x)
            # potential at each point
            # TODO: allow for a time varying potential
        else:
            # no potential
            self.potential = None
            
        if self.boundaries is not None and not self.periodic:
            self._setbcs(0)
        if exact is not None:
            self.deriv = exact

    def _setbcs(self, key):
        # make wavefunction satisfy BCs
        if not self.periodic:
            self.psi[key, 0]  = self.boundaries[0]
            self.psi[key, -1] = self.boundaries[1]

    def exact_solve(self):
        # used for solving stationary states exactly to determine error in the numeric solution
        self.exact_psi = numpy.array([numpy.exp(1j * self.eigenvalue * time) * self.psi[0] for time in self.t])
        self.exact_prob = prob(self.exact_psi)
        self.exact_I = simps(self.exact_psi, dx=self.dx)

    def solve(self):
        #set up triadiagonal matrix
        if self.order==2:
            # print("order 2")
            A = tridiagonal([self.mu, 1 - 2 * self.mu, self.mu], self.nx)
        if self.order==4:
            # print("order 4")
            A = pentdiagonal([-self.mu/12, 4*self.mu/3, 1 - 5*self.mu/2, 4*self.mu/3, -self.mu/12], self.nx)
        
        if self.periodic:
            # set periodic BCs
            if self.order==2:
                # print("periodic!")
                A[-1, 0] = self.mu
                A[ 0,-1] = self.mu
                
            if self.order ==4:
                # print("periodic!")
                A[0,-1], A[0,-2] =  4*self.mu/3, -self.mu/12
                A[-1,0], A[-1, 1] = 4*self.mu/3, -self.mu/12
                # A[-1, 0], A[-1, 1] = self.mu, 2*self.mu

        else:
            # set dirichlet BCs
            A[0,   1] = 0
            A[-1, -2] = 0

        if self.potential is not None:
            A = A + self.mu * numpy.diag(self.potential)

        # print(numpy.around(10000* numpy.imag(A)))
        # B = numpy.conjugate(A)

        if ( numpy.all(self.potential == 0) or self.potential is None ) and self.order > 2:
            # enable optimization for free potential for high enough order derivative
            # print("zero potential!")
            B = fractional_matrix_power(A, numpy.e)
            C = numpy.linalg.inv(B)
        else:
            B = numpy.conj(A)
            C = numpy.dot(numpy.linalg.inv(A),B)
        
        """
        for i in range(self.nt - 1):
            self.psi[i + 1] = numpy.dot(C, self.psi[i])
        """
        
        self.psi  = solve.solve(self.psi, C, self.nt)  # solve wavefunction for all time
        self.prob = prob(self.psi)                       # calculate the probability density
        self.I    = simps( self.prob, dx=self.dx)        # calculate the integrated probability density

        if self.eigenvalue is not None:
            self.e = simps( numpy.power(self.prob - self.exact_prob, 2), dx=self.dx)  # calculate the error in the solution

        # print("finished solving {} points for {} time instances".format(self.nx, self.nt))
