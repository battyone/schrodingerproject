import numpy as np
# function definitions


class WaveFunction():
    """class that defines a wavefunction.
    i will always be used as the index for x, and j as the index for t.
    """
    def __init__(self, V, xrange=(0,1), trange=(0,1), dx=0.01, dt=0.005):
        # class constructor,
        self.dx = dx                                          # x step size
        self.dt = dt                                          # t half-step size
        self.nx = round((xrange[1] - xrange[0])/ dx)          # number of spatial points
        self.nt = round((trange[1] - trange[0])/ dt)          # number of time points
        self.psi = np.empty([self.nx, self.nt])               # matrix to hold wavefunction
        self.imax = self.nx - 1                               # maximum i index
        self.jmax = self.nt - 1                               # maximum j index
        self.j = 0                                            # index indicating the most recently calculated time slice
        self.x = np.linspace[xrange[0], xrange[1], self.nx]   # vector of x coordinates of spatial points
        potential = np.vectorize(V)                           # vectorize V function to allow it to act on each element of x array
        self.V = potential(self.x)                            # calculate potential at each point xi
        """ in general potential may vary with time, 
        so this would need to be changed 
        for time varying potentials """
        
    def deriv_x2(self, i, j):
        # returns the second derivative of Ψ, ∂²Ψ/∂x² with respect to x at point x_i and time t_j
        if 0 > i > self.imax:
            return (self.psi[i + 1, j]  - 2*self.psi[i, j] + self.psi[i - 1, j] )/(self.dx**2)
        elif i == 0:  # different behaviour at endpoints due to no x_-1 or x_n+1
            return (self.psi[0,j] + self.psi[2,j] - 2*self.psi[1,j])/(self.dx**2)
        else:
            # i == imax
            return (self.psi[self.imax,j] + self.psi[self.imax-2,j] - 2*self.psi[self.imax-1,j])/( self.dx**2)
    
    def S(self, j):
        # returns an approximation of ∂Ψ/∂t at time t_j
        deriv = np.empty([self.nx,1])
        for i in range(self.nx):
            deriv[i] = 0.5j * self.deriv_x2(i, j) - 1j * self.V[i] * self.psi[i,j]

        return deriv

    def RK3(self, j):
        # adds the next set of points Ψ_i_j+2 using Runge-Kutta 3
        k1 = self.dt * self.S(j)
        k2 = self.dt * self.S(j+1)
        k3 = 


# potential
def V(x):
    # infinite square well potential
    return 0 if 0 < x < 1 else float("inf")


def initial(x):
    # initial state (guassian)
    return np.exp(-(x - 0.5)**2)


# boundary conditions
foo = WaveFunction(V)
foo.psi[0,:] = initial(foo[0,:])

