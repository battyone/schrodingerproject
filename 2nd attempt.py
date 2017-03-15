""" Schrodinger equation method of lines solution """
import numpy as np
import matplotlib.pyplot as plt
# function definitions

def potential(x):
    # infinite square well potential
    # return float("inf")
    # return 0 if 0 < x < 1 else float("inf") # square well
    return 0    # free particle
    # return x**2 # harmonic potential


def initial(x):
    # initial state (guassian)
    var = 0.002
    return np.sin(x*np.pi)
    # return np.exp(1j*x)
    # return 1/np.sqrt(2*np.pi*var) * np.exp(-(x - 0.5)**2/(2*var))



def S(psi, V, dx2):
    # supposed to return an approximation of ∂Ψ/∂t
    x = np.empty([len(psi),], dtype=complex)
    for i in range(len(psi)):
        imax=len(psi)-1
        if i == 0:
            x[i] = 0.5j/dx2 * (psi[0] + psi[2] - 2*psi[1]) - 1j * V[i]
        elif i == imax:
            x[i] = 0.5j/dx2 * (psi[imax] + psi[imax-2] - 2*psi[imax-1])/dx2 - 1j * V[i]
        else:
            x[i] = 0.5j/dx2 * (psi[i-1] - 2*psi[i] + psi[i+1])/dx2 - 1j * V[i]
    return x


xrange = (0,1,20) # min x, max x, number of x points
trange = (0,0.01,10000) # min t, max t, number of time points
dx, dt = [(k[1] - k[0])/k[2] for k in (xrange, trange)] # time & space steps

x = np.linspace(xrange[0], xrange[1], xrange[2])  # vector of x coordinates of spatial points
vec_potential = np.vectorize(potential)           # vectorize V function to allow it to act on each element of x array
V = vec_potential(x)                              # calculate potential at each point xi

psi = np.empty([xrange[2],2], dtype=complex)      # array to store x points, contains 2 rows to store prior time level

# boundary conditions
psi[:,0] = initial(x)          # set initial wavefunction to gaussian
dx2 = complex(dx**2)
mu = complex(dt/dx2)
fig = plt.figure()
for j in range(1):     # iterate over every time
    # Runge-Kutta 4
    k1 = S(psi[:,0], V, dx2)
    k2 = S(psi[:,0] + k1/2, V, dx2)
    k3 = S(psi[:,0] + k2/2, V, dx2)
    k4 = S(psi[:,0] + k3, V, dx2)
    psi[:,0] = psi[:,0] +  mu*(k1 + 2 * k2 + 2 * k3 + k4)/6
    if (j % 100) == 0:
        prob = psi[:,0]*np.conj(psi[:,0])
        plt.plot(x,prob)

    
# plt.plot(x,np.real(psi[:,0]))
# plt.plot(x,np.imag(psi[:,0]))
plt.show()