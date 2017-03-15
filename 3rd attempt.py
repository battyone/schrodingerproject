""" 
@Author: Robert Brown

This program solves the Scrodinger equation for given inital conditions using the finite difference method.
This uses Euler
∂u/∂t = ∂²u/∂x²
with bcs: u(0) = u(x_max) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def deriv(psi, dtype='none'):
    # returns ∂ψ at time jΔt
    nx = len(psi)
    
    dpsi = np.empty(nx)  # ∂²ϕ
    
    # calculate differnces 
    for i in range(1,nx-1):
        dpsi[i] = psi[i+1] - 2*psi[i] + psi[i-1]
    
    # set differnces at endpoints
    dpsi[0] = dpsi[1]
    dpsi[-1] = dpsi[-2]
    
    return {'real':-dpsi, 'imag':dpsi }[dtype]


class Wavefunction:
    def __init__(self, initial, 
                 dx=0.02,
                 dt=0.0001,
                 xrange=(0,1),
                 trange=(0,1),
                 boundaries=(0,0)
                    ):
        # class initilizer
        self.mu = dt/(2*dx**2)
        self.nx = round((xrange[1] - xrange[0])/(dx))  # number of space steps
        self.nt = round((trange[1] - trange[0])/(dt))  # number of time steps
        self.real = np.empty([self.nt,self.nx])
        self.imag = np.empty([self.nt,self.nx])
        self.shape = self.real.shape            # shape of wavefunction array
        self.x = np.linspace(xrange[0], xrange[1], self.nx)
        init = initial(self.x)                  # generate the initial wavefunction
        self.real[0] = np.real(init)
        self.imag[0] = np.imag(init)
        self.boundaries = boundaries        
        
    def solve(self):
        kreal = np.empty([3,self.nx])
        kimag = np.empty([3,self.nx])
        
        kreal[0] = kimag[0] = self.boundaries[0]
        kreal[-1] = kimag[-1] = self.boundaries[1]
        
        for i in range(0,self.nt-1):            
            kreal[0] = deriv(self.imag[i], dtype='real')
            kimag[0] = deriv(self.real[i], dtype='imag')
            
            kreal[1] = deriv(self.imag[i] + self.mu * kreal[0]/2, dtype='real')
            kimag[1] = deriv(self.real[i] + self.mu * kimag[0]/2, dtype='imag')
            
            kreal[2] = deriv(self.imag[i] + self.mu * kreal[1]/2, dtype='real')
            kimag[2] = deriv(self.real[i] + self.mu * kimag[1]/2, dtype='imag')

            self.real[i+1] = self.real[i] + self.mu * ((1/6)*kreal[0] + (2/3)*kreal[1] + (1/6)*kreal[2])
            self.imag[i+1] = self.imag[i] + self.mu * ((1/6)*kimag[0] + (2/3)*kimag[1] + (1/6)*kimag[2])
            

    def prob(self, i):
        return self.real[i] ** 2 + self.imag[i] ** 2

        
        
def initial(x):
    return 0.3*np.sin(2*x*np.pi)

t_int = 0.00001
foo = Wavefunction(initial, dt=t_int, trange=(0,0.1))
foo.solve()
    
fig, ax = plt.subplots()
line1, = ax.plot(foo.x, foo.real[0], linewidth=3, color='b')
line2, = ax.plot(foo.x, foo.imag[0], linewidth=3, color='r')
time = ax.text(.7, 1.5, '$t = 0$', fontsize=15)

ax.set_xlabel("Position, $x$")
ax.set_ylabel("Wavefunction,  $\\Psi(x, t)$")
ax.set_ylim([-2,2])

def animate(i):
    line1.set_ydata(foo.real[i])  # update the data
    line2.set_ydata(foo.imag[i]) 
    time.set_text("t = {0:.4f}".format(t_int*i))
    return line1, line2, ax
    
ani = animation.FuncAnimation(fig, animate, np.arange(0, foo.nt,10), interval=1, blit=False)
plt.show()