from numpy import power, zeros_like, empty_like

def harmonic(x, k=500):
    # harmonic potential
    return 0.5 * k * power(x, 2)


def free(x):
    # free particle potential
    return zeros_like(x)


def finite_barrier(x, x0=0, V0=2, a=0.5):
    # finite potential barrier
    # x0 - barrier location
    # a - barrier thickness
    # V0 - barrier height
    V = empty_like(x)
    for i in range(x.shape[0]):
        if x0 < x[i] - x0 < a:
            V[i] = V0
        else:
            V[i] = 0
    return V
