from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.complex128 # declare datatype as complex128
ctypedef np.complex128_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
def solve(np.ndarray[DTYPE_t, ndim=2] psi, np.ndarray[DTYPE_t, ndim=2] C, int nt):
    # returns a solved wavefuntion array
    cdef int i
    for i in range(nt - 1):
        psi[i + 1] = np.dot(C, psi[i])
    return psi