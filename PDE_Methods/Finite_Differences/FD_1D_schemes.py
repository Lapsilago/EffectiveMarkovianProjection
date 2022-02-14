# -*- coding: utf-8 -*-
"""

'Copyright 2022 Mike Felpel, Dr. Joerg Kienitz, Dr. Thomas McWalter

'Redistribution and use in source and binary forms, with or without modification,
'are permitted provided that the following conditions are met:

'1. Redistributions of source code must retain the above copyright notice,
'this list of conditions and the following disclaimer.

'2. Redistributions in binary form must reproduce the above copyright notice,
'this list of conditions and the following disclaimer in the documentation
'and/or other materials provided with the distribution.

'3. Neither the name of the copyright holder nor the names of its contributors
'may be used to endorse or promote products derived from this software without
'specific prior written permission.

'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
'"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
'THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
'ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
'FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
'(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
'LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
'ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
'OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
'THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""
The following time marching schemes are formulated for a forward setting.
This means we start with a known initial value U_0 and evolve forward in time.
"""


from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import numpy as np


"""
The fully implicit scheme for time independent matrix A and b
"""
def FI_scheme_deterministic(m, tsteps, U_0, delta_t, A, b):
    U = U_0
    I = np.identity(m)
    lhs = csc_matrix(I - delta_t * A)
    inv_lhs = inv(lhs)
    for n in range(1, tsteps + 1):
        U = inv_lhs * (U + delta_t * b)
    return U


"""
A single Lawson-Swayne step with time dependent matrices A
"""
def LS_step_time_dependent(m, U_0, delta_t, A_1, A_2, b_1, b_2):
    sqrt2 = np.sqrt(2.)
    ls_step_coeff = 1. - sqrt2 * 0.5
    dt = delta_t * ls_step_coeff
    
    U = U_0
    U_1 = FI_scheme_deterministic(m, 1, U, dt, A_1, b_1)
    U_2 = FI_scheme_deterministic(m, 1, U_1, dt, A_2, b_2)
    U = (sqrt2 + 1.) * U_2 - sqrt2 * U_1
    
    return U,U_1,U_2

