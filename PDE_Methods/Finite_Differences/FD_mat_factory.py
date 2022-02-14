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


import numpy as np
from PDE_Methods.Finite_Differences import FD_coeff as cf


"""
Matrix A for the 1d pde scheme
"""
def make_A_hagan_one_dim_locvol(m, Delta_s, a11):
    A = np.zeros((m, m))
    
    # Equation 10
    l_10 = [-1, 0, 1]
    
    for i in range(1, m-1):
        for k in l_10:
            A[i , i + k] += a11[i + k] * cf.FD_coeff_delta_ls(i, k, Delta_s)
    
    A[0 , 0] = a11[0] * cf.FD_coeff_delta_ls(0, 1, Delta_s)
    A[0 , 1] = a11[1] * cf.FD_coeff_delta_ls(0, 1, Delta_s)
    
    A[m-1 , m-2] = a11[m-2] * cf.FD_coeff_delta_ls(m - 2, 1, Delta_s)
    A[m-1 , m-1] = a11[m-1] * cf.FD_coeff_delta_ls(m - 2, 1, Delta_s)
    return A

