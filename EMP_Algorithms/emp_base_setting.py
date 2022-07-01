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
This file presents the fundamental setting in using the EMP. The EMP considers 
projected volatility functions of the form

\sigma_{proj}^2(t,x) = C(x)^2 a(t)^2 e^{G(t)} (1+2b(t)z(x)+c(t)z(x)^2)

Since these coefficients are not in a one to one correspondence to the coefficients
of the Hagan 1d-PDE schemes, this file serves as a translation between these coefficients.
This includes:
    a(t)
    b(t)
    c(t)
    G(t)
"""


import autograd.numpy as np


"""
Get all EMP coefficients
"""
def get_all_emp_coef(T,params):
    # import the model functions in dependence on the model type
    importlib = __import__('importlib')
    sf = importlib.import_module("Models." + params.model + "." +
                                 params.model + "_functions")
    
    # get the pde coefficients
    a_pde = sf.fkm_coeff_a(params)
    b_pde = sf.fkm_coeff_b(params)
    c_pde = sf.fkm_coeff_c(params)
    G_pde = sf.fkm_coeff_G(T,sf.func_Gamma(params.forward,params),params)
    
    # translate the pde coefficients into emp coefficients
    a_emp = np.sqrt(a_pde)
    b_emp = b_pde/a_pde
    c_emp = c_pde/a_pde
    G_emp = G_pde*T
    return a_emp, b_emp, c_emp, G_emp


"""
Get the EMP projected volatility function \sigma_{proj}^2(t,x)
In accordance to Proposition 2.3
"""
def get_emp_locvol(T,x,params):
    # import the model functions in dependence on the model type
    importlib = __import__('importlib')
    sf = importlib.import_module("Models." + params.model + "." +
                                 params.model + "_functions")
    
    # get all emp coefficients
    a_emp, b_emp, c_emp, G_emp = get_all_emp_coef(T, params)
    
    # get C values
    C_vec = sf.func_C(x,params)
    
    # get z values
    z_vec = sf.func_z_F(x,params)
    
    # return function from Proposition 2.3
    return C_vec**2*a_emp**2*np.exp(G_emp)*(1 + 2*b_emp*z_vec + c_emp*z_vec**2)


"""
Get the EMP projected volatility function \sigma_{proj}^2(t,x) for a full surface
"""
def get_emp_locvol_surface(time_grid,strike_grid,params):
    # (t,K) surface 
    locvol_surface = np.empty((len(time_grid),len(strike_grid)))
    
    # compute values for all maturity/strike combinations
    for time_index in np.arange(0,len(time_grid),1):
        locvol_surface[time_index,:] = get_emp_locvol(time_grid[time_index],strike_grid,params)
    
    return locvol_surface

