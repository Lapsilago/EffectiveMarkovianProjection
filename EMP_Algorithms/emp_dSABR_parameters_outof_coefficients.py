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


import autograd.numpy as np
import scipy.optimize as optimize


"""
Compute the parameters of the dSABR model out of the EMP coefficients using 
the procedure of Section 2.3.4
"""
def compute_dSABR_parameters_out_of_coefficients(T,a_base, G_base, a_hat, b_tilde,
                                                 c_tilde, params_base, params_new):
    # import model functions
    importlib = __import__('importlib')
    sf_base = importlib.import_module("Models." + params_base.model + "." +
                                      params_base.model + "_functions")
    sf_proj = importlib.import_module("Models." + params_new.model + "." +
                                      params_new.model + "_functions")
    
    # ATM C values
    C_val_base = sf_base.func_C(params_base.forward, params_base)
    C_val_proj = sf_proj.func_C(params_base.forward, params_new)
    
    # initial guess for alpha
    alpha_initial_guess = get_initial_guess_alpha(a_base, b_tilde, G_base, 
                                                  C_val_base, C_val_proj,
                                                  params_base, params_new)
    
    # optimization to solve for alpha
    result = optimize.root(root_function, alpha_initial_guess, tol=1e-20,
                           args = (C_val_base, C_val_proj,a_hat,b_tilde,
                                   params_new.beta,params_new.forward,
                                   params_new.displacement,T))
    alpha_new = result.x
    
    # compute nu
    nu_new = np.sqrt(c_tilde)*alpha_new
    
    #compute rho and keep the absolute value of rho smaller 1
    rho_new = max(min(b_tilde/np.sqrt(c_tilde),0.999),-0.999)
    return alpha_new, nu_new, rho_new


"""
Initial Guess for alpha using the first order Taylor approximation
"""
def get_initial_guess_alpha(a_base, b_tilde, G_base, C_val_base, C_val_proj,
                            params_base, params_proj):    
    f1 = C_val_base**2/C_val_proj**2 * a_base**2
    f2 = b_tilde * params_proj.beta * (params_proj.forward +
                                       params_proj.displacement)**(params_proj.beta-1)
    return np.sqrt(-f1*(1+G_base)/(f1*f2-1))


"""
Error function for the optimization to solve for alpha**2, see Equation 2.15
"""
def root_function(variable, C_base, C_proj, a_hat, b_tilde, beta,
                  forward, displacement, T):
    term1 = variable**2 * np.exp(b_tilde * variable**2 *
                                 beta * (forward + displacement)**(beta-1) * T)
    return term1 - a_hat**2

