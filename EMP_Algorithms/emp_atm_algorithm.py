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
from Models import Params
from EMP_Algorithms import emp_base_setting as emp_base
from EMP_Algorithms import emp_dSABR_parameters_outof_coefficients as emp_dSABR_parameters


"""
EMP-ATM algorithm projecting to a dSABR model
"""
def eff_SABR_parameters_ATM(T, SABR_displacement, SABR_beta, params_base):
    # define SABR params object
    params_new = Params()
    params_new.model = "sabr"
    params_new.beta = SABR_beta
    params_new.forward = params_base.forward
    params_new.displacement = SABR_displacement
    
    # EMP coefficients of the base model
    a_emp_base, b_emp_base, c_emp_base, G_emp_base = emp_base.get_all_emp_coef(T,params_base)
    
    # Easier valuation functions for the nSABR model
    if SABR_beta == 0:
        # compute projected EMP coefficients
        atilde, btilde, ctilde = nSABR_projected_EMP_coefficients(
                a_emp_base, b_emp_base, c_emp_base, G_emp_base, params_base)
        
        # set new alpha
        params_new.alpha = atilde
        
        # set new nu, if the value is admissible
        if(ctilde<0): 
            params_new.nu = 0.01
            print("ERROR: ATM nu value was imaginary!")
        else:
            params_new.nu = params_new.alpha*np.sqrt(ctilde)
        
        # set new rho
        rho_computed = btilde*params_new.alpha/params_new.nu
        
        # keep the absolute value of rho smaller 1
        params_new.rho = max(min(rho_computed,0.999),-0.999)
        
    else:
        # compute projected EMP coefficients
        a_hat_new, btilde, ctilde = dSABR_projected_EMP_coefficients(
                a_emp_base, b_emp_base, c_emp_base, G_emp_base, params_new, params_base)
        
        # transform EMP coefficients back to model parameters
        params_new.alpha, params_new.nu, params_new.rho = emp_dSABR_parameters.compute_dSABR_parameters_out_of_coefficients(
                T, a_emp_base,G_emp_base,a_hat_new, btilde, ctilde, params_base, params_new)
    
    return params_new


"""
Computation of the EMP-ATM coefficients for a projection to the nSABR model
"""
def nSABR_projected_EMP_coefficients(a_emp_base, b_emp_base, c_emp_base,
                                     G_emp_base, params_base):
    # import the model functions in dependence on the model type
    importlib = __import__('importlib')
    sf_base = importlib.import_module("Models." + params_base.model +
                                      "." + params_base.model + "_functions")
    
    # Evaluate the C function and its derivatives
    C_val = sf_base.func_C(params_base.forward,params_base)
    C_val_prime = sf_base.func_Gamma(params_base.forward,params_base)
    C_val_prime_prime = compute_C_prime_prime(params_base.forward,params_base)
    
    # compute atilde
    atilde = C_val * a_emp_base * np.exp(0.5 * G_emp_base)
    
    # match the derivatives
    btilde = (C_val_prime + b_emp_base)/C_val
    ctilde = (C_val_prime**2 + C_val_prime_prime*C_val + 
              3*C_val_prime*b_emp_base + c_emp_base)/(C_val**2)
    
    return atilde, btilde, ctilde


"""
Computation of the EMP-ATM coefficients for a projection to the dSABR model
"""
def dSABR_projected_EMP_coefficients(a_emp_base, b_emp_base, c_emp_base,
                                     G_emp_base, params_new, params_base):
    # import the model functions in dependence on the model type
    importlib = __import__('importlib')
    sf_base = importlib.import_module("Models." + params_base.model +
                                      "." + params_base.model + "_functions")
    sf_proj = importlib.import_module("Models." + params_new.model +
                                      "." + params_new.model + "_functions")
    
    # Evaluate C and its derivatives
    C_val = sf_base.func_C(params_base.forward,params_base)
    C_val_prime = sf_base.func_Gamma(params_base.forward,params_base)
    C_val_prime_prime = compute_C_prime_prime(params_base.forward,params_base)
    
    # Evaluate tilde(C) and its derivatives
    C_val_tilde = sf_proj.func_C(params_new.forward,params_new)
    C_val_prime_tilde = sf_proj.func_Gamma(params_new.forward,params_new)
    C_val_prime_prime_tilde = compute_C_prime_prime(params_new.forward,params_new)
    
    # new alpha hat 
    a_hat_new = C_val * a_emp_base * np.exp(0.5 * G_emp_base) / C_val_tilde
    
    # matching the derivatives
    btilde = (C_val_prime + b_emp_base)/C_val*C_val_tilde - C_val_prime_tilde
    term_1 = C_val_prime**2 + C_val_prime_prime*C_val
    term_2 = 3*C_val_prime*b_emp_base + c_emp_base
    term_3 = -C_val_prime_tilde**2 - C_val_prime_prime_tilde * C_val_tilde-3*C_val_prime_tilde * btilde
    ctilde = C_val_tilde**2/C_val**2 *(term_1 + term_2) + term_3
    return a_hat_new, btilde, ctilde


"""
Approximation function for C(x)''
"""
def compute_C_prime_prime(x,params): 
    return params.beta*(params.beta-1)/(x + params.displacement)**(2-params.beta)

