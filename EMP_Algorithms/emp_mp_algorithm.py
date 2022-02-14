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
Match the projected volatility using the EMP-MP algorithm with an nSABR model as a 
reference model.
"""
def emp_mp_to_SABR(T, point_x1, point_x2, SABR_displacement, SABR_beta, params_base):
    # define SABR params object
    params_new = Params()
    params_new.model = "sabr"
    params_new.beta = SABR_beta
    params_new.forward = params_base.forward
    params_new.displacement = SABR_displacement
    
    # EMP coefficients of the base model
    a_emp_base, b_emp_base, c_emp_base, G_emp_base = emp_base.get_all_emp_coef(T,params_base)
    
    # compute projected EMP coefficients
    a_hat_new = compute_new_dSABR_a_hat_coef(a_emp_base, G_emp_base, params_base, params_new)
    b_emp_new = compute_new_dSABR_b_coef(point_x1, point_x2, b_emp_base,
                                         c_emp_base, params_base, params_new)
    c_emp_new = compute_new_dSABR_c_coef(point_x1, point_x2, b_emp_base,
                                         c_emp_base, params_base, params_new)
    
    # transform EMP coefficients back to model parameters
    params_new.alpha, params_new.nu, params_new.rho = emp_dSABR_parameters.compute_dSABR_parameters_out_of_coefficients(
            T,a_emp_base,G_emp_base,a_hat_new,b_emp_new,c_emp_new,params_base, params_new)
    return params_new


"""
See EMP paper for explicit formula
"""
def compute_new_dSABR_a_hat_coef(a_emp_base, G_emp_base, params_base,
                                 params_proj):
    # import model functions
    importlib = __import__('importlib')
    sf_base = importlib.import_module("Models." + params_base.model + "." +
                                      params_base.model + "_functions")
    sf_proj = importlib.import_module("Models." + params_proj.model + "." +
                                      params_proj.model + "_functions")
    
    # values C(f)
    C_val_base = sf_base.func_C(params_base.forward, params_base)
    C_val_proj = sf_proj.func_C(params_base.forward, params_proj)
    
    return C_val_base * a_emp_base * np.exp(0.5 * G_emp_base) / C_val_proj


"""
Compute hat{b} as in Section 2.3.2 for the dSABR model
"""
def compute_new_dSABR_b_coef(point_x1, point_x2, b_emp_base, c_emp_base,
                             params_base, params_proj):
    # import model functions
    importlib = __import__('importlib')
    sf_proj = importlib.import_module("Models." + params_proj.model +
                                      "." + params_proj.model + "_functions")
    
    # helpel r functions
    r_val_x1 = compute_helper_func_r(point_x1, b_emp_base, c_emp_base,
                                     params_base, params_proj)
    r_val_x2 = compute_helper_func_r(point_x2, b_emp_base, c_emp_base,
                                     params_base, params_proj)
    
    # Compute tilde{z}(x)
    ztilde_x1 = sf_proj.func_z_F(point_x1,params_proj)
    ztilde_x2 = sf_proj.func_z_F(point_x2,params_proj)
    
    term_1 = r_val_x1 * ztilde_x2**2
    term_2 = r_val_x2 * ztilde_x1**2
    term_3 = ztilde_x1*ztilde_x2**2 - ztilde_x1**2*ztilde_x2
    
    return 0.5*(term_1 - term_2)/term_3


"""
Compute hat{c} as in Section 2.3.2 for the nSABR model
"""
def compute_new_dSABR_c_coef(point_x1, point_x2, b_emp_base, c_emp_base,
                             params_base, params_proj):
    # import model functions
    importlib = __import__('importlib')
    sf_proj = importlib.import_module("Models." + params_proj.model + "." +
                                      params_proj.model + "_functions")
    
    # helper function r
    r_val_x1 = compute_helper_func_r(point_x1, b_emp_base, c_emp_base, 
                                     params_base, params_proj)
    r_val_x2 = compute_helper_func_r(point_x2, b_emp_base, c_emp_base,
                                     params_base, params_proj)
    
    # compute tilde{z}(x)
    ztilde_x1 = sf_proj.func_z_F(point_x1,params_proj)
    ztilde_x2 = sf_proj.func_z_F(point_x2,params_proj)
    
    term_1 = r_val_x2*ztilde_x1 - r_val_x1*ztilde_x2
    term_2 = ztilde_x1*ztilde_x2**2 - ztilde_x1**2*ztilde_x2
    return term_1/term_2


"""
See EMP paper for explicit formula
"""
def compute_helper_func_r(x, b_emp_base, c_emp_base, params_base, params_proj):
    # import model functions
    importlib = __import__('importlib')
    sf_base = importlib.import_module("Models." + params_base.model + "." +
                                      params_base.model + "_functions")
    sf_proj = importlib.import_module("Models." + params_proj.model + "." +
                                      params_proj.model + "_functions")
    
    # values C function base model
    C_val_atm_base = sf_base.func_C(params_base.forward, params_base)
    C_val_x_base = sf_base.func_C(x, params_base)
    
    # values C function projection model
    C_val_atm_proj = sf_proj.func_C(params_base.forward, params_proj)
    C_val_x_proj = sf_proj.func_C(x, params_proj)
    
    # value z(x)
    z_val_x = sf_base.func_z_F(x, params_base)
    
    # polynomial expression
    p_helper = 1 + 2*b_emp_base*z_val_x + c_emp_base*z_val_x**2
    
    return C_val_x_base**2/C_val_atm_base**2 * C_val_atm_proj**2/C_val_x_proj**2 * p_helper - 1

