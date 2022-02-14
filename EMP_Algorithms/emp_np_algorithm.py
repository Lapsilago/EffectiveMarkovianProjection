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
from Models import Params
from EMP_Algorithms import emp_base_setting as emp_base
from EMP_Algorithms import emp_mp_algorithm as emp_mp_alg
from EMP_Algorithms import emp_dSABR_parameters_outof_coefficients as emp_dSABR_parameters


"""
Match the local volatility using the EMP-NP algorithm with an nSABR model as a 
reference model.
"""
def emp_np_to_SABR(T, SABR_displacement, SABR_beta, params_base):
    # define nSABR params object
    params_new = Params()
    params_new.model = "sabr"
    params_new.beta = SABR_beta
    params_new.forward = params_base.forward
    params_new.displacement = SABR_displacement
    
    # N support points
    support_points = get_support_points(params_base)
    support_values = emp_base.get_emp_locvol(T,support_points,params_base)
    
    if SABR_beta == 0:
        # initial guess
        initial_guess = get_initial_guess_nSABR(T, 0.5*params_base.forward,
                                        2*params_base.forward, SABR_displacement,
                                        params_base)
        
        # optimization
        result = optimize.minimize(error_function_nSABR, initial_guess,
                                   args = (support_values,support_points,
                                           params_new.forward ))
        a_emp_new, b_emp_new, c_emp_new = result.x
        
        # translate to nSABR parameters
        params_new.alpha = a_emp_new
        params_new.nu = params_new.alpha*np.sqrt(c_emp_new)
        rho_computed = b_emp_new * params_new.alpha/params_new.nu
        
        # keep the absolute value of rho smaller 1
        params_new.rho = max(min(rho_computed,0.999),-0.999)
        
    else: 
        # initial guess
        initial_guess = get_initial_guess_dSABR(T, 0.5*params_base.forward,
                                          3*params_base.forward,  SABR_displacement, 
                                          SABR_beta, params_base)
        
        # optimization
        result = optimize.minimize(error_function_dSABR, initial_guess,
                                   args = (support_values,support_points,
                                           params_new.forward,
                                           params_new.displacement,
                                           params_new.beta ))
        a_emp_new, b_emp_new, c_emp_new = result.x
        
        # emp coefficients of the base model
        a_emp_base, b_emp_base, c_emp_base, G_emp_base = emp_base.get_all_emp_coef(T,params_base)
        
        # translate into dSABR parameters
        params_new.alpha, params_new.nu, params_new.rho = emp_dSABR_parameters.compute_dSABR_parameters_out_of_coefficients(
                T,a_emp_base,G_emp_base,a_emp_new,b_emp_new,c_emp_new,params_base, params_new)
    
    return params_new


"""
Along these points the minimization is applied
"""
def get_support_points(params):
    return np.arange(0.5*params.forward , 3.5*params.forward , 0.5*params.forward)


"""
Initial Guess using the EMP-MP algorithm to project to a nSABR
"""
def get_initial_guess_nSABR(T, point_x1, point_x2, nSABR_displacement, params_base):
    # apply the EMP-MP to get an initial guess for the parametrization
    params_mp = emp_mp_alg.emp_mp_to_SABR(T, point_x1, point_x2,
                                          nSABR_displacement, 0, params_base)
    
    # translate the parametrization into EMP-coefficients    
    a_emp_guess, b_emp_guess, c_emp_guess, G_emp_guess = emp_base.get_all_emp_coef(T,params_mp)
    
    return [a_emp_guess,b_emp_guess,c_emp_guess]


"""
Error function for the optimization for a nSABR model
"""
def error_function_nSABR(variable, goal_val, support_points, forward):
    a, b, c = variable
    locvol_new =  a**2*(1 + 2*b*(support_points-forward) + c*(support_points-forward)**2)
    return sum(((goal_val-locvol_new)/goal_val)**2)


"""
Initial Guess using the EMP-MP algorithm to project to a dSABR
"""
def get_initial_guess_dSABR(T, point_x1, point_x2,  dSABR_displacement,
                            dSABR_beta, params_base):
    # use emp-mp for a first guess 
    params_mp = emp_mp_alg.emp_mp_to_SABR(T, point_x1, point_x2,
                                          dSABR_displacement, dSABR_beta, params_base)
    
    a_emp_guess, b_emp_guess, c_emp_guess, G_emp_guess = emp_base.get_all_emp_coef(T, params_mp)
    a_hat_guess = a_emp_guess * np.exp(0.5*G_emp_guess)
    return [a_hat_guess, b_emp_guess, c_emp_guess]


"""
Error function for the optimization for a dSABR model
"""
def error_function_dSABR(variable, goal_val, support_points, forward,
                         displacement, beta):
    a, b, c = variable
    C_tilde = (support_points + displacement)**beta
    z_tilde = ((support_points + displacement)**(1-beta) -
               (forward + displacement)**(1-beta))/(1-beta)
    locvol_new =  C_tilde**2*a**2*(1 + 2*b*z_tilde + c*z_tilde**2)
    return sum(((goal_val-locvol_new)/goal_val)**2)

