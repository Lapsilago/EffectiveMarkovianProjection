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
from Models.slvsabr import slvsabr_functions as sf_slv
from SLV_Algorithms import calibration_slv_leverage_general as calibrator_leverage


#%% Calibration scheme for the slv-nSABR model


"""
Calibration schmeme applied to calibrate the leverage function of the 
slv-nSABR model
"""
def calibrate_slv_nSABR(market_surface, T_ex, forward, alpha, nu, rho,
                        L_grid, sigma_grid, displacement):
    # initzialize the base nSABR model
    params_slv_nSABR = set_parameter_specifications_slv_nSABR(forward, alpha, nu,
                                                              rho, sigma_grid,
                                                              L_grid, displacement)
    
    # calibrate the time leverage sigma 
    params_slv_nSABR.sigma_val = calibrator_leverage.compute_sigma_val(market_surface,
                                                                       params_slv_nSABR)
    
    # calibrate the space leverage L
    params_slv_nSABR.L_val = calibrator_leverage.compute_L_val(T_ex, market_surface,
                                                               params_slv_nSABR)
    
    # initialize the remaining functions
    params_slv_nSABR = sf_slv.initialize_all_emp_coeff(params_slv_nSABR)
    return params_slv_nSABR


"""
This function initializes a Params() class for the slv-nSABR model
"""
def set_parameter_specifications_slv_nSABR(forward, alpha, nu, rho,
                                           sigma_grid, L_grid, displacement):
    params_slvnSABR = Params()
    params_slvnSABR.model = "slvsabr"
    params_slvnSABR.basemodel = "sabr"
    params_slvnSABR.forward = forward
    params_slvnSABR.beta = 0.
    params_slvnSABR.alpha = alpha
    params_slvnSABR.nu =  nu
    params_slvnSABR.rho = rho
    params_slvnSABR.displacement = displacement
    params_slvnSABR.L_grid = L_grid
    params_slvnSABR.L_val = np.ones(len(L_grid))
    params_slvnSABR.sigma_grid = sigma_grid
    params_slvnSABR.sigma_val = np.ones(len(sigma_grid))
    return params_slvnSABR


#%% Evaluation of the calibrated model
"""
These functions allow the evaluation of the calibrated model
"""


"""
This function evaluates the projected volatility function on the internal 
model time grid and a specified strike grid
"""
def compute_slv_projvol_surface(params_slvnSABR, strike_grid):
    # get the internal model time grid
    time_grid = params_slvnSABR.sigma_grid
    
    # define the projected volatility surface
    projvol_slv = np.zeros((len(time_grid),len(strike_grid)))
    
    # load the already precomputet emp coefficients on the internal time grid
    emp_a_slv = params_slvnSABR.emp_a_on_grid
    emp_b_slv = params_slvnSABR.emp_b_on_grid 
    emp_c_slv = params_slvnSABR.emp_c_on_grid 
    emp_G_slv = params_slvnSABR.emp_G_on_grid 
    
    # compute the strike dependent functions
    C_vec = sf_slv.func_C(strike_grid, params_slvnSABR)
    z_vec = sf_slv.func_z_F(strike_grid, params_slvnSABR)
    
    # compute the local volatility 
    for time_index in np.arange(0,len(time_grid)):
        poly = (1 + 2*emp_b_slv[time_index]*z_vec + emp_c_slv[time_index]*z_vec**2)
        E = np.exp(emp_G_slv[time_index])
        projvol_slv[time_index,:] = params_slvnSABR.sigma_val[time_index]**2 * C_vec**2*emp_a_slv[time_index]**2*E*poly
    return projvol_slv
