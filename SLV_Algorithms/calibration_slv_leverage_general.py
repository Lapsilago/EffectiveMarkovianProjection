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


#%% General slv calibration scheme
"""
The general calibration scheme to calibrate the piecewise leverage functions 
for time and space
"""


'''
This function calibrates the time leverage function sigma() along the market
ATM surface
'''
def compute_sigma_val(market_surface, params_slv):
    # import the corresponding model functionalities
    importlib = __import__('importlib')
    sf_slv = importlib.import_module("Models." + params_slv.model + "." + params_slv.model + "_functions")
    
    # find the strike index i depicting the ATM value
    i_index = np.rint(sf_slv.index_search_space(params_slv.forward,params_slv.L_grid,params_slv)[0]).astype(int)
    
    # get the market ATM values
    market_locvol = market_surface[:,i_index]
    
    # initialize the slv model to get the initial emp coefficients
    params_slv = sf_slv.initialize_all_emp_coeff(params_slv)
    
    # get the values of the C function, accessible through the underlying base model
    C_val = sf_slv.func_C_base(params_slv.L_grid[i_index],params_slv)
    
    # define an array for the sigma values
    sigma_val = np.ones(len(params_slv.sigma_grid))
    
    # run through the time grid
    for time_index in np.arange(0,len(params_slv.sigma_grid),1):
        # get the a coefficient
        emp_a = params_slv.emp_a_on_grid[time_index]
        
        # get the G coefficient
        coeff_G_tm  = params_slv.emp_G_on_grid[time_index]
        
        # compute the value of sigma 
        denominator = emp_a**2*C_val**2*np.exp(coeff_G_tm)
        sigma_val[time_index] = np.sqrt(market_locvol[time_index]/denominator)
        
        # update params to compute new values of G and a 
        params_slv.sigma_val = sigma_val
        params_slv = sf_slv.initialize_all_emp_coeff(params_slv)
        
    return sigma_val


'''
This function calibrates the space leverage function L() along a specified
maturity T_fit. This function assumes, that the sigma() function is already
calibrated
'''
def compute_L_val(T_fit, market_surface, params_slv):
    # import the corresponding model functionalities
    importlib = __import__('importlib')
    sf_slv = importlib.import_module("Models." + params_slv.model + "." + params_slv.model + "_functions")
    
    # ATM index i
    i_index = np.rint(sf_slv.index_search_space(params_slv.forward,params_slv.L_grid,params_slv)[0]).astype(int)
    
    # index of the fitting maturity in the time grid
    T_index = np.rint(sf_slv.index_search_time(T_fit,params_slv.sigma_grid)[0]).astype(int)
    
    # get the market values for the fitting maturity
    market_locvol = market_surface[T_index,:]
    market_locvol_atm = market_locvol[i_index]
    
    # initialize the slv model to get the correct emp coefficients
    params_slv = sf_slv.initialize_all_emp_coeff(params_slv)
    
    # load the EMP coefficients for the corresponding fitting maturity
    emp_b = params_slv.emp_b_on_grid[T_index]
    emp_c = params_slv.emp_c_on_grid[T_index]
    
    # get the values of the C function, accessible through the underlying base model
    C_val = sf_slv.func_C_base(params_slv.L_grid, params_slv)
    C_val_atm = sf_slv.func_C_base(params_slv.forward,params_slv)
    
    # initialize slv functions
    L_val = np.zeros(len(params_slv.L_grid))
    z_val = np.zeros(len(params_slv.L_grid))
    
    # and set the specified value for the ATM index i
    z_val[i_index] = 0
    L_val[i_index] = 1
    
    # compute C base increments
    z_base = sf_slv.func_F_z_base(params_slv.L_grid, params_slv)
    z_base_increments = z_base[1:] - z_base[:-1]
    
    # compute the OTM strike values of the leverage
    for grid_index in np.arange(i_index + 1, len(params_slv.L_grid),1):
        # update the transformed values
        z_val[grid_index] = z_val[grid_index-1] + z_base_increments[grid_index-1]/L_val[grid_index-1]
        
        # update the leverage function
        L_val[grid_index] = compute_single_l_val(market_locvol[grid_index]/market_locvol_atm,
             emp_b, emp_c ,C_val[grid_index]/C_val_atm, z_val[grid_index])
    
    # compute the OTM strike values of the leverage
    for grid_index in np.arange(i_index-1,-1,-1):
        # update the the transformed values
        z_val[grid_index] = z_val[grid_index + 1] - z_base_increments[grid_index]/L_val[grid_index+1]
        
        # update the leverage function
        L_val[grid_index] = compute_single_l_val(market_locvol[grid_index]/market_locvol_atm,
             emp_b, emp_c ,C_val[grid_index]/C_val_atm, z_val[grid_index])
    
    return L_val


"""
This function computes the new l value given all information of the previous step
"""
def compute_single_l_val(locvol, emp_b, emp_c ,C_val, z_val):
    denominator = (1 + 2*emp_b*z_val + emp_c*z_val**2)*C_val**2
    return np.sqrt(locvol/denominator)

