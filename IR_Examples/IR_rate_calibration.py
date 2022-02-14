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


from scipy import optimize
from Helper.Normal_Vol import volBachelier as vb 
from EMP_Algorithms import emp_mp_algorithm as emp_mp
from IR_Examples import IR_spread_options as spread_pricing


#%% Single rate calibration

"""
Function calibrates the parameters alpha, nu and rho to given implied volatility values
"""
def calibrate_rate(T,calibration_points,calibration_values,params):
    # initial values for the optimization
    initial_guess = [params.alpha,params.nu,params.rho]
    
    # bounds for the parameters
    bnds = ((0., 0.1), (0.01, 1), (-0.9999,0.9999))
    
    # optimization 
    result = optimize.minimize(error_function_rate, initial_guess,
                               args = (T,calibration_points,calibration_values,params),
                               bounds = bnds)
    params.alpha, params.nu, params.rho = result.x
    return params


"""
Minimization function for the calibration of a single rate
"""
def error_function_rate(variable,T,calibration_points,calibration_values,params):
    # set new parameters
    params.alpha, params.nu, params.rho = variable
    
    # matching using EMP-MP 
    nSABR_displacement = 4*params.forward
    params_EMP = emp_mp.emp_mp_to_SABR(T, 2*params.forward, 3*params.forward,
                                       nSABR_displacement,0,params)
    
    # compute the implied volatility along the calibration points
    call, iv = spread_pricing.nSABR_vanilla_call(T,calibration_points,params_EMP)
    
    # return a scaled MSE
    return sum(((iv-calibration_values)/calibration_values)**2)


#%% Spread calibration

"""
Function calibrates the correlation parameter omega between rates to given 
implied volatility values
"""
def calibrate_spread(omega_start,T,calibration_points,calibration_values,
                     params_rate_1,params_rate_2,T_1,T_2):
    # initial values for the optimization
    initial_guess = omega_start
    
    # bound for the parameters
    bnds = [(-1,1)]
    
    #optimization
    result = optimize.minimize(error_function_spread, initial_guess,
                               args = (T,calibration_points,calibration_values,
                                       params_rate_1,params_rate_2,T_1,T_2),
                                       bounds=bnds)
    return result.x


"""
Minimization function for the calibration of the spread
"""
def error_function_spread(variable,T,calibration_points,calibration_values,
                          params_rate_1,params_rate_2,T_1,T_2):
    
    # get parameters for the spread depending on the new correlation
    params_spread = spread_pricing.get_parameters_spread_general(
            -1,1,params_rate_1,params_rate_2,variable,0)
    
    # compute the price and implied volatility
    CMS_spread_price = spread_pricing.compute_CMS_Spread_Caplet(
            T,calibration_points,params_spread, params_rate_1, params_rate_2,
            T_1,T_2)
    iv_CMS_spread = vb.vol_bachelier_diff_prices(1, calibration_points,
                                               params_spread.forward, T,
                                               CMS_spread_price)
    
    # return a scaled MSE
    return sum(((iv_CMS_spread-calibration_values)/calibration_values)**2)

