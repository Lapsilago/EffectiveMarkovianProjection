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
This file contains the computational formulas taken from the paper
CMS Spread Options, by Patrick S. Hagan and Andrew Lesniewski
"""


import autograd.numpy as np
from scipy.stats import norm
from Models import Params


#%% nSABR Calls from CMS Spread Options


"""
Equation 3.44a from CMS Spread Options
"""
def nSABR_vanilla_call(T,K,params):
    sig_N = sigma_N(T,K,params)
    temp = (params.forward-K)/(sig_N*np.sqrt(T))
    price = (params.forward-K)*norm.cdf(temp) + sig_N*np.sqrt(T)*norm.pdf(temp)
    return price, sig_N


"""
Equation 3.46a from CMS Spread Options
"""
def sigma_N(T,K,params):
    # convert K into an array if only one strike is considered
    if isinstance(K,float):
        kfloat = K
        K = np.zeros(1)
        K[0] = kfloat
    
    output = np.array(K)
    xi = np.array(K)
    Y = np.array(K)
    
    # find the ATM values
    index_atm = (np.abs((K-params.forward)) < 0.0000000001)
    index_not_atm = (np.abs((K-params.forward)) >= 0.0000000001)
    
    # output for the ATM
    sigma_atm = sigma_N_atm(T,params)
    output[index_atm] = sigma_atm
    
    # adjustment if not ATM
    xi[index_not_atm] = func_xi(K[index_not_atm],params)
    Y[index_not_atm] = func_Y(K[index_not_atm],params)
    output[index_not_atm] = sigma_atm*xi[index_not_atm]/Y[index_not_atm]
    
    return output


"""
Equation 3.46b from CMS Spread Options
"""
def sigma_N_atm(T,params):
    return params.alpha*(1 + (2-3*params.rho**2)*params.nu**2*T/24)


"""
Equation 3.46c from CMS Spread Options
"""
def func_xi(K,params):
    return params.nu * (params.forward-K)/params.alpha


"""
Equation 3.46d from CMS Spread Options
"""
def func_Y(K,params):
    xi = func_xi(K,params)
    nominator = np.sqrt(1 - 2*params.rho*xi + xi**2) - params.rho + xi
    return np.log(nominator/(1-params.rho))


"""
Equation 3.47a from CMS Spread Options
"""
def nSABR_quadratic_call(T,K,params):
    s_Q = s_q(T,params)
    S_adj = s_adj(T,K,params)
    sigma_Q = sigma_q(T,K,params)
    
    temp = (S_adj-K)/(sigma_Q*np.sqrt(T))
    return ((params.forward-K)**2 + s_Q**2*T)*norm.cdf(temp) +(S_adj-K)*sigma_Q*np.sqrt(T)*norm.pdf(temp)


"""
Equation 3.47c from CMS Spread Options
"""
def nSABR_quadratic_swap(T,K,params):
    return (params.forward-K)**2 + s_q(T,params)**2*T


"""
Equation 3.48a from CMS Spread Options
"""
def sigma_q(T,K,params):
    return sigma_N(T,K,params)*(1+params.nu**2*T/6)


"""
Equation 3.48b from CMS Spread Options
"""
def s_q(T,params):
    return sigma_N(T,params.forward,params)*(1+(4+3*params.rho**2)*params.nu**2*T/24)


"""
Equation 3.48c from CMS Spread Options
"""
def s_adj(T,K,params):
    # convert K into an array if only one strike is considered
    if isinstance(K,float):
        kfloat = K
        K = np.zeros(1)
        K[0] = kfloat
    
    output = np.array(K)
    
    # find the ATM values
    index_atm = (np.abs((K-params.forward)) < 0.0000000001)
    index_not_atm = (np.abs((K-params.forward)) >= 0.0000000001)
    
    # ATM value
    output[index_atm] = params.forward
    
    # adjustment if not ATM
    output[index_not_atm] = params.forward + 0.5*(sigma_q(T,K[index_not_atm],params)**2-
          sigma_q(T,params.forward,params)**2)/(K[index_not_atm]-params.forward)*T
    return output


#%% CMS Spread Caplet


"""
Equation 4.2a from CMS Spread Options
"""
def compute_CMS_Spread_Caplet(T,K,params_spread, params_rate_1, params_rate_2,
                              T_1,T_2):
    # get the discount factor DF(0,payment_date_T_0) where payment_date_T_0 = T
    DF_0_T = 1-T*params_spread.forward
    
    # get the convexity coefficient
    lambda_s = get_convexity_coeff(T,T_1,T_2,params_spread,params_rate_1, params_rate_2)
    
    # get the vanilla call
    vanilla_call, iv = nSABR_vanilla_call(T,K,params_spread)
    
    # get the quadratic call
    quadratic_call = nSABR_quadratic_call(T,K,params_spread)
    
    return DF_0_T*((1-lambda_s*(params_spread.forward-K))*vanilla_call + lambda_s*quadratic_call)


"""
Equation 4.3 from CMS Spread Options 
Attention: the indexing of the rates in the paper is different compared to CMS Spread Options
"""
def get_convexity_coeff(T,T_1,T_2,params_spread,params_rate_1, params_rate_2):
    quadratic_swap_rate_1 = nSABR_quadratic_swap(T,params_rate_1.forward,params_rate_1)
    quadratic_swap_rate_2 = nSABR_quadratic_swap(T,params_rate_2.forward,params_rate_2)
    quadratic_swap_spread = nSABR_quadratic_swap(T,params_spread.forward,params_spread)
    
    lambda_1 = get_convexity_coeff_rate(T_1,T)
    lambda_2 = get_convexity_coeff_rate(T_2,T)
    return (lambda_2*quadratic_swap_rate_2-lambda_1*quadratic_swap_rate_1)/quadratic_swap_spread


"""
Approximation of the convexity coefficient in accordance to Effective Markovian Projection
"""
def get_convexity_coeff_rate(T_i,T_0):
    return 0.5*(T_i -T_0)




#%% nSABR basket

"""
Equation A.45a from CMS Spread Options with spread = lambda2*R2- lambda1*R1
"""
def compute_Delta_general(lambda_1, lambda_2, params_rate_1, params_rate_2,
                          corr_omega):
    term_1 = lambda_2**2 * params_rate_2.alpha**2
    term_2 = 2*corr_omega*lambda_1*lambda_2*params_rate_1.alpha*params_rate_2.alpha
    term_3 = lambda_1**2 * params_rate_1.alpha**2
    return np.sqrt(term_1 + term_2 + term_3)


"""
Equation A.45b from CMS Spread Options
"""
def compute_rho_bar_general(lambda_1, lambda_2, params_rate_1, params_rate_2, corr_omega):
    Delta = compute_Delta_general(lambda_1, lambda_2, params_rate_1,
                                  params_rate_2, corr_omega)
    term_1 = lambda_2 * params_rate_2.alpha * params_rate_2.rho
    term_2 = lambda_1 * params_rate_1.alpha * params_rate_1.rho
    return (term_1 + term_2)/Delta


"""
Equation A.45c from CMS Spread Options
"""
def compute_nu_bar_general(lambda_1, lambda_2, params_rate_1, params_rate_2, corr_omega):
    Delta = compute_Delta_general(lambda_1, lambda_2, params_rate_1,
                                  params_rate_2, corr_omega)
    term_1 = lambda_2**2 * params_rate_2.alpha**2 * params_rate_2.nu
    term_2 = corr_omega*lambda_1*lambda_2*params_rate_1.alpha*params_rate_2.alpha*(
            params_rate_1.nu+params_rate_2.nu)
    term_3 = lambda_1**2 * params_rate_1.alpha**2 * params_rate_1.nu
    return (term_1 + term_2 + term_3)/(Delta**2)


"""
Equation A.46 from CMS Spread Options
"""
def get_parameters_spread_general(lambda_1, lambda_2, params_rate_1,
                                  params_rate_2, corr_omega,
                                  nSABR_displacement):
    # initialize parameter object
    params_spread = Params()
    params_spread.model = "sabr"
    params_spread.beta = 0.
    params_spread.displacement = nSABR_displacement
    
    # compute the forward of the spread
    params_spread.forward = lambda_2*params_rate_2.forward + lambda_1*params_rate_1.forward
    
    # approximation using Equation A45
    params_spread.alpha = compute_Delta_general(lambda_1, lambda_2, params_rate_1, 
                                  params_rate_2, corr_omega)
    params_spread.nu = compute_nu_bar_general(lambda_1, lambda_2, params_rate_1,
                                    params_rate_2, corr_omega)
    params_spread.rho = compute_rho_bar_general(lambda_1, lambda_2, params_rate_1,
                                      params_rate_2, corr_omega)
    
    return params_spread

