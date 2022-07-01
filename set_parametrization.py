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


from Models import Params


"""
Return a ZABR model with parameters in accordance to Table 3
"""
def c(set_number,case):
    params = Params()
    params.model = "zabr"
    params.forward = 0.005
    beta_set, nu_set, rho_set, gamma_set = get_set_parameters(set_number)
    params.beta = beta_set[case-1]
    params.alpha = 0.3*params.forward**(1-params.beta)
    params.nu =  nu_set[case-1]
    params.rho = rho_set[case-1]
    params.displacement = 0.002
    params.gamma = gamma_set[case-1]
    return params


"""
Return the parametrization based on the selected set
"""
def get_set_parameters(set_number):
    if(set_number == 1): 
        beta_set, nu_set, rho_set, gamma_set = get_set_1()
    elif(set_number == 2): 
        beta_set, nu_set, rho_set, gamma_set = get_set_2()
    elif(set_number == 3): 
        beta_set, nu_set, rho_set, gamma_set = get_set_3()
    elif(set_number == 4): 
        beta_set, nu_set, rho_set, gamma_set = get_set_4()
    else:
        beta_set, nu_set, rho_set, gamma_set = get_test_set()
    return beta_set, nu_set, rho_set, gamma_set


"""
Return the maturity of the selected set
"""
def get_set_maturity(set_number):
    if(set_number == 1): 
        maturtity_set = 5
    elif(set_number == 2): 
        maturtity_set = 5
    elif(set_number == 3): 
        maturtity_set = 10
    elif(set_number == 4): 
        maturtity_set = 1
    else:
        maturtity_set = 1
    return maturtity_set


"""
Return the beta of the selected set and case
"""
def get_beta(set_number,case):
    beta_set, nu_set, rho_set, gamma_set = get_set_parameters(set_number)
    return beta_set[case-1]


"""
Set 1 in accordance to Table 3
"""
def get_set_1():
    #Set 1
    beta_set = [0.2,0.4,0.6,0.8]
    nu_set = [0.3,0.3,0.3,0.3,0.3]
    rho_set = [-0.3,-0.3,-0.3,-0.3]
    gamma_set = [0.8,0.8,0.8,0.8]
    return beta_set, nu_set, rho_set, gamma_set


"""
Set 2 in accordance to Table 3
"""
def get_set_2():
    #Set 2
    beta_set = [0.2,0.4,0.6,0.8]
    nu_set = [0.3,0.3,0.3,0.3,0.3]
    rho_set = [-0.7,-0.7,-0.7,-0.7]
    gamma_set = [0.9,0.9,0.9,0.9]
    return beta_set, nu_set, rho_set, gamma_set


"""
Set 3 in accordance to Table 3
"""
def get_set_3():
    #Set 3
    beta_set = [0.2,0.4,0.6,0.8]
    nu_set = [0.3,0.3,0.3,0.3,0.3]
    rho_set = [0,0,0,0]
    gamma_set = [0.8,0.8,0.8,0.8]
    return beta_set, nu_set, rho_set, gamma_set


"""
Set 4 in accordance to Table 3
"""
def get_set_4():
    ##Set 4
    beta_set = [0.2,0.4,0.6,0.8]
    nu_set = [0.5,0.5,0.5,0.5,0.5]
    rho_set = [-0.3,-0.3,-0.3,-0.3]
    gamma_set = [0.9,0.9,0.9,0.9]
    return beta_set, nu_set, rho_set, gamma_set


"""
A general test parametrization
"""
def get_test_set():
    beta_set = [0.2,0.4,0.6,0.8]
    nu_set = [0.3,0.3,0.3,0.3,0.3]
    rho_set = [-0.3,-0.3,-0.3,-0.3]
    gamma_set = [0.9,0.9,0.9,0.9]
    return beta_set, nu_set, rho_set, gamma_set


"""
Return a ZABR model with parameters in accordance to Figure 5
"""
def set_ZABR_worst_case_parameters():
    # Set ZABR Parameter
    params = Params()
    params.model = "zabr"
    params.forward = 0.00892529785633087
    params.beta = 0.569701285123825
    params.alpha = 0.00201458430290222
    params.nu =  0.509256491661072
    params.rho = -0.0525886398553849
    params.displacement = 0.00961106300354004
    params.gamma = 0.267791885137558
    return params


#%% Specifications for the IR-Example


"""
Return a ZABR model with parameters in accordance to Table 9 and 10
"""
def set_ZABR_parameters_1Y2Y(version):
    params_zabr = Params()
    params_zabr.model = "zabr"
    params_zabr.forward = 0.003
    params_zabr.displacement = 0.002
    params_zabr.alpha = 0.3*params_zabr.forward 
    params_zabr.nu =  0.3
    params_zabr.rho = -0.5
    
    beta, gamma = get_beta_gamma(version)
    params_zabr.beta = beta
    params_zabr.gamma = gamma
    
    return params_zabr


"""
Return a ZABR model with parameters in accordance to Table 9 and 10
"""
def set_ZABR_parameters_1Y5Y(version):
    params_zabr = Params()
    params_zabr.model = "zabr"
    params_zabr.forward = 0.005
    params_zabr.displacement = 0.002
    params_zabr.alpha = 0.3*params_zabr.forward 
    params_zabr.nu =  0.3
    params_zabr.rho = -0.7
    
    beta, gamma = get_beta_gamma(version)
    params_zabr.beta = beta
    params_zabr.gamma = gamma
    
    return params_zabr


"""
Get the specifications in accordance to Tables 10
"""
def get_beta_gamma(version):
    if(version=="v1"):
        return 0.4, 0.8
    elif(version=="v2"):
        return 0.4, 0.9
    elif(version=="v3"):
        return 0.5, 0.8


"""
Return a nSABR model with parameters in accordance to Table 9 and 10
"""
def set_market_parameters_1Y2Y():
    params_market = Params()
    params_market.model = "sabr"
    params_market.forward = 0.003
    params_market.beta = 0.
    params_market.alpha = 0.3*params_market.forward 
    params_market.nu =  0.3
    params_market.rho = -0.5
    params_market.displacement = 0.
    return params_market


"""
Return a nSABR model with parameters in accordance to Table 9 and 10
"""
def set_market_parameters_1Y5Y():
    params_market = Params()
    params_market.model = "sabr"
    params_market.forward = 0.005
    params_market.beta = 0.
    params_market.alpha = 0.3*params_market.forward 
    params_market.nu =  0.3
    params_market.rho = -0.7
    params_market.displacement = 0.
    return params_market


#%% Specifications for the SLV Example
"""
Return a ZABR model characterizing a market test surface
"""
def set_ZABR_standard_parameters_slv_surface():
    params_market = Params()
    params_market.model = "zabr"
    params_market.forward = 0.005
    params_market.beta = 0.4
    params_market.gamma = 0.9
    params_market.alpha = 0.3*params_market.forward**(1-params_market.beta)
    params_market.nu =  0.3
    params_market.rho = -0.7
    params_market.displacement = 0.0011
    return params_market

