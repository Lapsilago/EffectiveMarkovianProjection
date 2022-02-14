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


from Helper.Normal_Vol import volBachelier as vb 
from IR_Examples import IR_spread_options as spread_pricing


"""
Compute the implied volatility for a midcurve option in accordance to 
Effective Markovian Projection
"""
def compute_iv_original_rate(K,T,T_i,T_0,T_1,T_2,params):
    #compute convexity coeff
    lambda_i = get_convexity_coeff_rate(T_i,T_0,T_1,T_2)
    
    #compute adjusted strike
    K_adj = adjust_strike(K,lambda_i,params.forward)
    
    #vanilla call of the rate
    price_call, sig_N = spread_pricing.nSABR_vanilla_call(T,K_adj,params)
    
    #scale the call price
    Call = (1-lambda_i*K)*price_call
    
    #compute the iv
    iv_call = vb.vol_bachelier_diff_prices(1, K, params.forward, T, Call)
    return iv_call


"""
Approximation of the convexity coefficient in accordance to Effective Markovian Projection
"""
def get_convexity_coeff_rate(T_i,T_0,T_1,T_2):
    return 0.5*(T_0 + T_i - T_1 - T_2)


"""
Adjustment of the strike in accordance to Effective Markovian Projection, Equation 3.5
"""
def adjust_strike(K,lambda_i,forward_i):
    return K * (1-lambda_i*forward_i)/(1-lambda_i*K)

