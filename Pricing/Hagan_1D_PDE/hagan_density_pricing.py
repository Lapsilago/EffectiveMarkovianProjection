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


"""
Call price with density which has dirac masses saved in 0 and nsteps-1
"""
def compute_call_price(k, ymin, ymax, PP_vec, nsteps, hh, params):
    # transform k into an array if a float is given
    if isinstance(k,float):
        kfloat = k
        k = np.zeros(1)
        k[0] = kfloat
    p = np.zeros(len(k))
    i = 0
    
    #function import dependends on model
    importlib = __import__('importlib')
    sf = importlib.import_module("Models." + params.model + "." +
                                 params.model + "_functions")
    
    for strike in k:
        #transform strike into the grid setting
        zstrike = sf.func_z_F(strike,params)
        ystrike = sf.func_y_z(zstrike,params)  
        
        # call pricing starts here
        if ystrike <= ymin:
            p[i] = params.forward - strike
        elif ystrike >= ymax:
            # The strike is far in the right tail and the density is effectively 0
            p[i] = 0.
            break
        else:
            # Get the maximum value of F and weigh it with the right delta mass  
            Fmax = sf.func_F_z(sf.func_z_y(ymax,params),params)
            p[i] = (Fmax - strike) * PP_vec[nsteps-1]
            
            #in between values
            for K in range(nsteps - 2,0,-1):
                ym1 = ymin + (K - 0.5) * hh 
                ft1 = sf.func_F_z(sf.func_z_y(ym1,params),params) 
                if ft1 > strike:
                    p[i] = p[i] + (ft1 - strike) * PP_vec[K] * hh
                else:
                    break
            
            # Now K is the value where the payoff is zero and at K+1 the payoff is positive
            # Calculating the value for subgridscale
            # last admissible value ft>strike
            ymK = ymin + K * hh
            # first value with ft<strike
            ymKm1 = ymin + (K - 1) * hh
            
            ftK = sf.func_F_z(sf.func_z_y(ymK,params),params)
            ftKm1 = sf.func_F_z(sf.func_z_y(ymKm1,params),params)
            
            diff = ftK - ftKm1
            b = (2 * ft1 - ftKm1 - ftK) / diff
            subgridadjust = 0.5 * hh * PP_vec[K] * (ftK - strike) ** 2 / diff \
                    * (1 + b * (ftK + 2 * strike - 3 * ftKm1) / diff)
            p[i] = p[i] + subgridadjust
            
            # check on validiy
            if params.forward > strike:
                intrinsic_value = params.forward-strike
            else:
                intrinsic_value = 0.0
            
            if strike == -params.displacement:
                    p[i] = intrinsic_value
            else:
                if p[i] <= intrinsic_value:
                    p[i] = intrinsic_value + 1.0e-10
            
        i = i+1
    return p 

