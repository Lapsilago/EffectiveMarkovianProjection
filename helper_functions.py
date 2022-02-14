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
from Models.sabr import sabr_functions as sf
from Models.zabr import zabr_functions as zf
from Pricing.Hagan_1D_PDE import hagan_pde_density as hagan_densmaker
from Pricing.Hagan_1D_PDE import hagan_density_pricing as hagan_pricer


"""
Function computes the implied volatility using the 1d-PDE scheme for the 
SABR and ZABR model as possible input models.
"""
def compute_hagan_pde_iv_curve(T, strikes, nsteps, tsteps, nd, params):
    # density computation
    if params.model == 'zabr':
        Q_current_vec, ymin, ymax, hh = hagan_densmaker.create_density_ls_scheme_adaptive(
                strikes,zf.fkm_func_locvol_haganscheme, T, nsteps, tsteps, nd, params)
    elif params.model == 'sabr':
        Q_current_vec, ymin, ymax, hh = hagan_densmaker.create_density_ls_scheme_adaptive(
                strikes,sf.fkm_func_locvol_haganscheme, T, nsteps, tsteps, nd, params)
    
    # compute price and implied volatility
    pval_FKM_Hagan_1D_PDE = hagan_pricer.compute_call_price(strikes, ymin, ymax,
                                                        Q_current_vec, nsteps,
                                                        hh, params)
    iv_FKM_Hagan_1D_PDE = vb.vol_bachelier_diff_prices(1, strikes, params.forward,
                                                     T, pval_FKM_Hagan_1D_PDE)
    
    return iv_FKM_Hagan_1D_PDE

