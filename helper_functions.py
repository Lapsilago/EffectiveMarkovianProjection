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
from Helper.Normal_Vol import volBachelier as vb 
from Models.sabr import sabr_functions as sf
from Models.slvsabr import slvsabr_functions as sf_slv
from Models.zabr import zabr_functions as zf
from Pricing.Hagan_1D_PDE import hagan_pde_density as hagan_densmaker
from Pricing.Hagan_1D_PDE import hagan_density_pricing as hagan_pricer
from Pricing.Hagan_1D_PDE import slv_pde_density as slv_densmaker
from Pricing.Hagan_1D_PDE import slv_density_pricing as slv_pricer

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


"""
Function computes the implied volatility using the 1d-PDE scheme in a slv setup 
for the slv-SABR and ZABR model as possible input models.
"""
def compute_hagan_pde_iv_slv_setup(T, strikes, nsteps, tsteps, nd, params):
    # density(theta) computation
    if params.model == 'slvsabr':
        Q_current_vec, ymin, ymax, hh = slv_densmaker.create_density_ls_scheme_adaptive_slv(
                strikes,sf_slv.fkm_func_locvol_haganscheme, T, nsteps, tsteps, nd, params)
        
        # find time regime
        T_index = sf_slv.index_search_time(T, params.sigma_grid)
        if T in params.sigma_grid:
            T_index = T_index-1
        
        # compute prices
        pval_FKM_Hagan_1D_PDE = slv_pricer.compute_call_price(strikes, ymin, ymax,
                                                            Q_current_vec, nsteps,
                                                            hh, T_index, params)
        
    elif params.model == 'zabr':
        Q_current_vec, ymin, ymax, hh = hagan_densmaker.create_density_ls_scheme_adaptive(
                strikes,zf.fkm_func_locvol_haganscheme_emp_gamma, T, nsteps, tsteps, nd, params)
        
        # compute prices
        pval_FKM_Hagan_1D_PDE = hagan_pricer.compute_call_price(strikes, ymin, ymax,
                                                            Q_current_vec, nsteps,
                                                            hh, params)
    
    # compute implied vol
    iv_FKM_Hagan_1D_PDE = vb.vol_bachelier_diff_prices(1, strikes, params.forward,
                                                       T, pval_FKM_Hagan_1D_PDE)
    
    return iv_FKM_Hagan_1D_PDE


"""
This function gets the index m s.t: t \in J_m
The form of J_m is presented in the paper 
"Effective Stochastic Local Volatility Models"
"""
def index_search_time(x,grid):
    if isinstance(x,float) or isinstance(x,int):
        xfloat = x
        x = np.zeros(1)
        x[0] = xfloat
    
    # rounding to 10 decimals for all values is applied
    x = np.around(x,decimals = 10) 
    grid = np.around(grid, decimals = 10)
    
    # output
    x_index = np.searchsorted(grid,x,side='right')-1
    return x_index


"""
This function evaluates the projected volatility function on a full surface for
the slv-nSABR
"""
def compute_slv_locvol_surface(time_grid,strike_grid,params_slvnSABR):
    
    # define the local volatility surface
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
