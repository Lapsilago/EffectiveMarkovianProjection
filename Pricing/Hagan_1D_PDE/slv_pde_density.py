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
This class constructs a probability density for a local volatility model based on
the Hagan 1D-PDE scheme adjusted to a slv-setup.
"""


import numpy as np

from Pricing.Hagan_1D_PDE import slv_pde_grid as gridmaker
from PDE_Methods.Finite_Differences import FD_mat_factory as matmaker
from PDE_Methods.Finite_Differences import FD_mat_modification as matmodifier
from PDE_Methods.Finite_Differences import FD_1D_schemes as fd

"""
The function solves the probability density through the hagan 1D PDE scheme
with the lawson swayne method.
Save QL and QR in Q_current_vec[0] and Q_current_vec[nsteps-1]
Adjustments for a piecewise constant slv-setup in accordance to the paper
"Effective Stochastic Local Volatility Models" are applied
"""
def create_density_ls_scheme_adaptive_slv(kval, locvol_func, T, nsteps, tsteps, nd, params):
    
    #function import depends on model
    importlib = __import__('importlib')
    sf = importlib.import_module("Models." + params.model + "." + params.model + "_functions")
    
    # find the boundary F values based on the first time step
    F_min, F_max = gridmaker.find_boundary_first(kval,nd,T,nsteps,params)
    
    # create density vector
    Q_current_vec = np.zeros(nsteps)
    
    # create vector for additional information
    all_grid_values = np.zeros((tsteps,nsteps))
    all_hh_values = np.zeros((tsteps,1))
    all_theta_values = np.zeros((tsteps,nsteps))
    
    # Lawson-Swayne specific times on which the locvol function is evaluated
    dt = T/tsteps
    sqrt2 = np.sqrt(2.)
    B = 1. - sqrt2 * 0.5
    dt1 = dt * B
    
    # running time interval index
    t_index_old = 0
    
    # time marching starts here
    for it in range(0,tsteps):
        
        # current step
        t1 = it*dt 
        
        # find time regime
        t1_index = sf.index_search_time(t1, params.sigma_grid)
        
        # first interval
        if it == 0:
            
            # create the grid 
            ymin, ymax, hh, j0, ym_vec, Fm_vec = gridmaker.make_grid_slv(F_min, F_max,
                                                                         t1_index, nsteps,
                                                                         params)
            all_grid_values[0,:] = Fm_vec
            all_hh_values[0,0] = hh
            
            # initial density function
            Q_current_vec[j0] = 1 / hh
            all_theta_values[0,:] = Q_current_vec
            
        else:
            # change of time intervals, compute the new grid
            if t1_index > t_index_old:
                
                # create the grid valid in the current time regime
                ymin, ymax, hh, j0, ym_vec, Fm_vec =  gridmaker.make_grid_slv(F_min, F_max,
                                                                             t1_index, nsteps,
                                                                             params)
                all_grid_values[it, :] = Fm_vec
                all_hh_values[it, 0] = hh
                
                # transform density to the new grid 
                Q_current_vec = adjust_density_to_grid(Q_current_vec, nsteps,
                                                       hh, all_hh_values[it-1,:],
                                                       Fm_vec, all_grid_values[it-1,:])
            else:
                # grid already computed
                all_grid_values[it, :] = all_grid_values[it-1, :]
                all_hh_values[it, 0] = all_hh_values[it-1, 0]
        
        # values of the locvol function at the LS-intermediate timesteps
        locvol_vec_1 = locvol_func(ym_vec, Fm_vec, t1 + dt1, j0, nsteps, hh, params)
        locvol_vec_2 = locvol_func(ym_vec, Fm_vec, t1 + 2*dt1, j0, nsteps, hh, params)
        
        # construct the corresponding discrete FD operator matrices 
        A_1 = matmaker.make_A_hagan_one_dim_locvol(nsteps, Fm_vec, locvol_vec_1)
        A_2 = matmaker.make_A_hagan_one_dim_locvol(nsteps, Fm_vec, locvol_vec_2)
        
        # impose mirror boundary conditions on A
        A_1_wboundary = matmodifier.impose_mirror_boundary_conditions(nsteps,A_1,hh)
        A_2_wboundary = matmodifier.impose_mirror_boundary_conditions(nsteps,A_2,hh)
        
        # Lawson-Swayne timemarching
        Q_current_vec ,Q_1, Q_2 = fd.LS_step_time_dependent(nsteps,Q_current_vec,
                                                           dt/(2*hh),A_1_wboundary,
                                                           A_2_wboundary,0,0)
        all_theta_values[it, :] = Q_current_vec
        
        # save the index of the current time regime
        t_index_old = t1_index
    
    return Q_current_vec, ymin, ymax, hh


"""
Adjustment of the density function from one grid to the other
"""
def adjust_density_to_grid(Q_current_vec, nsteps, hh_new, hh_old, Fm_new, Fm_old):
    
    # new density function
    Q_current_vec_new = np.array(Q_current_vec)
    m = len(Fm_old)
    
    # differences of the old grid
    delta_F_old = np.ones(m)
    delta_F_old[1:m] = Fm_old[1:m]-Fm_old[0:(m-1)]
    
    for j in np.arange(1,nsteps-1):
        F_new_j = Fm_new[j]
        F_new_jm1 = Fm_new[j-1]
        
        # piecewise minimum
        F_min_j = np.array(Fm_old)
        F_min_j[F_min_j > F_new_j] = F_new_j
        
        # piecewise maximum
        F_max_jm1 = np.array(Fm_old)
        F_max_jm1[F_max_jm1 < F_new_jm1] = F_new_jm1
        
        diff = np.zeros(m)
        diff[1:m] = F_min_j[1:m] - F_max_jm1[0:(m-1)]
        diff[diff<0] = 0
        
        # density adjustment
        Q_current_vec_new[j] = np.sum(Q_current_vec*diff/delta_F_old)* hh_old/hh_new
        
    return Q_current_vec_new