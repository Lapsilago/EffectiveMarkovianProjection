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
the Hagan 1D-PDE scheme.
"""


import numpy as np

from Pricing.Hagan_1D_PDE import hagan_pde_grid as gridmaker
from PDE_Methods.Finite_Differences import FD_mat_factory as matmaker
from PDE_Methods.Finite_Differences import FD_mat_modification as matmodifier
from PDE_Methods.Finite_Differences import FD_1D_schemes as fd



"""
The function solves the probability density through the hagan 1D PDE scheme
with the lawson swayne method.
Save QL and QR in Q_current_vec[0] and Q_current_vec[nsteps-1]
"""
def create_density_ls_scheme_adaptive(kval, locvol_func, T, nsteps, tsteps, nd, params):
    
    # create the grid proposed in the hagan scheme
    ymin, ymax, hh, j0, ym_vec, Fm_vec =  gridmaker.make_grid_adaptive(kval,nd,T,nsteps,params)
    
    # create initial density
    Q_current_vec = np.zeros(nsteps)
    Q_current_vec[j0] = 1 / hh
    
    # Lawson-Swayne specific times on which the locvol function is evaluated
    dt = T/tsteps
    sqrt2 = np.sqrt(2.)
    B = 1. - sqrt2 * 0.5
    dt1 = dt * B
    
    # time marching starts here
    for it in range(0,tsteps):
        
        # current step
        t1 = it*dt 
        
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
        Q_current_vec,Q_1,Q_2 = fd.LS_step_time_dependent(nsteps,Q_current_vec,dt/(2*hh),A_1_wboundary,A_2_wboundary,0,0)
        
    return Q_current_vec, ymin, ymax, hh

