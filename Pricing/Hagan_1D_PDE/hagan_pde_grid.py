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


import numpy as np


"""
Transforamtion of the PDE based on the Hagan 1D PDE Numerics
Construction of the Grid
Adaptive adjustment of the grid based on the strikes under consideration is applied
"""
def make_grid_adaptive(strikes,nd,timet,nsteps,params):
    #handle the case where stike is a single float-> transform to array
    if isinstance(strikes,float):
        kfloat = strikes
        strikes = np.zeros(1)
        strikes[0] = kfloat
    
    #function import dependends on model
    importlib = __import__('importlib')
    sf = importlib.import_module("Models." + params.model + "." + params.model + "_functions")
    
    #barrier of the model: holds the smallest number y can reach
    zbar = sf.func_z_F(-params.displacement,params)
    ybar = sf.func_y_z(zbar,params)
    
    #grid boundarys
    ymin = -nd * np.sqrt(timet)
    ymax = -ymin
    
    # adapt this value to be compliant with the strike value
    mstrike = np.max(strikes)
    zstrike = sf.func_z_F(mstrike,params)
    ystrike = sf.func_y_z(zstrike,params)
    
    if ymax <= ystrike: ymax = ystrike
    if ybar > ymin: ymin = ybar
    
    #uniform grid in y
    h0 = (ymax - ymin) / (nsteps-2)    # h0 is the step size having n-2 steps between ymin and ymax
    j0 = int((-ymin) / h0 + 0.5)       # j0 is index such that F_j0 <= forward <= F_j0+1 (+0.5 is to get usual rounding in Python)
    hh = ( -ymin) / (j0 - 0.5)        # hh is the grid shift distance
    y_vec = np.ones(nsteps)
    y_vec[0] = 0
    y_vec = np.cumsum(y_vec) * hh + ymin
    
    #Middle points for y
    ym_vec = y_vec - 0.5*hh
    
    #transform back to z
    zm_vec = sf.func_z_y(ym_vec,params)
    
    #transform back to F
    Fm_vec = sf.func_F_z(zm_vec,params)
    
    #transform boundaries
    ymax = y_vec[nsteps-1]
    zmax = sf.func_z_y(ymax,params); zmin = sf.func_z_y(ymin,params)
    Fmax = sf.func_F_z(zmax,params); Fmin = sf.func_F_z(zmin,params)
    
    #fill in first and last middle point
    Fm_vec[0] = 2 * Fmin - Fm_vec[1]
    Fm_vec[nsteps-1] = 2 * Fmax - Fm_vec[nsteps - 2]
    
    
    return ymin, ymax, hh, j0, ym_vec, Fm_vec

