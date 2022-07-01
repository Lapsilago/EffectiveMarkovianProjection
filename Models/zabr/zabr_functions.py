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
General Parameter describtion:
alpha - current volatility value
beta - cev coefficient
gamma - cev coefficient for the volatility
nu - vol of vol coefficient
rho - correlation coefficient
forward - current forward value
displacement - displacement coefficient
"""



#%% General SABR functions
"""
These are general functions specifying the model.
"""


"""
Displaced SABR Backbone: Function C(F)
"""
def func_C(x,params):
    return (x+params.displacement)**(params.beta)


"""
Approximation function for C(x)'
"""
def func_Gamma(x,params):
    if isinstance(x,float) or isinstance(x,int):
        xfloat = x
        x = np.zeros(1)
        x[0] = xfloat
    
    ind_forw = [x==params.forward]
    ind_not_forw = [x!=params.forward]
    
    out = np.array(x)
    out[tuple(ind_forw)] = params.beta/(x[tuple(ind_forw)] + params.displacement)**(1-params.beta)
    out[tuple(ind_not_forw)] = (func_C(x[tuple(ind_not_forw)],params)-func_C(params.forward,params))/(x[tuple(ind_not_forw)]-params.forward)
    return out


#%% Felpel, Kienitz and McWalter Coefficients
"""
These are the cofficients for the local volatility approximation based on
Felpel,Kienitz and McWalter
"""

"""
Polynomial Coefficient a()
"""
def fkm_coeff_a(params):
    return params.alpha**2


"""
Polynomial Coefficient b()
"""
def fkm_coeff_b(params):
    return params.rho * params.alpha**params.gamma * params.nu


"""
Polynomial Coefficient c()
"""
def fkm_coeff_c(params):
    return params.nu**2 * params.alpha**(2*(params.gamma-1))*(1+(params.gamma-1)*params.rho**2)


"""
Exponential Coefficient c()
"""
def fkm_coeff_G(t,Gamma_vec,params):
    term1 = params.rho * params.nu * params.alpha**params.gamma * Gamma_vec
    term2 = params.rho**2 * params.nu**2 * params.alpha**(2*(params.gamma-1))*(params.gamma-1)
    return  term1 - term2


"""
Function D(F) defined as in the Hagan 1D Scheme:
This is a different function than the function D(T,F) defined in Felpel, Kienitz and McWalter
D(T,F)_{FKM} = 0.5*D(F)**2*E(T,F)
"""
def fkm_func_D(z,F,params):
    return np.sqrt(fkm_coeff_a(params) + 2*fkm_coeff_b(params)*z + fkm_coeff_c(params)*z**2)*func_C(F,params)


"""
Function E(T,F) defined as in the Hagan 1D Scheme
"""
def fkm_func_E(T,F,params):
    Gamma = func_Gamma(F,params)
    Gval = fkm_coeff_G(T,Gamma,params)
    return np.exp(Gval*T)


#%% Hagan 1D-PDE function
"""
These are the transformations used for the Hagan 1D PDE scheme
"""

"""
Expression Based on the locvol function used for the hagan 1D PDE:
Return D(F)E(T,F) 
Note that the locvol would correspond to D(F)sqrt(E(T,F))
The Gamma function corresponds constantly to -C(f)'
"""
def fkm_func_locvol_haganscheme_emp_gamma(y,F,T,j0,nsteps,hh,params):
    Dval = np.zeros(nsteps)
    Dval[1:nsteps-1] = fkm_func_D(func_z_y(y[1:nsteps-1],params),F[1:nsteps-1],params)
    
    #Mirror Shadowpoints
    Dval[0] = Dval[1]
    Dval[nsteps-1] = Dval[nsteps - 2]
    
    Eval = np.ones(len(Dval))
    Eval[1:nsteps-1] = fkm_func_E(T,params.forward,params)
    
    return Dval*Eval


"""
Expression Based on the locvol function used for the hagan 1D PDE:
Return D(F)E(T,F) 
Note that the locvol would correspond to D(F)sqrt(E(T,F))
"""
def fkm_func_locvol_haganscheme(y,F,T,j0,nsteps,hh,params):
    Dval = np.zeros(nsteps)
    Dval[1:nsteps-1] = fkm_func_D(func_z_y(y[1:nsteps-1],params),F[1:nsteps-1],params)
    
    #Mirror Shadowpoints
    Dval[0] = Dval[1]
    Dval[nsteps-1] = Dval[nsteps - 2]
    
    Eval = np.ones(len(Dval))
    Eval[1:j0] = fkm_func_E(T,F[1:j0],params)
    Eval[j0+1:nsteps-1] = fkm_func_E(T,F[j0+1:nsteps-1],params)
    Eval[j0] = fkm_func_E(T,params.forward,params)
    
    return Dval*Eval


"""
The function evaluates z(F) = int_f^F\frac{1}{C(u)}du
"""
def func_z_F(F,params):
    if isinstance(F,float):
        Ffloat = F
        F = np.zeros(1)
        F[0] = Ffloat
    if params.beta == 1.:
    # log-normal case
        return  np.log((F+params.displacement)/(params.forward+params.displacement))
    elif params.beta == 0.:
    # normal case
        return F-params.forward
    else:
    # general cev case
        betam = 1 - params.beta
        return ((F+params.displacement)**betam - (params.forward+params.displacement)**betam) / betam


"""
The inverse transformation to get F out of z
"""
def func_F_z(zm, params):
    if isinstance(zm,float):
        kfloat = zm
        zm = np.zeros(1)
        zm[0] = kfloat
    if params.beta == 1.:
    # log-normal case
        par = np.log(params.forward+params.displacement)
        return np.exp(zm + par)-params.displacement
    elif params.beta == 0.:
    # normal case
        par = (params.forward+params.displacement) + zm
        return par - params.displacement
    else:
    # general cev case
        betm = 1 - params.beta
        par = (params.forward + params.displacement) ** betm + betm * zm
        par[par < 0.] = 0.
        return par ** (1 / betm)-params.displacement


"""
The function evaluates y(F) = int_f^F\frac{1}{D(z(F))}du for given z 
"""
def func_y_z(zm, params):
    gammam = params.gamma-1
    w = np.sqrt(1 + gammam * params.rho**2)
    c = params.nu * params.alpha**gammam * w
    L = params.rho/w + c * zm / params.alpha
    yz = -1 / c * np.log((np.sqrt(1 - (params.rho/w)**2 + L**2) - L) / (1 - params.rho/w))
    return yz


"""
The inverse transformation to get z out of y
"""
def func_z_y(ym, params):
    gammam = params.gamma-1
    w = np.sqrt(1 + gammam * params.rho**2)
    c = params.nu * params.alpha**gammam * w
    zy = np.sinh(c * ym) + params.rho/w * (np.cosh(c * ym) - 1)
    return params.alpha / c * zy

