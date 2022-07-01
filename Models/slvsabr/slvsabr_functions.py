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
from Models.sabr import sabr_functions as sf_base


#%% SLV interval indexing functions
"""
These are functions establishing the piecewise constant interval definition
for the leverage functions
"""


"""
This function gets the index n s.t: x \in I_n
The form of I_n is presented in the paper 
"Effective Stochastic Local Volatility Models"
"""
def index_search_space(x,grid,params):
    if isinstance(x,float) or isinstance(x,int):
        xfloat = x
        x = np.zeros(1)
        x[0] = xfloat
    
    # rounding to 10 decimals for all values is applied
    x = np.around(x,decimals = 10) 
    grid = np.around(grid, decimals = 10)
    
    # output
    x_index = np.zeros(len(x))
    
    # find index_i
    i_index = np.searchsorted(grid,params.forward,side='left')
    
    # define the different regions
    region_left = (x < grid[i_index])    # up to K_{i-1}
    region_right = (grid[i_index] < x)    # from K_{i+1}
    region_atm = (grid[i_index-1]< x) & (x < grid[i_index+1])
    
    # region left
    x_index[region_left] =  np.searchsorted(grid,x[region_left],side='left')
    
    # region right
    x_index[region_right] =  np.searchsorted(grid,x[region_right],side='right')-1
    
    # region atm
    x_index[region_atm] = i_index
    
    # return an integer for the index value
    return np.rint(x_index).astype(int)


"""
This function gets the index m s.t: x \in J_m = [t_m , t_{m+1})
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
    
    # return an integer
    return np.rint(x_index).astype(int)


#%% Leverage function L()
"""
Specifications for the leverage function
"""


"""
This function returns the value L() for the corresponding index
"""
def func_L_based_on_index(x_index,params):
    return params.L_val[np.rint(x_index).astype(int)]


"""
Leverage function L(x) evaluated for an array of values x
"""
def func_L(x,params):
    if isinstance(x,float) or isinstance(x,int):
        xfloat = x
        x = np.zeros(1)
        x[0] = xfloat
    
    x_index = index_search_space(x,params.L_grid,params)
    x_l_val = func_L_based_on_index(x_index,params)
    return x_l_val


#%% Backbone function C()
"""
These are general functions specifying the model backbone C()
"""



"""
Displaced SABR Backbone: Function C_base(F)
"""
def func_C_base(x,params):
    return sf_base.func_C(x,params)


"""
Backbone: Function C(F)
"""
def func_C(x,params):
    val_C_base = func_C_base(x,params)
    val_L = func_L(x,params)
    return val_C_base*val_L


"""
Approximation function for C(x)'
Output is constantly C(f)' to guarantee differentiability
"""
def func_Gamma(x,params):
    if isinstance(x,float) or isinstance(x,int):
        xfloat = x
        x = np.zeros(1)
        x[0] = xfloat
    
    out = np.array(x)
    out = func_L(params.forward,params)*params.beta/(
            (params.forward + params.displacement)**(1-params.beta))
    return out


#%% EMP coefficients for the SLV-model
"""
These are general functions specifying the model EMP coefficients
"""


"""
This function initializes all relevant EMP coefficients on the slv time grid
"""
def initialize_all_emp_coeff(params):
    # initialize deltas on time grid 
    params.sigma_grid_delta = initialize_sigma_grid_delta(params)
    
    #initailze theta functions
    params.theta_0_on_grid, params.theta_1_on_grid, params.theta_2_on_grid = initialize_theta_functions(params)
    
    #initialize Sigma functions
    params.Sigma_0_2_on_grid, params.Sigma_1_2_on_grid, params.Sigma_1_1_on_grid = initialize_Sigma_functions(params)
    
    #initialize a coefficient
    params.emp_a_on_grid = emp_coeff_a_on_time_grid(params)
    
    #initialize b coefficient
    params.emp_b_on_grid = emp_coeff_b_on_time_grid(params)
    
    #initialize c coefficient
    params.emp_c_on_grid = emp_coeff_c_on_time_grid(params)
    
    #initialize G coefficient
    params.emp_G_on_grid = emp_coeff_G_on_time_grid(params)
    
    return params


"""
This function computes the differences of the time grid: delta_m = t_{m+1}-t_m
"""
def initialize_sigma_grid_delta(params):
    # lenght of the time grid
    n = len(params.sigma_grid)
    
    # upper values t_{m+1}
    sigma_up = params.sigma_grid[1:n]
    
    # lower values t_m
    sigma_down = params.sigma_grid[0:(n-1)]
    return sigma_up-sigma_down


"""
This function initalizes the relevant theta functions on the time grid, i.e
in the notation of the paper: theta(i,0,t_j)
"""
def initialize_theta_functions(params):
    # only the index up to 2 is relevant
    theta_0 = emp_theta_on_time_grid(0, params)
    theta_1 = emp_theta_on_time_grid(1, params)
    theta_2 = emp_theta_on_time_grid(2, params)
    return theta_0, theta_1, theta_2


"""
The i-theta function in the special case where t0 = 0 and T = t_j
"""
def emp_theta_on_time_grid(i, params):
    # the values sigma_m, the last value is not needed for the grid computation
    sigma = params.sigma_val[:-1]
    
    # the values delta_m
    delta = params.sigma_grid_delta
    
    # the product depending on the index i
    prod_i = sigma**i * delta
    
    # create the theta function
    theta_i = np.zeros(len(prod_i)+1)
    
    # calculate the values for j>0
    theta_i[1:] = np.cumsum(prod_i)
    
    # the value for t0 is approximated with the right limit
    theta_i[0] = theta_i[1]
    return theta_i


"""
This function initalizes the relevant Sigma functions on the time grid, i.e
in the notation of the paper: Sigma(i,k,0,t_j)
"""
def initialize_Sigma_functions(params):
    # only the relevant indices are considered
    Sigma_0_2 = emp_Sigma_on_time_grid(0, 2, params)
    Sigma_1_2 = emp_Sigma_on_time_grid(1, 2, params)
    Sigma_1_1 = emp_Sigma_on_time_grid(1, 1, params)
    return Sigma_0_2, Sigma_1_2, Sigma_1_1


"""
The i-k-Sigma function in the special case where t0 = 0 and T = t_j
"""
def emp_Sigma_on_time_grid(i, k, params):
    # the values sigma_m, the last value is not needed for the grid computation
    sigma = params.sigma_val[:-1]
    
    # the values delta_m
    delta = params.sigma_grid_delta
    
    # the k-th and i-th product
    prod_k = sigma**k * delta
    prod_i = sigma**i * delta
    
    # the first sum
    sum_1 = 0.5*np.cumsum(sigma**(i+k)*delta**2)
    
    # the second sum
    sum_2 = np.cumsum(prod_i)*np.cumsum(prod_k)
    
    # the third sum
    sum_3 = np.cumsum(np.cumsum(prod_k) * prod_i)
    
    # create the Sigma function
    Sigma_i_k = np.zeros(len(prod_i)+1)
    
    # calculate the values for j>1
    Sigma_i_k[1:] = sum_1 + sum_2 - sum_3
    
    # the value for j=1 is adjusted since no mixed terms appear here
    Sigma_i_k[1] = sum_1[0]
    
    # the value for t0 is approximated with the right limit
    Sigma_i_k[0] = Sigma_i_k[1]
    return Sigma_i_k


"""
The function a(t) = alpha
"""
def emp_coeff_a_on_time_grid(params):
    return np.full(len(params.sigma_grid),params.alpha)


"""
The function b(t)
"""
def emp_coeff_b_on_time_grid(params):
    # load the necessary theta functions
    theta_1 = params.theta_1_on_grid
    theta_2 = params.theta_2_on_grid
    
    return  params.rho * params.nu * theta_1 / (params.alpha * theta_2)


"""
The function c(t)
"""
def emp_coeff_c_on_time_grid(params):
    # load the necessary theta functions
    theta_1 = params.theta_1_on_grid
    theta_2 = params.theta_2_on_grid
    
    # load the necessary Sigma-i-k functions
    Sigma_0_2 = params.Sigma_0_2_on_grid
    Sigma_1_2 = params.Sigma_1_2_on_grid
    Sigma_1_1 = params.Sigma_1_1_on_grid
    
    # Compute the various quotients
    term1 = Sigma_0_2 /theta_2**2
    term2 = theta_1**2/theta_2**2
    term3 = theta_1*Sigma_1_2/theta_2**3
    term4 = Sigma_1_1/theta_2**2
    
    return params.nu**2/ params.alpha**2 * (2*term1 + params.rho**2 *(term2 - 6* term3 +4*term4))


"""
The function G(t)
"""
def emp_coeff_G_on_time_grid(params):
    # load the necessary theta functions
    theta_0 = params.theta_0_on_grid
    theta_2 = params.theta_2_on_grid
    
    # load the already computed c function
    emp_c = params.emp_c_on_grid
    
    return params.nu**2*theta_0 - params.alpha**2*theta_2*emp_c


#%% Felpel, Kienitz and McWalter Coefficients
"""
These are the cofficients for the local volatility approximation based on
Felpel,Kienitz and McWalter
"""


"""
Function D(F) defined as in the Hagan 1D Scheme:
This is a different function than the function D(T,F) defined in Felpel, Kienitz and McWalter
D(T,F)_{FKM} = 0.5*D(F)**2*E(T,F)
"""
def fkm_func_D(T, z, F, params):
    # find the corresponding time index which is relevant 
    T_index = index_search_time(T,params.sigma_grid)
    
    # load the emp coefficients for the time grid
    # with the approximation coeff(t) = coeff(t_n)
    a_coef = params.emp_a_on_grid[T_index]
    b_coef = params.emp_b_on_grid[T_index]
    c_coef = params.emp_c_on_grid[T_index]
    
    # load the sigma value
    sigma_val = params.sigma_val[T_index]
    
    # return the polynomial form
    return sigma_val * a_coef * np.sqrt(1 + 2*b_coef*z + c_coef*z**2)*func_C(F,params)


"""
Function E(T,F) defined as in the Hagan 1D Scheme
"""
def fkm_func_E(T,F,params):
    # find the corresponding time index which is relevant
    T_index = index_search_time(T,params.sigma_grid)
    
    # load the emp coefficients for the time grid
    # with the approximation coeff(t) = coeff(t_n)
    Gval = params.emp_G_on_grid[T_index]
    
    return np.exp(Gval)


"""
Local Volatility Function based on the FKM Approximation
"""
def fkm_func_locvol(T,F,params):
    # transform the variable
    zval = func_z_F(F,params)
    
    # compute the local vol used for the Hagan 1d PDE scheme
    Dval = fkm_func_D(T, zval, F, params)
    Eval = fkm_func_E(T, F, params)
    return Dval*np.sqrt(Eval)


#%% 1D-PDE function
"""
These are the transformations used for the 1D PDE scheme
"""

"""
This function is the index function M(n_F)
"""
def compute_M_SLV(n_index_F,forward_index,z_val_noSLV_grid,params):
    if isinstance(n_index_F,float) or isinstance(n_index_F,int):
        zfloat = n_index_F
        n_index_F = np.zeros(1)
        n_index_F[0] = zfloat
    
    M_val = np.zeros(len(n_index_F))
    for index in np.arange(0,len(n_index_F)):
        index_val = n_index_F[index]
        
        # ATM no adjustment is performed
        if(np.abs(index_val-forward_index)<0.00000000001):
            M_val[index] = 0
        
        # recursive evaluation on the grid starting from ATM
        elif(index_val < forward_index):
            for j in np.arange(index_val,forward_index):
                factor_temp = (1/params.L_val[int(np.round(j+1))]-1/params.L_val[int(np.round(j))])
                M_val[index] = M_val[index] + factor_temp * z_val_noSLV_grid[int(np.round(j))]
        else:
            for j in np.arange(forward_index+1,index_val+1):
                factor_temp = (1/params.L_val[int(np.round(j-1))]-1/params.L_val[int(np.round(j))])
                M_val[index] = M_val[index] + factor_temp*z_val_noSLV_grid[int(np.round(j))]
    return M_val


"""
The function evaluates z(F) = int_f^F\frac{1}{L(u)C(u)}du
"""
def func_z_F(F,params):
    if isinstance(F,float):
        Ffloat = F
        F = np.zeros(1)
        F[0] = Ffloat
    
    L_grid = params.L_grid
    
    # the ATM index i 
    forward_index = index_search_space(params.forward,params.L_grid,params)
    
    # indizes n_F
    n_F_index = index_search_space(F,params.L_grid,params)
    
    # values z_base(K_i)
    z_val_noSLV_L_grid = sf_base.func_z_F(L_grid,params)
    
    # values z_base(F)
    z_val_noSLV = sf_base.func_z_F(F,params)
    
    # values M(n_F)
    M_val = compute_M_SLV(n_F_index,forward_index,z_val_noSLV_L_grid,params)
    
    # values l_{n_F}
    F_L_val = func_L_based_on_index(n_F_index,params)
    return z_val_noSLV/F_L_val + M_val


"""
The inverse transformation to get F out of z
"""
def func_F_z(zm, params):
    if isinstance(zm,float):
        kfloat = zm
        zm = np.zeros(1)
        zm[0] = kfloat
    
    # the ATM index i 
    forward_index = index_search_space(params.forward, params.L_grid,params)
    
    # compute index n_z
    n_z_index = index_search_space(zm, func_z_F(params.L_grid,params),params)
    
    # values z_base(K_i)
    z_val_noSLV_L_grid = sf_base.func_z_F(params.L_grid,params)
    
    # values M(n_z)
    M_val = compute_M_SLV(n_z_index,forward_index,z_val_noSLV_L_grid,params)
    
    # values l_{n_F}
    z_L_val = func_L_based_on_index(n_z_index,params)
        
    # adjustet zm
    zm_adj = z_L_val*(zm - M_val)
    
    return func_F_z_base(zm_adj,params)


"""
The inverse transformation to get F out of z for the underlying base model
"""
def func_F_z_base(zm, params):
    return sf_base.func_F_z(zm,params)


"""
The function evaluates y(F) = int_f^F\frac{1}{D(z(F))}du for given z 
"""
def func_y_z(zm, T_index, params):
    # load the EMP coefficients
    a_coef = params.emp_a_on_grid[T_index]
    b_coef = params.emp_b_on_grid[T_index]
    c_coef = params.emp_c_on_grid[T_index]
    sig_coef = params.sigma_val[T_index]
    
    # compute the transformation 
    w = np.sqrt(c_coef)
    d = b_coef/w
    M = w*zm + d
    term1 = np.sqrt(1-d**2+M**2) - M
    term2 = 1-d
    yz = -1 / (sig_coef*a_coef*w) * np.log(term1/term2)
    return yz


"""
The inverse transformation to get z out of y
"""
def func_z_y(ym, T_index, params):
    # load the EMP coefficients
    a_coef = params.emp_a_on_grid[T_index]
    b_coef = params.emp_b_on_grid[T_index]
    c_coef = params.emp_c_on_grid[T_index]
    sig_coef = params.sigma_val[T_index]
    
    # compute the transformation 
    w = np.sqrt(c_coef)
    d = b_coef/w
    term1 = sig_coef*a_coef*w 
    zy = np.sinh(term1 * ym) + d * (np.cosh(term1 * ym) - 1)
    return zy / w


"""
Expression Based on the locvol function used for the hagan 1D PDE:
Return D(F)E(T,F) 
Note that the locvol would correspond to D(F)sqrt(E(T,F))
"""
def fkm_func_locvol_haganscheme(y,F,T,j0,nsteps,hh,params):
    # get the D(F) term 
    Dval = np.zeros(nsteps)
    Dval[1:nsteps-1] = fkm_func_D(T, func_z_F(F[1:nsteps-1],params), F[1:nsteps-1], params)
    
    #Mirror Shadowpoints
    Dval[0] = Dval[1]
    Dval[nsteps-1] = Dval[nsteps - 2]
    
    # get the E(T,F) term
    Eval = np.ones(len(Dval))
    Eval[1:j0] = fkm_func_E(T,F[1:j0],params)
    Eval[j0+1:nsteps-1] = fkm_func_E(T,F[j0+1:nsteps-1],params)
    Eval[j0] = fkm_func_E(T,params.forward,params)
    
    return Dval*Eval


"""
Expression Based on the locvol function used for the hagan 1D PDE:
Return D(F)E(T,F) 
Note that the locvol would correspond to D(F)sqrt(E(T,F))
"""
def fkm_func_locvol_haganscheme_test(F,T,j0,nsteps,hh,params):
    Dval = np.zeros(nsteps)
    Dval[1:nsteps-1] = fkm_func_D(T, func_z_F(F[1:nsteps-1],params), F[1:nsteps-1], params)
    
    #Mirror Shadowpoints
    Dval[0] = Dval[1]
    Dval[nsteps-1] = Dval[nsteps - 2]
    
    Eval = np.ones(len(Dval))
    Eval[1:j0] = fkm_func_E(T,F[1:j0],params)
    Eval[j0+1:nsteps-1] = fkm_func_E(T,F[j0+1:nsteps-1],params)
    Eval[j0] = fkm_func_E(T,params.forward,params)
    
    return Dval**2*Eval

