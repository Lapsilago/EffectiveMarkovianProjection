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
Computation of the implied normal volatility in accordance to 
Le Floc’h, Fast and Accurate Analytic Basis Point Volatility
"""


import autograd.numpy as np
import math as m


"""
Implementation in accordance to 
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def vol_bachelier(sign, k, forward, t, price):
    # transform variables into arrays if necessary
    if isinstance(k,float):
        kfloat = k
        k = np.zeros(1)
        k[0] = kfloat
    if isinstance(forward,float):
        forwardfloat = forward
        forward = np.zeros(1)
        forward[0] = forwardfloat
    
    # four rational expansion
    aLFK4 = get_aLFK4()
    bLFK4 = get_bLFK4()
    cLFK4 = get_cLFK4()
    dLFK4 = get_dLFK4()
    
    # constants
    cut_off_atm = 1.0e-10 # ATM Cutoff level #np.finfo(float).eps
    cut_off_rounding = 0  #1.0e-50 # cut-off for rounding
    cut_off1 = 0.15       # cut-off for -C(x)/x
    cut_off2 = 0.0091     # 1st cut-off for tilde(eta)
    cut_off3 = 0.088      # 2nd cut-off for tilde(eta)
    
    x = (forward-k) * sign   # intrinsic value of the Call (sign=1)
                                         # or Put (sign = -1)
                          
    if any(price < x):
        print("Error in input data! Need price >= intrinsic value")
        print("price=", price, "intrinsic=", x, "diff=", price - x)
        return -1.
    
    Nk = len(k) 
    vol = np.zeros(Nk)               # returned by function   
    
    # identify ATM and non ATM strikes
    index_atm = np.abs(forward - k) < cut_off_atm  # atm strike, resp. close (cut off) to atm
    index_natm = np.abs(forward - k) >= cut_off_atm  # itm/otm strike, resp. not close (cut off) to atm
    
    #calculate ATM vol
    vol[index_atm]= price * np.sqrt (2. * m.pi / t )
    
    if any(index_natm):
        #computing -C(x)/x
        z = np.zeros(Nk)
        index_x_positiv = (x >= 0) 
        index_x_npositiv = (x<0)
        
        index_z_positiv = np.logical_and(index_x_positiv,index_natm)
        index_z_npositiv = np.logical_and(index_x_npositiv,index_natm)
        
        z[index_z_positiv] = (price - x[index_z_positiv])/x[index_z_positiv]
        z[index_z_npositiv] = -price / x[index_z_npositiv]
        
        #find ITM/OTM area     
        index_z_otm = np.logical_and(np.logical_and((cut_off_rounding < z),
                                                 (z<= cut_off1)),index_natm) 
        index_z_itm = np.logical_and((z>cut_off1),index_natm) 
        u = np.zeros(Nk)
        num = np.zeros(Nk)
        den    = np.zeros(Nk)
        if any(index_z_otm):
            #compute u 
            betaStart = - np.log(cut_off1)
            betaEnd = - np.log(np.finfo(float).tiny)
            u[index_z_otm] = -(np.log(z[index_z_otm])+ betaStart) /(betaEnd - betaStart)
            
            #cutoffs 
            index_u_cut2 = (u < cut_off2)
            index_u_cut3 = np.logical_and((u < cut_off3),(u >= cut_off2))
            index_u_ncut3 = (u >= cut_off3)
            
            if any(index_u_cut2):    
                num[index_u_cut2] = num_otm(u[index_u_cut2],bLFK4) 
                den[index_u_cut2] = den_otm(u[index_u_cut2],bLFK4)
                        
            if any(index_u_cut3):
                num[index_u_cut3] = num_otm(u[index_u_cut3],cLFK4)                 
                den[index_u_cut3] = den_otm(u[index_u_cut3],cLFK4) 
                
            if any(index_u_ncut3):
                num[index_u_ncut3] = num_otm(u[index_u_ncut3],dLFK4)  
                den[index_u_ncut3] = den_otm(u[index_u_ncut3],dLFK4) 
                
            vol[index_z_otm] = abs(x[index_z_otm])/(np.sqrt(num[index_z_otm]/den[index_z_otm] * t ))
            
        if any(index_z_itm):
            index_z_itm_pos = np.logical_and(index_z_itm,index_x_positiv)
            index_z_itm_npos = np.logical_and(index_z_itm,index_x_npositiv)
            
            if any(index_z_itm_pos):
                z[index_z_itm_pos] = np.abs(x[index_z_itm_pos]) / price
                u[index_z_itm_pos] = eta(z[index_z_itm_pos])
                num[index_z_itm_pos] = num_itm(u[index_z_itm_pos],aLFK4)                
                den[index_z_itm_pos] = den_itm(u[index_z_itm_pos],aLFK4)                
                vol[index_z_itm_pos] = price * num[index_z_itm_pos] / den[index_z_itm_pos] / np.sqrt(t)
            
            if any(index_z_itm_npos):
                z[index_z_itm_npos] = np.abs(x[index_z_itm_npos]) / (price- x[index_z_itm_npos])
                u[index_z_itm_npos] = eta(z[index_z_itm_npos])
                num[index_z_itm_npos] = num_itm(u[index_z_itm_npos],aLFK4)
                den[index_z_itm_npos] = den_itm(u[index_z_itm_npos],aLFK4)                
                vol[index_z_itm_npos] = (price- x[index_z_itm_npos]) * num[index_z_itm_npos] / den[index_z_itm_npos] / np.sqrt(t)  
    return vol


"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def get_aLFK4():
    return [0.06155371425063157,2.723711658728403,10.83806891491789,
             301.0827907126612,1082.864564205999, 790.7079667603721, 
             109.330638190985, 0.1515726686825187, 1.436062756519326, 
             118.6674859663193, 441.1914221318738, 313.4771127147156, 
             40.90187645954703]



"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
bLFK4 = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, d1, d2, d3, d4, d5, d6, d7]
"""
def get_bLFK4():
    return [0.6409168551974357, 788.5769356915809, 445231.8217873989, 
             149904950.4316367, 32696572166.83277, 4679633190389.852, 
             420159669603232.9, 2.053009222143781e+16, 3.434507977627372e+17, 
             2.012931197707014e+16, 644.3895239520736, 211503.4461395385, 
             42017301.42101825, 5311468782.258145, 411727826816.0715, 
             17013504968737.03, 247411313213747.3]


"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
cLFK4 = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, f1, f2, f3, f4, f5, f6, f7]
"""
def get_cLFK4():
    return [0.6421106629595358, 654.5620600001645, 291531.4455893533,
            69009535.38571493, 9248876215.120627, 479057753706.175, 
            9209341680288.471, 61502442378981.76, 107544991866857.5,
            63146430757.94501, 437.9924136164148, 90735.89146171122, 
            9217405.224889684, 400973228.1961834, 7020390994.356452, 
             44654661587.93606, 76248508709.85633]


"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
dLFK4 = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, h1, h2, h3, h4, h5, h6, h7]
"""
def get_dLFK4():
    return [0.936024443848096, 328.5399326371301, 177612.3643595535,
            8192571.038267588, 110475347.0617102, 545792367.0681282,
            1033254933.287134, 695066365.5403566, 123629089.1036043, 
            756.3653755877336, 173.9755977685531, 6591.71234898389, 
            82796.56941455391, 396398.9698566103, 739196.7396982114, 
             493626.035952601, 87510.31231623856]


"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def num_otm(u,LFK4):
    return LFK4[0] + u *(LFK4[1] +u *(LFK4[2] +u *(LFK4[3] + u *(LFK4[4] +
            u*(LFK4[5] +u *(LFK4[6] + u *(LFK4[7] +u *(LFK4[8] +u * LFK4[9] ))))))))

"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def den_otm(u,LFK4):
    return 1.0 + u *(LFK4[10] +u *(LFK4[11] + u *(LFK4[12] +u*(LFK4[13] +
                   u *(LFK4[14] +u *(LFK4[15] +u *(LFK4[16] ) ))))) )

"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def num_itm(u,LFK4):
        return LFK4[0] + u *(LFK4[1] +u *(LFK4[2] +u *(LFK4[3] +u *(LFK4[4] +
              u *(LFK4[5] +u *(LFK4[6] + u*(LFK4[7]))) ))))

"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def den_itm(u,LFK4):
    return 1.0 + u *(LFK4[8] +u*(LFK4[9] + u *(LFK4[10] +u *(LFK4[11] +u *(LFK4[12] )))))


"""
Fast and Accurate Analytic Basis Point Volatility, Appendix B, Listing 1
"""
def eta(z):
# case for avoiding incidents of 0/0, z close to zero...
    if len(z) == 0: return z
    index_z_small = (z < 1e-2)
    index_z_nsmall = (z >= 1e-2)
    eta_val=np.zeros(len(z))
    eta_val[index_z_small] = 1 -z[index_z_small] *(0.5+ z[index_z_small] *
           (1.0/12+ z[index_z_small] *(1.0/24+ z[index_z_small] *
            (19.0/720+ z[index_z_small] *(3.0/160+ z[index_z_small] *
             (863.0/60480+ z[index_z_small] *(275.0/24192) )))) ))
    eta_val[index_z_nsmall] =  -z[index_z_nsmall]/ np.log1p(-z[index_z_nsmall])    # log1p(x) = log(1+x)
    return eta_val


"""
volBachelier for different prices
"""
def vol_bachelier_diff_prices(sign, k, forward, t, price):
    iv = np.zeros(len(price))
    i = 0
    for y in price:
        strike = k[i]
        iv[i] = vol_bachelier(1,strike,forward,t,y) 
        i = i+1
    return iv

