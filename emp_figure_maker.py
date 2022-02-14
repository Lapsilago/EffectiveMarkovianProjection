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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import set_parametrization as set_param


#%% Figures 1 and 2

"""
Figure displays the projected variance curves for the EMP-ATM algotihm
Alongside the used matching point
"""
def make_figure_1_and_2_ATM(set_number, strikes,projvar_curves, points_x, points_y):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,2], label = "EMP-ATM")
    ax.plot(points_x, points_y, 'x', label = "Matching Points")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('EMP-ATM'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return


"""
Figure displays the projected variance curves for the EMP-NP algotihm
Alongside the used matching points
"""
def make_figure_1_and_2_NP(set_number,strikes,projvar_curves,points_x,points_y):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,3], label = "EMP-NP")
    ax.plot(points_x, points_y, 'x', label = "Matching Points")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('EMP-NP'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return


"""
Figure displays the projected variance curves for the EMP-MP algotihm
Alongside the used matching points
"""
def make_figure_1_and_2_MP(set_number,strikes,projvar_curves,points_x,points_y):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,1], label = "EMP-MP")
    ax.plot(points_x, points_y, 'x', label = "Matching Points")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('EMP-MP'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return


"""
Figure displays the projected variance curves for the EMP-MP algotihm
Alongside the used matching points
"""
def make_figure_1_and_2_MP_alternative(set_number,strikes,projvar_curves,points_x,points_y):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,4], label = "EMP-MP")
    ax.plot(points_x, points_y, 'x', label = "Matching Points")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('EMP-MP'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return


"""
Figure displays the projected variance curves for the EMP-MP algotihm
Alongside the used matching points
"""
def make_figure_1_MP_dashed(set_number,strikes,projvar_curves,points_x,points_y):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,1],'--', label = "EMP-MP")
    ax.plot(points_x, points_y, 'x', label = "Matching Points")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('EMP-MP'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return


#%% Figure 3

"""
Figure displays the projected variance curves for the different EMP algotihms
"""
def make_figure_3(set_number, case_number, strikes, projvar_curves):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, projvar_curves[:,case_number-1,0], label = "Original")
    ax.plot(strikes, projvar_curves[:,case_number-1,1], label = "EMP-MP")
    ax.plot(strikes, projvar_curves[:,case_number-1,2], label = "EMP-ATM")
    ax.plot(strikes, projvar_curves[:,case_number-1,3], label = "EMP-NP")
    
    ax.set(xlabel='strike', ylabel='projected variance',
      title= str('Beta ' + str(set_param.get_beta(set_number,case_number))))
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return 


#%% Figure 5

"""
Figure displays the projected variance curves for a worst case scenario
"""
def make_figure_5(strikes,locvol_curves):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, locvol_curves[:,0], label = "Original")
    ax.plot(strikes, locvol_curves[:,1], label = "EMP-MP Variant 1")
    ax.plot(strikes, locvol_curves[:,2], label = "EMP-MP Variant 2")
    ax.plot(strikes, locvol_curves[:,3], label = "EMP-ATM")
    ax.plot(strikes, locvol_curves[:,4], label = "EMP-NP")
    
    ax.set(xlabel='strike', ylabel='projected variance',
           title= str('Worst case scenario'))
    ax.legend(fontsize ='x-small')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    plt.show()
    return 


#%% Figures 6 to 9

"""
Figure displays the implied volatility curves for the different EMP algotihms
"""
def make_figure_6_to_9(set_number, case_number, strikes,iv_curves):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, iv_curves[:,case_number-1,0], label = "Original")
    ax.plot(strikes, iv_curves[:,case_number-1,1], label = "EMP-MP")
    ax.plot(strikes, iv_curves[:,case_number-1,2], label = "EMP-NP")
    
    ax.set(xlabel='strike', ylabel='implied volatility',
           title= str('Beta ' + str(set_param.get_beta(set_number,case_number))))
    ax.legend()
    
    plt.show()
    return 


#%% Figure 10

"""
Figure displays the implied volatility curves for the calibrated 1y2y rates
"""
def make_figure_10(strikes_market,iv_market,strikes,iv_zabr_v1,iv_zabr_v2,iv_zabr_v3):
    fig, ax = plt.subplots()
    
    ax.plot(strikes_market, iv_market*10000, 'x', label = "Samples")
    ax.plot(strikes, iv_zabr_v1*10000, label = "dZABR V1")
    ax.plot(strikes, iv_zabr_v2*10000, label = "dZABR V2")
    ax.plot(strikes, iv_zabr_v3*10000, label = "dZABR V3")
    
    ax.set(xlabel='strike', ylabel='implied volatility', title= '1y2y Swap Rate')
    ax.legend(fontsize ='x-small')
    
    plt.show()
    return


#%% Figure 11

"""
Figure displays the implied volatility curves for the calibrated 1y5y rates
"""
def make_figure_11(strikes_market,iv_market,strikes,iv_zabr_v1,iv_zabr_v2,iv_zabr_v3):
    fig, ax = plt.subplots()
    
    ax.plot(strikes_market, iv_market*10000, 'x', label = "Samples")
    ax.plot(strikes, iv_zabr_v1*10000, label = "dZABR V1")
    ax.plot(strikes, iv_zabr_v2*10000, label = "dZABR V2")
    ax.plot(strikes, iv_zabr_v3*10000, label = "dZABR V3")
    
    ax.set(xlabel='strike', ylabel='implied volatility', title= '1y5y Swap Rate')
    ax.legend(fontsize ='x-small')
    
    plt.show()
    return


#%% Figure 12

"""
Figure displays the implied volatility curves for the calibrated spread rates
"""
def make_figure_12(strikes,iv_market,iv_zabr_spread_curves):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, iv_market*10000, label = "Original nSABR")
    for version_1y2y in np.arange(1,4,1):
        for version_1y5y in np.arange(1,4,1):
            ax.plot(strikes, iv_zabr_spread_curves[:,version_1y2y-1,version_1y5y-1]*10000,
                    label = str("dZABR V"+str(version_1y2y)+" V" + str(version_1y5y)))
    
    ax.set(xlabel='strike', ylabel='implied volatility', title= 'CMS Spread')
    ax.legend(fontsize ='x-small')
    
    plt.show()
    return


#%% Figure 13

"""
Figure displays the implied volatility curves for the midcurve rates
"""
def make_figure_13(strikes,iv_zabr_v1_v1,iv_zabr_v2_v2,iv_zabr_v3_v3):
    fig, ax = plt.subplots()
    
    ax.plot(strikes, iv_zabr_v1_v1*10000, label = "dZABR V1 V1")
    ax.plot(strikes, iv_zabr_v2_v2*10000, label = "dZABR V2 V2")
    ax.plot(strikes, iv_zabr_v3_v3*10000, label = "dZABR V3 V3")
    
    ax.set(xlabel='strike', ylabel='implied volatility', title= 'Mid-Curve Rate')
    ax.legend(fontsize ='x-small')
    
    plt.show()
    return

