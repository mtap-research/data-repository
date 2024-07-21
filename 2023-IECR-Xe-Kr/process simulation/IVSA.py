import pandas as pd
import numpy as np
from tqdm import tqdm
import pyiast
import scipy.optimize as optim
import os
import matplotlib.pyplot as plt
import pickle
import csv
from scipy.optimize import curve_fit

def langmuir(p,M,K):
    return M*p*K/(1+p*K)

Arrh = lambda T,dH ,T_ref: np.exp(-dH/8.3145*(1/T - 1/T_ref)) # Arrhenius equation (Clasius-Clapeyron Equation)

def iso_mix(P_par, T, M_list, h_list, dH_list):
    
    p= np.linspace(0,100,100)
    i=0
    q_Xe=np.zeros(100)
    q_Kr=np.zeros(100)
    
    for i in range(0,100):
        Xe_M = M_list[0]
        Kr_M = M_list[1]
        Xe_K = h_list[0]*100000/Xe_M
        Kr_K = h_list[1]*100000/Kr_M
        q_Xe[i] = langmuir(p[i],Xe_M,Xe_K)
        q_Kr[i] = langmuir(p[i],Kr_M,Kr_K)
        i = i+1
        
    iso_Xe={"p" : p, "q" : q_Xe}
    iso_Kr={"p" : p,"q" : q_Kr}
    data_Xe=pd.DataFrame(iso_Xe)
    data_Kr=pd.DataFrame(iso_Kr)
    
    df_ISO_list =[data_Xe,data_Kr]
    iso_list = [pyiast.ModelIsotherm(df_ISO_list[0],loading_key = 'q', pressure_key = 'p',model = 'Langmuir',param_guess={"M": Xe_M,"K":Xe_K}),
               pyiast.ModelIsotherm(df_ISO_list[1],loading_key = 'q', pressure_key = 'p',model = 'Langmuir',param_guess={"M": Kr_M,"K":Kr_K})]

    P_norm = []
    for (p,dh,tref) in zip(P_par, dH_list,[T, T]):
        p_n = Arrh(T,dh,tref)*p 
        P_norm.append(p_n)
    P_norm_arr = np.array(P_norm)

    if P_norm_arr.ndim > 1:
        for i in range(len(P_norm[0])):
            p_tmp = P_norm_arr[i,:]
            p_tmp[p_tmp<0.000001] = 0.000001
            q_IAST_tmp = pyiast.iast(p_tmp,
                                     iso_list,
                                     warningoff=True)
    else:
        try:
            p_tmp = P_norm_arr
            p_tmp[p_tmp<0.000001] = 0.000001
          
            q_IAST_tmp = pyiast.iast(p_tmp,
                                    iso_list,
                                     warningoff=True)
        except:    
            try:
                x_IG = np.ones(len(p_tmp))/len(p_tmp)
                q_IAST_tmp = pyiast.iast(p_tmp,
                                        iso_list,adsorbed_mole_fraction_guess = x_IG,
                                        warningoff=True)
            except:
                try:
                    arg_min = np.argmin(p_tmp)
                    p_tmp[p_tmp<0.000001] = 0.000001
                    x_IG = 0.05*np.ones(len(p_tmp))
                    x_IG[arg_min] = 1 - 0.05*(len(p_tmp)-1)
                    q_IAST_tmp = pyiast.iast(p_tmp,
                                            iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)

                except:
                    try:
                        arg_max = np.argmax(p_tmp)
                        p_tmp[p_tmp<0.000001] = 0.000001
                        x_IG = 0.05*np.ones(len(p_tmp))
                        x_IG[arg_max] = 1 - 0.05*(len(p_tmp)-1)
        
                        q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                                warningoff=True)        
                    except:
                        try:
                            arg_max = np.argmax(p_tmp)
                            p_tmp[p_tmp<0.000001] = 0.000001
                            x_IG = 0.15*np.ones(len(p_tmp))
                            x_IG[arg_max] = 1 - 0.15*(len(p_tmp)-1)
                            q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                                warningoff=True)
                        except:
                            try:
                                arg_min = np.argmin(p_tmp)
                                p_tmp[p_tmp<0.000001] = 0.000001
                                x_IG = 0.01*np.ones(len(p_tmp))
                                x_IG[arg_min] = 1 - 0.01*(len(p_tmp)-1)
                      
                                q_IAST_tmp = pyiast.iast(p_tmp,
                                            iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)

                            except:
                                arg_max = np.argmax(p_tmp)
                                p_tmp[p_tmp<0.000001] = 0.000001
                                x_IG = 0.01*np.ones(len(p_tmp))
                                x_IG[arg_max] = 1 - 0.01*(len(p_tmp)-1)
                    
                                q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)
    
    S = q_IAST_tmp[0]/q_IAST_tmp[1]
           
    return q_IAST_tmp,S

def x2x(x_ini,P_high,P_low,M_list,h_list, dH_input, yfeed, Tfeed):
    
    dH_1, dH_2 = dH_input[:2]  
    dH = np.array([dH_1,dH_2])*1000 
    P_low_part = np.array(x_ini)*P_low 
    P_high_part = np.array(yfeed)*P_high
    P_low_part = np.reshape(P_low_part,2)
    P_high_part = np.reshape(P_high_part,2)
    
    q_des = iso_mix(P_low_part,Tfeed,M_list,h_list,dH)[0]
    q_sat_tot = iso_mix(P_high_part,Tfeed,M_list,h_list,dH)[0]

    Dq_tot = q_sat_tot-q_des
    sat_extent = np.array(yfeed)/Dq_tot 
    ind_lead_tot = np.argmax(sat_extent)
    dq = q_sat_tot - q_des
    x_out = dq/(np.sum(dq))
    return x_out,ind_lead_tot

def rec(x_ini, P_high, P_low, M_list, h_list, dH_input, yfeed, Tfeed):
    def x_err(xx):
        x_new,i_lead = x2x([xx, 1-xx], P_high,P_low, M_list, h_list, dH_input, yfeed, Tfeed)
        return (xx-x_new[0])**2
    
    sol = optim.least_squares(x_err,x_ini, bounds = [0,1])
    x_sol = sol.x
    _,i_lead = x2x([x_sol, 1- x_sol], P_high, P_low, M_list, h_list, dH_input, yfeed, Tfeed)

    Recovery = 1-(1-x_sol)/x_sol*yfeed[0]/yfeed[1]
    if Recovery < 0 or Recovery > 1:
        Recovery = 1-x_sol/(1-x_sol)*yfeed[1]/yfeed[0]

    return Recovery, i_lead, x_sol
