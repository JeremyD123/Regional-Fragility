# -*- coding: utf-8 -*-
"""
Regional-scale seismic fragility assessment of buildings

Created on Mar4 2024

Author: Jia-Yi Ding

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from poibin import PoiBin
# from fast_poibin import PoiBin
from scipy import stats

def plot_pb_distribution(X_unsafe,res_pmf,mu,sigma,im):
    config = {
        "font.family":'serif',
        "font.size": 20,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman']
    }
    rcParams.update(config)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(X_unsafe,res_pmf,linestyle='-',linewidth=1.5,color='#0072BD')
    mu = round(mu)
    sigma = round(sigma)
    textpri = r'$\mu$ = '+str(mu)+',$\sigma$ = '+str(sigma)
    if im == 0.3:
        plt.xlim(0,200), plt.ylim(0,0.12)
        plt.xticks(np.arange(0,201,40)), plt.yticks(np.arange(0,0.13,0.03))
        plt.text(30, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.35:
        plt.xlim(0,200), plt.ylim(0,0.06)
        plt.xticks(np.arange(0,201,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(90, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.4:
        plt.xlim(200,400), plt.ylim(0,0.06)
        plt.xticks(np.arange(200,401,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(290, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.45:
        plt.xlim(500,700), plt.ylim(0,0.06)
        plt.xticks(np.arange(500,701,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(650, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.5:
        plt.xlim(700,900), plt.ylim(0,0.06)
        plt.xticks(np.arange(700,901,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(820, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.55:
        plt.xlim(700,900), plt.ylim(0,0.06)
        plt.xticks(np.arange(700,901,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(850, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.6:
        plt.xlim(800,1000), plt.ylim(0,0.06)
        plt.xticks(np.arange(800,1001,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(860, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.65:
        plt.xlim(800,1000), plt.ylim(0,0.06)
        plt.xticks(np.arange(800,1001,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(900, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.70:
        plt.xlim(800,1000), plt.ylim(0,0.06)
        plt.xticks(np.arange(800,1001,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(910, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.75:
        plt.xlim(800,1000), plt.ylim(0,0.06)
        plt.xticks(np.arange(800,1001,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(930, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif im == 0.8:
        plt.xlim(800,1000), plt.ylim(0,0.06)
        plt.xticks(np.arange(800,1001,40)), plt.yticks(np.arange(0,0.08,0.02))
        plt.text(940, 0.03, textpri.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    
    plt.xlabel('Number of unsafe buildings under given IM, $X$$_{IM}$', fontsize=20)
    plt.ylabel('Probability of unsafe buildings', fontsize=20)
    plt.savefig(path_output+'/fig_reg_pdf_'+model_edp+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_pb_distribution_all(X_unsafe,res_pmf,mu,sigma):
    config = {
        "font.family":'serif',
        "font.size": 20,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman']
    }
    rcParams.update(config)
    
    plt.figure(figsize=(10, 5), dpi=400)
    
    if model_edp == 'MIDR':
        # legn0, = plt.plot(X_unsafe,res_pmf[:,0],linestyle='-',linewidth=1,color='#000000')
        legn1, = plt.plot(X_unsafe,res_pmf[:,6],linestyle='-',linewidth=1,color='#0072BD')
        legn2, = plt.plot(X_unsafe,res_pmf[:,7],linestyle='--',linewidth=1,color='#77AC30')
        legn3, = plt.plot(X_unsafe,res_pmf[:,8],linestyle='-',linewidth=1,color='#D95319')
        legn4, = plt.plot(X_unsafe,res_pmf[:,9],linestyle='--',linewidth=1,color='#7E2F8E')
        legn5, = plt.plot(X_unsafe,res_pmf[:,10],linestyle='-',linewidth=1,color='#A2142F')
        legn6, = plt.plot(X_unsafe,res_pmf[:,11],linestyle='--',linewidth=1,color='#000000')
        legn7, = plt.plot(X_unsafe,res_pmf[:,12],linestyle='--',linewidth=1,color='#0072BD')
        legn8, = plt.plot(X_unsafe,res_pmf[:,13],linestyle='-',linewidth=1,color='#77AC30')
        legn9, = plt.plot(X_unsafe,res_pmf[:,14],linestyle='--',linewidth=1,color='#D95319')
        legn10, = plt.plot(X_unsafe,res_pmf[:,15],linestyle='-',linewidth=1,color='#7E2F8E')
        legn11, = plt.plot(X_unsafe,res_pmf[:,16],linestyle='--',linewidth=1,color='#A2142F')
        legn12, = plt.plot(X_unsafe,res_pmf[:,17],linestyle='-',linewidth=1,color='#000000')
        plt.legend([legn1,legn2,legn3,legn4,legn5,legn6,legn7,legn8,legn9,legn10,legn11,legn12,],\
                [r'$AvgSA$ = 0.35 g, $\mu$ = '+str(round(mu[6]))+', $\sigma$ = '+str(round(sigma[6])),\
                 r'$AvgSA$ = 0.40 g, $\mu$ = '+str(round(mu[7]))+', $\sigma$ = '+str(round(sigma[7])),\
                 r'$AvgSA$ = 0.45 g, $\mu$ = '+str(round(mu[8]))+', $\sigma$ = '+str(round(sigma[8])),\
                 r'$AvgSA$ = 0.50 g, $\mu$ = '+str(round(mu[9]))+', $\sigma$ = '+str(round(sigma[9])),\
                 r'$AvgSA$ = 0.55 g, $\mu$ = '+str(round(mu[10]))+', $\sigma$ = '+str(round(sigma[10])),\
                 r'$AvgSA$ = 0.60 g, $\mu$ = '+str(round(mu[11]))+', $\sigma$ = '+str(round(sigma[11])),\
                 r'$AvgSA$ = 0.65 g, $\mu$ = '+str(round(mu[12]))+', $\sigma$ = '+str(round(sigma[12])),\
                 r'$AvgSA$ = 0.70 g, $\mu$ = '+str(round(mu[13]))+', $\sigma$ = '+str(round(sigma[13])),\
                 r'$AvgSA$ = 0.75 g, $\mu$ = '+str(round(mu[14]))+', $\sigma$ = '+str(round(sigma[14])),\
                 r'$AvgSA$ = 0.80 g, $\mu$ = '+str(round(mu[15]))+', $\sigma$ = '+str(round(sigma[15])),\
                 r'$AvgSA$ = 0.85 g, $\mu$ = '+str(round(mu[16]))+', $\sigma$ = '+str(round(sigma[16])),\
                 r'$AvgSA$ = 0.90 g, $\mu$ = '+str(round(mu[17]))+', $\sigma$ = '+str(round(sigma[17])),\
                 ], loc=0, ncol=2, fontsize=12, edgecolor='k')                            
    elif model_edp == 'PFA':
        # legn0, = plt.plot(X_unsafe,res_pmf[:,0],linestyle='-',linewidth=1,color='#000000')
        legn1, = plt.plot(X_unsafe,res_pmf[:,6],linestyle='-',linewidth=1,color='#0072BD')
        legn2, = plt.plot(X_unsafe,res_pmf[:,7],linestyle='--',linewidth=1,color='#77AC30')
        legn3, = plt.plot(X_unsafe,res_pmf[:,8],linestyle='-',linewidth=1,color='#D95319')
        legn4, = plt.plot(X_unsafe,res_pmf[:,9],linestyle='--',linewidth=1,color='#7E2F8E')
        legn5, = plt.plot(X_unsafe,res_pmf[:,10],linestyle='-',linewidth=1,color='#A2142F')
        legn6, = plt.plot(X_unsafe,res_pmf[:,11],linestyle='--',linewidth=1,color='#000000')
        legn7, = plt.plot(X_unsafe,res_pmf[:,12],linestyle='--',linewidth=1,color='#0072BD')
        legn8, = plt.plot(X_unsafe,res_pmf[:,13],linestyle='-',linewidth=1,color='#77AC30')
        legn9, = plt.plot(X_unsafe,res_pmf[:,14],linestyle='--',linewidth=1,color='#D95319')
        legn10, = plt.plot(X_unsafe,res_pmf[:,15],linestyle='-',linewidth=1,color='#7E2F8E')
        legn11, = plt.plot(X_unsafe,res_pmf[:,16],linestyle='--',linewidth=1,color='#A2142F')
        legn12, = plt.plot(X_unsafe,res_pmf[:,17],linestyle='-',linewidth=1,color='#000000')
        legn13, = plt.plot(X_unsafe,res_pmf[:,18],linestyle='--',linewidth=1,color='#EDB120')
        legn14, = plt.plot(X_unsafe,res_pmf[:,19],linestyle='-',linewidth=1,color='#EDB120')
        plt.legend([legn1,legn2,legn3,legn4,legn5,legn6,legn7,legn8,legn9,legn10,legn11,legn12,legn13,legn14,],\
                [r'$AvgSA$ = 0.35 g, $\mu$ = '+str(round(mu[6]))+', $\sigma$ = '+str(round(sigma[6])),\
                 r'$AvgSA$ = 0.40 g, $\mu$ = '+str(round(mu[7]))+', $\sigma$ = '+str(round(sigma[7])),\
                 r'$AvgSA$ = 0.45 g, $\mu$ = '+str(round(mu[8]))+', $\sigma$ = '+str(round(sigma[8])),\
                 r'$AvgSA$ = 0.50 g, $\mu$ = '+str(round(mu[9]))+', $\sigma$ = '+str(round(sigma[9])),\
                 r'$AvgSA$ = 0.55 g, $\mu$ = '+str(round(mu[10]))+', $\sigma$ = '+str(round(sigma[10])),\
                 r'$AvgSA$ = 0.60 g, $\mu$ = '+str(round(mu[11]))+', $\sigma$ = '+str(round(sigma[11])),\
                 r'$AvgSA$ = 0.65 g, $\mu$ = '+str(round(mu[12]))+', $\sigma$ = '+str(round(sigma[12])),\
                 r'$AvgSA$ = 0.70 g, $\mu$ = '+str(round(mu[13]))+', $\sigma$ = '+str(round(sigma[13])),\
                 r'$AvgSA$ = 0.75 g, $\mu$ = '+str(round(mu[14]))+', $\sigma$ = '+str(round(sigma[14])),\
                 r'$AvgSA$ = 0.80 g, $\mu$ = '+str(round(mu[15]))+', $\sigma$ = '+str(round(sigma[15])),\
                 r'$AvgSA$ = 0.85 g, $\mu$ = '+str(round(mu[16]))+', $\sigma$ = '+str(round(sigma[16])),\
                 r'$AvgSA$ = 0.90 g, $\mu$ = '+str(round(mu[17]))+', $\sigma$ = '+str(round(sigma[17])),\
                 r'$AvgSA$ = 0.95 g, $\mu$ = '+str(round(mu[18]))+', $\sigma$ = '+str(round(sigma[18])),\
                 r'$AvgSA$ = 1.00 g, $\mu$ = '+str(round(mu[19]))+', $\sigma$ = '+str(round(sigma[19])),\
                 ], loc=0, ncol=2, fontsize=12, edgecolor='k')
        
    plt.xlim(0,1000), plt.ylim(0,0.078)
    plt.xticks(np.arange(0,1001,200)), plt.yticks(np.arange(0.,0.07,0.02))
    
    plt.xlabel('Number of unsafe buildings under given IM', fontsize=20)
    plt.ylabel('Probability of unsafe buildings', fontsize=20)
    plt.savefig(path_output+'/fig_reg_pdf_'+model_edp+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def IOR_curve(p_unsafe_ior):
    # Immediate occupancy ratio curve for 20%, 40%, 60%, and 80%
    config = {
        "font.family":'serif',
        "font.size": 20,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    
    plt.figure(figsize=(8, 6), dpi=300)
    
    legn1, = plt.plot(im,p_unsafe_ior[:,0],linestyle='-',linewidth=1,color='#0072BD')
    legn2, = plt.plot(im,p_unsafe_ior[:,1],linestyle='-',linewidth=1,color='#77AC30')
    legn3, = plt.plot(im,p_unsafe_ior[:,2],linestyle='-',linewidth=1,color='#D95319')
    legn4, = plt.plot(im,p_unsafe_ior[:,3],linestyle='-',linewidth=1,color='#A2142F')
    legn5, = plt.plot(im,p_unsafe_ior[:,4],linestyle='--',linewidth=1,color='#A2142F')
           
    plt.legend([legn1,legn2,legn3,legn4,legn5,],['IOR=10%','IOR=30%','IOR=50%','IOR=70%','IOR=90%'],\
               loc=4,ncol=2,fontsize=12,edgecolor='k') 
    plt.xlabel('$X$$_{IM}$', fontsize=22)
    plt.ylabel('Probability of failling to achieve IOR', fontsize=22)
    plt.xlim((0.3,0.8)),plt.ylim((0.,1.0))
    plt.xticks(np.arange(0.3,0.9,0.1)), plt.yticks(np.arange(0.,1.2,0.2))
    
    plt.savefig(path_output+'fig_ior_curve.png', dpi=300, bbox_inches='tight')
    plt.show() 

def Pb_distribution():
    # Obtain probability of unsafe to occupy for each building
    data_file_sp = pd.read_excel(path_input+'/Prob_US_Building.xlsx',sheet_name=model_edp)
    data = data_file_sp.iloc[:,:].values
    
    # Immediate occupancy ratio
    ior_X = [200, 400, 600]
    p_unsafe_ior = np.zeros((len(im),len(ior_X)))
    
    # Average number of unsafe to occupy
    mu = np.zeros((len(im),1))
    sigma = np.zeros((len(im),1))
    p_unsafe = np.zeros((len(im)+1,3))
    
    X_unsafe = list(np.arange(0,len(data[:,0])+1,dtype=np.int64))
    res_pmf = np.zeros((len(data[:,0])+1,len(im)))
    for i in range(0,len(im)):
    # for i in np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]):
        prob_buid = data[:,i+1]
        # obtain pmf with the poibin package
        pb = PoiBin(prob_buid)
        res_pmf[:,i] = pb.pmf(X_unsafe)
        # obtain the mean and standard of this poibin
        mu = np.squeeze(mu)
        sigma = np.squeeze(sigma)
        mu[i] = sum(prob_buid)
        sigma[i] = (sum((1-prob_buid)*prob_buid))**0.5
        
        # plot_pb_distribution(X_unsafe,res_pmf[:,i],mu[i],sigma[i],im[i])
        
        # Upper and lower limits of number of unsafe to occupy
        X_low = mu[i]-1.96*sigma[i]
        X_up = mu[i]+1.96*sigma[i]
        # Probability of average number of unsafe to occupy
        p_unsafe[i+1,0] = round(mu[i])/len(prob_buid)
        p_unsafe[i+1,1] = round(X_low)/len(prob_buid)
        p_unsafe[i+1,2] = round(X_up)/len(prob_buid)
        
        # Probability of failling to achieve IOR
        p_unsafe_ior[i,:] = 1-pb.cdf(ior_X)
    
    plot_pb_distribution_all(X_unsafe,res_pmf,mu,sigma)
    # IOR_curve(p_unsafe_ior)
    
    return p_unsafe

# define function for fit, miu is mean，sigma is standard
def func(x, s, mu, sigma):    
    return stats.lognorm.cdf(abs(x-mu)/sigma, s)

if __name__ == "__main__":
    model_edp = 'MIDR' # 'MIDR' 'PFA' 
    
    path_input = "D:/Regional F/InputData"
    path_output = "D:/Regional F/OutputData"
    
    im = np.linspace(0.5,1.0,num=20)
    value_X = np.linspace(0,1.0,num=21)
    
    p_unsafe = Pb_distribution()
    config = {
        "font.family":'serif',
        "font.size": 18,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman']
    }
    rcParams.update(config)
    
    plt.figure(figsize=(8, 4), dpi=300)
    
    legn2 = plt.fill_between(value_X,p_unsafe[:,1],p_unsafe[:,2],facecolor='skyblue',edgecolor='skyblue',alpha=0.5)
    legn1, = plt.plot(value_X,p_unsafe[:,0],linestyle='-',linewidth=1.5,color='#A2142F')

    char_std = '95% confidence interval'
    
    plt.legend([legn1,legn2],['Average', char_std], loc=2, ncol=1, fontsize=16, edgecolor='k')       
    plt.xlabel('$AvgSA$ (g)', fontsize=20)
    plt.ylabel('Probability of exceedance', fontsize=20)
    plt.xlim((0.0,1.0)),plt.ylim((0.0,1.0))
    plt.xticks(np.arange(0.1,1.1,0.1)), plt.yticks(np.arange(0.,1.2,0.2))
    
    plt.savefig(path_output+'/fig_reg_fri_'+model_edp+'.png', dpi=300, bbox_inches='tight')
    plt.show() 
