from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error, r2_score
# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import joblib
from scipy.optimize import curve_fit

import time

def calculate_R2():
    # plot R2
    config = {
        "font.family":'serif',
        "font.size": 22,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    plt.figure(figsize=(8, 6), dpi=300)
    
    plt.scatter(x=Y_train_pred, y=Y_train, s=50, c='#EDB120', marker="P")
    plt.scatter(x=Y_test_pred, y=Y_test, s=50, c='#0072BD', marker="o")
    plt.legend(['Training set', 'Test set'], loc=0, ncol=1, fontsize=22, edgecolor='k')

    plt.xlabel("Predicted value", fontsize=24)
    plt.ylabel("Actual value", fontsize=24)
    
    train_R2 = round(r2_score(Y_train,Y_train_pred),3)
    train_MAE = round(mean_absolute_error(Y_train, Y_train_pred),3)
    train_MARE = round(mean_absolute_percentage_error(Y_train, Y_train_pred),3)
    train_RMSE = round(mean_squared_error(Y_train,Y_train_pred)**0.5,3)
    
    test_R2 = round(r2_score(Y_test,Y_test_pred),3)
    test_MAE = round(mean_absolute_error(Y_test, Y_test_pred),3)
    test_MARE = round(mean_absolute_percentage_error(Y_test, Y_test_pred),3)
    test_RMSE = round(mean_squared_error(Y_test, Y_test_pred)**0.5,3)
    
    textpri1 = r'Training set,$R^2$: '+str(train_R2)+',$MAE$: '+str(train_MAE)+\
        ',$MARE$: '+str(train_MARE)+',$RMSE$: '+str(train_RMSE)
    textpri2 = r'Testing set,$R^2$: '+str(test_R2)+',$MAE$: '+str(test_MAE)+\
        ',$MARE$: '+str(test_MARE)+',$RMSE$: '+str(test_RMSE)
    
    if model_edp == 'MIDR':
        plt.xlim(-6.8,-2.2), plt.ylim(-6.8,-2.2)
        plt.plot([-6.8,-2.2], [-6.8,-2.2], color="gray", ls="--")
        plt.xticks(np.arange(-6,-2.1,1.0)), plt.yticks(np.arange(-6,-2.1,1.0))
        plt.text(-4.5,-6.5, textpri1.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
        plt.text(-3.3,-6.5, textpri2.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    elif model_edp == 'PFA':
        plt.xlim(-2.1,0.9), plt.ylim(-2.1,0.9)
        plt.plot([-2.1,0.9], [-2.1,0.9], color="gray", ls="--")
        plt.xticks(np.arange(-2,1.0,1.0)), plt.yticks(np.arange(-2,1.0,1.0))
        plt.text(-0.60,-1.92, textpri1.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
        plt.text( 0.18,-1.92, textpri2.replace(",", "\n"), size=16, color ="k", style ="normal", weight ="light")
    plt.savefig(path_output+'/figr2_'+model_edp+'.png', dpi=300, bbox_inches='tight') 
    plt.show()

# fitting function: mu and sigma^2 as the mean and variance of the normal distribution
def func(x, s, mu, sigma):    
    return stats.lognorm.cdf(abs(x-mu)/sigma, s)
    
def caculate_fragility(model):
    num_IM = 20
    value_IM = np.linspace(0.,1.0,num=num_IM+1).reshape(-1,1)
            
    X_im = np.log(value_IM[1:])
    # obtain building design parameters
    data_file_sp = pd.read_excel(path_input+'/Virtual City Building.xlsx',sheet_name='ID')
    data_sp = data_file_sp.iloc[:,:].values
    
    # fragility curve for Slight, Moderate, Extensive, Complete
    config = {
        "font.family":'serif',
        "font.size": 22,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    
    # for i in range(len(data_sp[:,0])):
    for i in np.array([13,32,78,617,668,688]):
        X_sp = np.ones((num_IM,1))*data_sp[i,1:].reshape(1,-1)
        X_input = np.append(X_sp,X_im,axis=1)
        Y_dists = model.pred_dist(X_input)
        Y_para = Y_dists.params
        Y_loc = Y_para.get('loc')
        Y_scale = Y_para.get('scale')
        
        # fragility for slight, moderate, extensive, complete
        if model_edp == 'MIDR':
            if data_sp[i,1] == 1 and data_sp[i,4] <= 3:
                threshold = np.array([0.0050, 0.0080, 0.0200, 0.0500]).reshape(-1, 1)
            elif data_sp[i,1] == 1 and data_sp[i,4] > 3 and data_sp[i,4] <= 7:
                threshold = np.array([0.0033, 0.0053, 0.0133, 0.0333]).reshape(-1, 1)
            elif data_sp[i,1] == 1 and data_sp[i,4] > 7:
                threshold = np.array([0.0025, 0.0040, 0.0100, 0.0250]).reshape(-1, 1)
            elif data_sp[i,1] == 2 and data_sp[i,4] <= 3:
                threshold = np.array([0.0040, 0.0076, 0.0197, 0.0500]).reshape(-1, 1)
            elif data_sp[i,1] == 2 and data_sp[i,4] > 3 and data_sp[i,4] <= 7:
                threshold = np.array([0.0027, 0.0051, 0.0132, 0.0333]).reshape(-1, 1)
            elif data_sp[i,1] == 2 and data_sp[i,4] > 7:
                threshold = np.array([0.0020, 0.0038, 0.0099, 0.0250]).reshape(-1, 1)    
        elif model_edp == 'PFA':
            threshold = np.array([0.20, 0.40, 0.80, 1.60]).reshape(-1, 1)
            
        threshold = np.log(threshold)
        fragi = np.zeros((num_IM+1,len(threshold)))
        for j in range(len(threshold)):
            for k in range(num_IM):
                fragi[k+1,j] = 1-stats.norm.cdf(threshold[j],loc=Y_loc[k],scale=1.5*Y_scale[k])
                if fragi[k+1,j] < fragi[k,j]:
                    fragi[k+1,j] = fragi[k,j]
        
        sigma = np.ones(len(value_IM))
        sigma[[0]] = 0.001 # first and last points
        sigma1 = np.ones(len(value_IM))
        sigma1[[0,15]] = np.ones(2)*0.001
        sigma3 = np.ones(len(value_IM))
        sigma3[[0,6]] = np.ones(2)*0.001
        popt0, pcov0 = curve_fit(func, np.squeeze(value_IM), np.squeeze(fragi[:,0]), maxfev=5000,\
                                      sigma=sigma1, bounds=([0,-20,0], [10,20,20]))
        popt1, pcov1 = curve_fit(func, np.squeeze(value_IM), np.squeeze(fragi[:,1]), maxfev=5000,\
                                      sigma=sigma, bounds=([0,-10,0], [10,10,2]))
        popt2, pcov2 = curve_fit(func, np.squeeze(value_IM), np.squeeze(fragi[:,2]), maxfev=5000,\
                                      sigma=sigma, bounds=([0,-10,0], [10,10,2]))
        popt3, pcov3 = curve_fit(func, np.squeeze(value_IM), np.squeeze(fragi[:,3]), maxfev=5000,\
                                      sigma=sigma3, bounds=([0,-10,0], [10,10,2]))
            
        value_X = np.linspace(0.,1.0,num=51).reshape(-1,1)
        plt.figure(figsize=(8, 4), dpi=300)
        plt.plot(value_X,func(value_X,*popt0),linestyle='-',linewidth=2.0,color='#0072BD')
        plt.plot(value_X,func(value_X,*popt1),linestyle='-',linewidth=2.0,color='#77AC30')
        plt.plot(value_X,func(value_X,*popt2),linestyle='-',linewidth=2.0,color='#7E2F8E')
        plt.plot(value_X,func(value_X,*popt3),linestyle='-',linewidth=2.0,color='#A2142F')
        
        # plt.figure(figsize=(8, 4),  dpi=300)
        # plt.plot(value_IM,fragi[:,0],linestyle='-',linewidth=1,marker='v',markersize=6,color='#0072BD')
        # plt.plot(value_IM,fragi[:,1],linestyle='-',linewidth=1,marker='s',markersize=6,color='#77AC30')
        # plt.plot(value_IM,fragi[:,2],linestyle='-',linewidth=1,marker='o',markersize=6,color='#D95319')
        # plt.plot(value_IM,fragi[:,3],linestyle='-',linewidth=1,marker='^',markersize=6,color='#A2142F')
        
        plt.xlabel("$AvgSA$ (g)", fontsize=24)
        plt.ylabel("Probability of exceedance", fontsize=24)
        plt.xlim((0.,1.0)),plt.ylim((0.,1.0))
        plt.xticks(np.arange(0.0,1.1,0.5)), plt.yticks(np.arange(0.,1.1,0.5))
        plt.legend(['Slight','Moderate','Extensive','Complete'], loc=4, ncol=1, fontsize=20, edgecolor='k')
        plt.title('Building '+str(i), fontsize=22)
        plt.savefig(path_output+'/fig_frigi_'+model_edp+str(i)+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        time.sleep(1)

def caculate_prob(model):
    # obtain building design parameters
    data_file_sp = pd.read_excel(path_input+'/Virtual City Building.xlsx',sheet_name='ID')
    X_sp = data_file_sp.iloc[:,1:].values
    # obtain ground motion IMs
    data_file_im = pd.read_excel(path_input+'/Virtual City Building.xlsx',sheet_name=model_edp)
    data_im = data_file_im.iloc[:,:].values
    
    im = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    # im = np.array([7,11,15]) AvgSA = 0.4, 0.6, and 0.8 g
    prob_exceed = np.zeros((len(X_sp[:,0]),4))
    prob_US_build = np.zeros((len(X_sp[:,0]),len(im)))
    # for k in range(len(X_input[:,0])):
    for k in range(len(im)):
        X_im = np.ones((len(X_sp),1))*data_im[im[k],:].reshape(1,-1)
        X_im = np.log(X_im)
        X_input = np.append(X_sp,X_im,axis=1)
        
        Y_dists = model.pred_dist(X_input)
        Y_para = Y_dists.params
        Y_loc = Y_para.get('loc')
        Y_scale = Y_para.get('scale')
        for i in range(len(X_input[:,0])):
            # fragility for slight, moderate, extensive, complete
            if model_edp == 'MIDR':
                if X_input[i,0] == 1 and X_input[i,3] <= 3:
                    threshold = np.array([0.0050, 0.0080, 0.0200, 0.0500]).reshape(-1, 1)
                elif X_input[i,0] == 1 and X_input[i,3] > 3 and X_input[i,3] <= 7:
                    threshold = np.array([0.0033, 0.0053, 0.0133, 0.0333]).reshape(-1, 1)
                elif X_input[i,0] == 1 and X_input[i,3] > 7:
                    threshold = np.array([0.0025, 0.0040, 0.0100, 0.0250]).reshape(-1, 1)
                elif X_input[i,0] == 2 and X_input[i,3] <= 3:
                    threshold = np.array([0.0040, 0.0076, 0.0197, 0.0500]).reshape(-1, 1)
                elif X_input[i,0] == 2 and X_input[i,3] > 3 and X_input[i,3] <= 7:
                    threshold = np.array([0.0027, 0.0051, 0.0132, 0.0333]).reshape(-1, 1)
                elif X_input[i,0] == 2 and X_input[i,3] > 7:
                    threshold = np.array([0.0020, 0.0038, 0.0099, 0.0250]).reshape(-1, 1)
            elif model_edp == 'PFA':
                threshold = np.array([0.20, 0.40, 0.80, 1.60]).reshape(-1, 1)
                
            threshold = np.log(threshold)
            for j in range(len(threshold)):
                prob_exceed[i,j] = 1-stats.norm.cdf(threshold[j],loc=Y_loc[i],scale=1.5*Y_scale[i])
        prob_US_build[:,k] = prob_exceed[:,2] # extensive damage is unsafe to occupy
        
    for i in range(len(prob_US_build[0,:])-1):
        for j in range(len(prob_US_build[:,0])):
            if prob_US_build[j,i] > prob_US_build[j,i+1]:
                prob_US_build[j,i+1] = prob_US_build[j,i]
    
    return prob_US_build

if __name__ == "__main__":
    model_edp = 'PFA' # 'MIDR' 'PFA'
    dimSPs = 8         # dim of structural parameters
    
    path_input = "D:/Regional F/InputData"
    path_output = "D:/Regional F/OutputData"
    #load database
    dataXY = np.load(path_output+'/dataXY_'+model_edp+'.npz')
    X_train, X_test, Y_train, Y_test = dataXY['X_train'], \
        dataXY['X_test'], dataXY['Y_train'], dataXY['Y_test']
    
    #fit NGBoost model
    if model_edp == 'MIDR':
        b = DecisionTreeRegressor(min_samples_split=3, min_samples_leaf=1, max_depth=5, random_state=0)
        params = {
        'Dist': Normal,             # LogNormal Normal Exponential
        'Score': LogScore,          # LogScore
        'minibatch_frac': 0.64,     # =subsample
        'Base': b,                  # max_depth
        'n_estimators': 240,
        'learning_rate': 0.0084,
        # 'verbose': False,
        'random_state': 0,
        'verbose_eval': 20}                              
    elif model_edp == 'PFA':
        b = DecisionTreeRegressor(min_samples_split=3, min_samples_leaf=1, max_depth=5, random_state=0)
        params = {
        'Dist': Normal,             
        'Score': LogScore,
        'minibatch_frac': 0.64,
        'Base': b,
        'n_estimators': 280,
        'learning_rate': 0.0082,
        # 'verbose': False,
        'random_state': 0,
        'verbose_eval': 20}

    ngb = NGBRegressor(**params).fit(X_train, Y_train)
    #save model
    joblib.dump(ngb,path_output+'/ngb_'+model_edp+'.pkl')
    
    # test Mean Squared Error and R2
    Y_test_pred = ngb.predict(X_test)
    test_MAE = mean_absolute_error(Y_test, Y_test_pred)
    test_MARE = mean_absolute_percentage_error(Y_test, Y_test_pred)
    test_MSE = mean_squared_error(Y_test, Y_test_pred)
    print("Test MAE", test_MAE)
    print("Test MARE", test_MARE)
    print("Test RMSE", test_MSE**0.5)
    print("Coefficient of determination R^2:", r2_score(Y_test, Y_test_pred))
    # train Mean Squared Error and R2
    Y_train_pred = ngb.predict(X_train)
    train_MAE = mean_absolute_error(Y_train, Y_train_pred)
    train_MARE = mean_absolute_percentage_error(Y_train, Y_train_pred)
    train_MSE = mean_squared_error(Y_train, Y_train_pred)
    print("Train MAE", train_MAE)
    print("Train MARE", train_MARE)
    print("Train RMSE", train_MSE**0.5)
    print("Coefficient of determination R^2:", r2_score(Y_train, Y_train_pred))
    
    calculate_R2()
    
    #load model
    ngb = joblib.load(path_output+'/ngb_'+model_edp+'.pkl')
       
    caculate_fragility(ngb)
    prob_US_build = caculate_prob(ngb)
    