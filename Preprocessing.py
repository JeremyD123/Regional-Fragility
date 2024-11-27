# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

def get_data(model_edp):
    path_input = "D:/Regional F/InputData"
    # data_file_name = join(path, 'Build_Damage_Database.xlsx')
    data_file = pd.read_excel(path_input+'/Regional Building Damage.xlsx',sheet_name=model_edp)
    data = data_file.iloc[:,:].values
    data = np.nan_to_num(data)
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.quantile(data[:,-1],0.25)
    Q3 = np.quantile(data[:,-1],0.75)
    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1-0.5*(Q3-Q1)
    upper_bound = Q3+2.5*(Q3-Q1)
    
    cleaned_data = data[(data[:,-1] >= lower_bound) & (data[:,-1] <= upper_bound)]
    
    X = cleaned_data[:,:-1]
    Y = cleaned_data[:,-1]
    
    return X, Y
       
if __name__ == "__main__":
    model_edp = 'PFA'
    dimSPs = 8                           # dim of structural parameters
    X, Y = get_data(model_edp)   
    # Standardize features
    X[:,dimSPs:] = np.log(X[:,dimSPs:])
    Y = np.log(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=5) #1 2
    # save standardize model and data
    path_output = "D:/Regional F/OutputData"
    np.savez(path_output+'/dataXY_'+model_edp+'.npz',
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test) 
    