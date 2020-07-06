#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: combine params from Random Search and calculate likelihood               #
# Run this after running Random Search, before running Burden, Graphs               #
# Last Updated: July 6, 2020                                                        #
#####################################################################################
#combine_rs combines accepted parameters from Random Search

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import glob

def combine_rs():
    #load targets from file
    targets=pd.read_csv('calibration_targets.csv', dtype={'Attribute':str}) #load targets
    targets.set_index('Attribute', inplace=True)
    cov_targets=np.loadtxt("cov.csv", delimiter=",") #load target covariance matrix for multinormal distribution   
    #loop over all random search accepted parameters
    param_files = glob.glob('params_rs*.csv')
    target_files = glob.glob('targets_rs*.csv')
    if len(param_files) >= 1:
        params_all = pd.read_csv(param_files[0], index_col=0)
        targets_all = pd.read_csv(target_files[0], index_col=0)
        print(param_files[0])
        for file in param_files[1:]:
            print(file)
            param = pd.read_csv(file, index_col=0)
            params_all = params_all.append(param)
        for file in target_files[1:]:
            output = pd.read_csv(file, index_col=0)
            targets_all = targets_all.append(output)
            
        #calculate likelihood and add to files
        targets_temp = targets_all.copy()
        targets_temp.iloc[:,0:5]=np.log(targets_temp.iloc[:,0:5]) #log transformation of deaths outputs
        like=sp.stats.multivariate_normal.pdf(targets_temp, mean=targets.loc['mean_plus'], cov=cov_targets)
        params_all.insert(0, "Source", "RS")
        params_all.insert(1, "Like", like)
        targets_all.insert(0, "Source", "RS")
        targets_all.insert(1, "Like", like)

        params_all.to_csv("params_post_rs.csv")
        targets_all.to_csv("targets_post_rs.csv")
    else:
        print("Error: no parameter sets accepted from Random Search")
    
if __name__=='__main__':
    combine_rs()

