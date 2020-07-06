#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: sample from the combined posterior from Random Search                    #
# Run this after running Random Search, before running Burden, Graphs               #
# Last Updated: July 6, 2020                                                        #
#####################################################################################
#combine_rs combines accepted parameters from Random Search

import pandas as pd
import glob

def combine_rs():
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
            target = pd.read_csv(file, index_col=0)
            targets_all = targets_all.append(target)
        params_all.to_csv("params_post_rs.csv")
        targets_all.to_csv("targets_post_rs.csv")
    else:
        print("Error: no parameter sets accepted from Random Search")
    

if __name__=='__main__':
    combine_rs()

