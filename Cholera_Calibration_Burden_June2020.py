################################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                              #
# Purpose: estimate cholera burden from parameter sets (more output than calibration targets)  #
# Use this script after running calibration and combining SIR and IMIS posteriors              #
# Last Updated: June 18, 2020                                                                  #
################################################################################################

#load packages
import numpy as np
import pandas as pd

#load functions

#cholera_model_burden runs the same cholera model as for calibration (cholera_model)
    #but it generates different output, such as total cases and deaths
from cholera_model_burden import cholera_model_burden

#calc burden calculates cholera burden for each parameter set and saves it to file
#its input is calib_type, which can take on the following values (as strings): imis, sir, rs, naive, and prior
def calc_burden(calib_type):
    #load Bangladesh background mortality and population age distribution
    BGD_ASM=np.loadtxt(open("BGD_ASM.csv", "rb"), delimiter=",", skiprows=1)
    BGD_Pop=np.loadtxt(open("BGD_Pop.csv", "rb"), delimiter=",", skiprows=0)
    #load parameter sets to run through model
    if (calib_type=="imis") | (calib_type=="sir") | (calib_type=="rs") :
        params=pd.read_csv("params_post_" + calib_type + ".csv", index_col=0)
        params=params.iloc[:,2:]
    elif calib_type=="naive":
        params=pd.read_csv("naive_params.csv", index_col=0)
    elif calib_type=="prior":
        params=pd.read_csv("params_prior.csv", index_col=0)
        params=params.iloc[:,2:]
    else:
        print("Incorrect type provided - options include imis, sir, rs, naive, prior")
        return(None)
    n=np.shape(params)[0]
    output=cholera_model_burden(params, BGD_ASM, BGD_Pop, n) #run model using each param set 
    output.to_csv('cholera_burden_'+calib_type+'.csv')
    return(output)

if __name__=='__main__':
    calib_type="imis" #options: imis, sir, rs, naive, prior
    print(calib_type)
    output=calc_burden(calib_type)

