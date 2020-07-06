#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: run calibration of a cholera natural history model using SIR - pt 1      #
# Run this before running Cholera_Calibration_SIR2 and Cholera_Calibration_IMIS     #
# Last Updated: June 8, 2020                                                        #
#####################################################################################

#load required packages and set default options
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from pyDOE import *

#load functions
from cholera_model import cholera_model
from sorted_rank import sorted_rank
from sample_priors import sample_priors

#run_sir runs SIR (also the first part of IMIS). Inputs include:
# 1. array: if running in parallel, tracks which instance this is (e.g. 1-1000)
# 2. iterations: loop through the n samples this many times (e.g. 100)
# 3. n: number of samples to draw per iteration (e.g. 10,000)
# 4. chain: if there are multiple chains to combine later, track which chain this is

def run_sir(array, iterations, n, chain):
    #load data
    BGD_ASM=np.loadtxt(open("BGD_ASM.csv", "rb"), delimiter=",", skiprows=1) #load age-specific background mortality
    BGD_Pop=np.loadtxt(open("BGD_Pop.csv", "rb"), delimiter=",", skiprows=0) #load population age distribution
    targets=pd.read_csv('calibration_targets.csv', dtype={'Attribute':str}) #load targets
    targets.set_index('Attribute', inplace=True)
    cov_targets=np.loadtxt("cov.csv", delimiter=",") #load target covariance matrix for multinormal distribution   

    for k in range(0, iterations):
        print(k)
        params=sample_priors(n) #sample parameter sets from prior distributions
        output=cholera_model(params, BGD_ASM, BGD_Pop, n) #run model using each param set 
        output_temp=output.copy()
        output_temp.iloc[:,0:5]=np.log(output_temp.iloc[:,0:5]) #log transformation of deaths outputs
        like=sp.stats.multivariate_normal.pdf(output_temp, mean=targets.loc['mean_plus'], cov=cov_targets)

        #saving SIR outputs
        output=pd.DataFrame(output, columns=list(targets))
        output.to_csv('targets_sir'+str(chain)+'_'+str(array)+'_'+str(k)+'.csv')
        params.to_csv('params_sir'+str(chain)+'_'+str(array)+'_'+str(k)+'.csv')
        np.savetxt('likelihood_sir'+str(chain)+'_'+str(array)+'_'+str(k)+'.csv', like, delimiter=',')
    
if __name__=='__main__':
    array = 1 #parallelizable variable - e.g. int(sys.argv[1])
    iterations=100
    n=10000
    chain=1 #if mixing multiple SIR or IMIS chains
    print(array)
    print(iterations)
    print(n)
    print(chain)
    run_sir(array, iterations, n, chain)