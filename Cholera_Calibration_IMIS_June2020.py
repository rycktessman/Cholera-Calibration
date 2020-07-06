#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: run multivariate normal sampling part of IMIS                            #
# Run this file after running Cholera_Calibration_SIR2_June2020                     #
# Last Updated: June 16, 2020                                                       #
#####################################################################################

#load packages
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

#load functions
from cholera_model import cholera_model

def run_imis(arrays, iterations, n, chain, m, start, end, b, c):
    #load data used to calculate GOF
    BGD_ASM=np.loadtxt(open("BGD_ASM.csv", "rb"), delimiter=",", skiprows=1) #load age-specific background mortality
    BGD_Pop=np.loadtxt(open("BGD_Pop.csv", "rb"), delimiter=",", skiprows=0) #load population age distribution
    targets=pd.read_csv('calibration_targets.csv', dtype={'Attribute':str}) #load targets
    targets.set_index('Attribute', inplace=True)
    cov_targets=np.loadtxt("cov.csv", delimiter=",") #load target covariance matrix for multinormal distribution
    
    #load likelihood sum from SIR
    #find highest-likelihood parameter set so far (from SIR and/or previous round of IMIS)
    if start==0:
        print("Initializing IMIS")
        like_sum=np.loadtxt(open('like_sum_sir'+str(chain)+'.csv', "rb"), delimiter=",")
        params=pd.read_csv('params_sort_sir'+str(chain)+'_0.csv', index_col=0)
        param_best=params.loc[0].copy()
    else:
        print('Starting IMIS from round', str(start))
        like_sum=np.loadtxt(open('like_sum_imis'+str(chain)+'_'+str(start-1)+'.csv', "rb"), delimiter=",")
        param_best=pd.read_csv('param_best'+str(chain)+'_'+str(start-1)+'.csv', header=None, index_col=0)
        param_best.rename(columns={1:0}, inplace=True)
        param_best=param_best[0]
    
    #calculate covariance & inverse cov of "all" parameter sets (random set because all is too many)
    random_params=pd.read_csv('params_sir'+str(chain)+'_1_0.csv', index_col=0)
    cov_all=np.cov(np.transpose(random_params)) 
    inv_cov_all=np.linalg.inv(cov_all)
    
    #load other "candidate" closest parameter sets from file
    #1. b closest parameter sets from SIR2
    params_close_all=pd.read_csv('params_close_sir'+str(chain)+'.csv', index_col=0) #includes additional columns
    #2. highest m*n likelihood parameter sets from SIR2
    for i in range(0,m):
        params=pd.read_csv('params_sort_sir'+str(chain)+'_'+str(i)+'.csv', index_col=0)
        like=np.loadtxt(open('like_sort_sir'+str(chain)+'_'+str(i)+'.csv', "rb"), delimiter=",")[:,1]
        params.insert(0, "Distance", np.zeros(n)) #placeholder
        params.insert(1, "Like", like)
        params_close_all = params_close_all.append(params)
    #drop duplicates (some closest and highest-likelihood might be the same)
    params_close_all=params_close_all.reset_index(drop=True)
    params_close_all.drop_duplicates(subset=['pCholera_0', 'pCholera_1', 'pCholera_2'], inplace=True) #just specify a few columns - this is enough to remove dups
    
    print(np.max(params_close_all.Like))
    #run IMIS as many times as specified by start and end
    for round in range(start, end):
        print(round)       
        if round==0: #first time running IMIS - load covariance from SIR2 output
            cov_close=np.loadtxt(open('cov_close_sir'+str(chain)+'.csv', "rb"), delimiter=",")
        #after 1st time running IMIS, identify b closest parameter sets from:
            #1. b closest parameter sets from SIR2
            #2. highest m*n likelihood parameter sets from SIR2
            #3. new multivariate normal samples from previous rounds of IMIS
        else: #calculate/recalculate distances across all candidate parameter sets
            params=params_close_all.iloc[:,2:] #remove distance and like columns
            distances_temp=np.sum(np.multiply(np.dot(np.array(params-param_best), inv_cov_all), np.array(params-param_best)),axis=1)
            distances=np.sqrt(distances_temp)
            params_close_all["Distance"] = distances
            params_close = params_close_all.sort_values(by=['Distance'])
            params_close=params_close.reset_index(drop=True)
            params_close=params_close.loc[0:b-1] #take b closest only
            #calculate cov of b closest params
            weights_close=params_close.Like/np.sum(params_close.Like)
            cov_close=np.cov(np.transpose(params_close.iloc[:,2:].copy()), aweights=weights_close)
        
        #sample c parameter sets from multivariate normal using param_best and cov_close, append to params
        params_new=np.random.multivariate_normal(param_best, cov_close, size=c)
        #remove parameters not in the range [0,1]
        params_new=params_new[(params_new>0).all(1)]
        params_new=params_new[(params_new<1).all(1)]
        print(np.shape(params_new)[0]/c) # % of multinormals in the range [0,1]
        params_new=pd.DataFrame(params_new, columns=list(random_params))
        
        #evaluate GOF of new parameter sets
        output=cholera_model(params_new, BGD_ASM, BGD_Pop, np.shape(params_new)[0]) #run model using each param set 
        output_temp=output.copy()
        output_temp.iloc[:,0:5]=np.log(output_temp.iloc[:,0:5]) #log transformation of deaths outputs
        like=sp.stats.multivariate_normal.pdf(output_temp, mean=targets.loc['mean_plus'], cov=cov_targets)

        #saving IMIS outputs
        output=pd.DataFrame(output, columns=list(targets))
        output.to_csv('targets_imis'+str(chain)+'_'+str(round)+'.csv')
        params_new.to_csv('params_imis'+str(chain)+'_'+str(round)+'.csv')
        np.savetxt('likelihood_imis'+str(chain)+'_'+str(round)+'.csv', like, delimiter=',')
        
        #append to params_close
        params_new.insert(0, "Distance", np.zeros(np.shape(params_new)[0])) #placeholder
        params_new.insert(1, "Like", like)
        params_close_all = params_close_all.append(params_new)
        print(np.max(params_close_all.Like))

if __name__=='__main__':
    arrays = 1000 #number of arrays from SIR
    iterations = 100 #number of iterations per array from SIR
    n = 10000 #number of draws per iteration (e.g. per csv file) from SIR
    chain = 1 #if mixing multiple SIR or IMIS chains
    m = 5 #files of parameter sets from SIR sorted by likelihood (from SIR2)
    start = 0 #0 if initializing IMIS for first time, otherwise round of IMIS to start from
    end = 5 #number of rounds of multivariate normal sampling to run
    b = 100000 #find 100,000 closest parameter sets to highest likelihood
    c = 100000 #sample 100,000 new param sets from multinormal distribution for each IMIS round
    
    print(arrays)
    print(iterations)
    print(n)
    print(chain)
    print(m)
    print(start)
    print(end)
    print(b)
    print(c)
    run_imis(arrays, iterations, n, chain, m, start, end, b, c)

