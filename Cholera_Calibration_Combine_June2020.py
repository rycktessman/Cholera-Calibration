#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: sample from the combined posterior from SIR and/or IMIS                  #
# Run this after running SIR1 and SIR2, and if running IMIS, after IMIS             #
# Last Updated: July 6, 2020                                                        #
#####################################################################################

import numpy as np
import pandas as pd

#sample_post combines posterior samples from IMIS and/or SIR. Inputs include:
    #1. imis: if only SIR was run, set imis=0. Otherwise, set to equal number of rounds of imis conducted (equal to end-start from Cholera_Calibration_IMIS)
    #2. m: number of sorted files from SIR2
    #3. if multiple chains of IMIS were run, which chain is this
    #4. samples: number of posterior samples to generate
    
def sample_post(m, imis, chain, samples):
    #load random set of prior samples from IMIS/SIR to compare against posterior
    params_prior=pd.read_csv('params_sir'+str(chain)+'_1_0.csv', index_col=0)
    targets_prior=pd.read_csv('targets_sir'+str(chain)+'_1_0.csv', index_col=0)
    like=np.loadtxt('likelihood_sir'+str(chain)+'_1_0.csv', delimiter=",")
    params_prior.insert(0, "Source", "SIR_prior")
    params_prior.insert(1, "Like", like)
    targets_prior.insert(0, "Source", "SIR_prior")
    targets_prior.insert(1, "Like", like)
    params_prior.to_csv('params_prior.csv')
    targets_prior.to_csv('targets_prior.csv')    
    
    #load and combine parameters from SIR
    for i in range(0, m):
        print(i)
        if (i==0):
            params_sir_all=pd.read_csv('params_sort_sir'+str(chain)+'_0.csv', index_col=0) 
            targets_sir_all=pd.read_csv('targets_sort_sir'+str(chain)+'_0.csv', index_col=0) 
            like=np.loadtxt('like_sort_sir'+str(chain)+'_0.csv', delimiter=",")[:,1]
            params_sir_all.insert(0, "Source", "SIR"+str(i))
            params_sir_all.insert(1, "Like", like)
            targets_sir_all.insert(0, "Source", "SIR"+str(i))
            targets_sir_all.insert(1, "Like", like)
        else:
            params_temp=pd.read_csv('params_sort_sir'+str(chain)+'_' +str(i)+'.csv', index_col=0)
            targets_temp=pd.read_csv('targets_sort_sir'+str(chain)+'_' +str(i)+'.csv', index_col=0)
            like=np.loadtxt('like_sort_sir'+str(chain)+'_'+str(i)+'.csv', delimiter=",")[:,1]
            params_temp.insert(0, "Source", "SIR"+str(i))
            params_temp.insert(1, "Like", like)
            targets_temp.insert(0, "Source", "SIR"+str(i))
            targets_temp.insert(1, "Like", like)
            params_sir_all=params_sir_all.append(params_temp)
            targets_sir_all=targets_sir_all.append(targets_temp)
            del params_temp
            del targets_temp
    params_sir_all.reset_index(inplace=True, drop=True)
    targets_sir_all.reset_index(inplace=True, drop=True)
    
    #sample from params_sir w/ weights to generate posterior distribution samples
    weights=params_sir_all.Like/np.sum(params_sir_all.Like) 
    params_sir_post=params_sir_all.sample(n=samples, replace=True, weights=weights)
    params_sir_post=params_sir_post.sort_values(by=['Like'], ascending=False)
    targets_sir_post=targets_sir_all.loc[params_sir_post.index]
    
    #save posterior samples to file
    params_sir_post.to_csv('params_post_sir.csv')
    targets_sir_post.to_csv('targets_post_sir.csv')  
    
    if imis>0: #load and combine parameters from IMIS with those from SIR
        params_imis_all=params_sir_all.copy()
        targets_imis_all=targets_sir_all.copy()
        for i in range(0, imis):
            print(i)
            params_temp=pd.read_csv('params_imis'+str(chain)+'_' +str(i)+'.csv', index_col=0)
            targets_temp=pd.read_csv('targets_imis'+str(chain)+'_' +str(i)+'.csv', index_col=0)
            like=np.loadtxt('likelihood_imis'+str(chain)+'_'+str(i)+'.csv', delimiter=",")
            params_temp.insert(0, "Source", "IMIS"+str(i))
            params_temp.insert(1, "Like", like)
            targets_temp.insert(0, "Source", "IMIS"+str(i))
            targets_temp.insert(1, "Like", like)
            params_imis_all=params_imis_all.append(params_temp)
            targets_imis_all=targets_imis_all.append(targets_temp)
            del params_temp
            del targets_temp
        params_imis_all.reset_index(inplace=True, drop=True)
        targets_imis_all.reset_index(inplace=True, drop=True)
    
        #sample from params_imis w/ weights to generate posterior distribution samples
        weights=params_imis_all.Like/np.sum(params_imis_all.Like) 
        params_imis_post=params_imis_all.sample(n=samples, replace=True, weights=weights)
        params_imis_post=params_imis_post.sort_values(by=['Like'], ascending=False)
        targets_imis_post=targets_imis_all.loc[params_imis_post.index]
        
        #save posterior samples to file
        params_imis_post.to_csv('params_post_imis.csv')
        targets_imis_post.to_csv('targets_post_imis.csv')
    
    
    


if __name__=='__main__':
    m=5 #number of sorted files saved from SIR
    imis=5 #number of rounds of IMIS run
    chain=1 #if combining multiple chains, which chain is this
    samples=50000 #number of posterior samples to draw
    
    sample_post(m, imis, chain, samples)