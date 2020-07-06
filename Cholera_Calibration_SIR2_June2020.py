#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: sum likelihoods and identify highest likelihood parameter sets from SIR  #
    #to use in sampling from the posterior and to use in IMIS                       #
# Run this file after running Cholera_Calibration_SIR1_June2020                     #
# Run this file before running Cholera_Calibration_IMIS_June2020                    #
# Last Updated: June 15, 2020                                                       #
#####################################################################################

import numpy as np
import pandas as pd

#calc_like_sum sums the likelihoods from each parameter set from SIR - 
    #(SIR also functions as the first step in IMIS)
    
#sort_like takes a subset of the m*n parameter sets with the highest likelihood from SIR,
    #sort them, and saves them to CSV files to use for sampling from the posterior 
    #and for use in IMIS
    #in IMIS, the highest likelihood parameter set and those closest to it
    #are then used to parameterize the mean and covariance, respectively, of a 
    #multivariate normal distribution to sample parameters from more densely.
    #if IMIS=1, then sort_like also finds the b parameter sets closest to the best
    #and calculate the covariance across these parameter sets for use in IMIS
   
# Inputs to these functions include:
# 1. arrays: maximum "array" from Cholera_Calibration_SIR1_June2020.py
# 2. iterations: same as from SIR1: looped through the n samples this many times (e.g. 100)
# 3. chain: same as from SIR1. if there are multiple chains to combine later, track which chain this is
# 4. n: same as from SIR1. number of samples drawn per iteration (e.g. 10,000)
# 5. m: number of files with sorted likelihoods, each of size n, to output for IMIS2
# 6. imis: whether IMIS will be run (if so, finds parameter sets closest to highest likelihood)
# 7. b: number of parameter sets to find that are closest to highest likelihood (if imis=1)
    #if imis=0, set b=0
    
def calc_like_sum(arrays, iterations, chain):
    like_sum=0
    #open likelihood files and sum likelihoods over all files
    for i in range(1, arrays+1):
        print(i)
        for j in range(0, iterations):
            like=np.loadtxt(open('likelihood_sir'+str(chain)+'_'+str(i)+'_'+str(j)+'.csv', "rb"), delimiter=",")
            like_sum+=np.sum(like)
    print(like_sum)
    like_sum=np.matrix(like_sum)
    np.savetxt('like_sum_sir'+str(chain)+'.csv', like_sum, delimiter=",")
    return(like_sum)

def sort_like(iterations, arrays, n, chain, m, imis, b):
    #initialize dataframes to store highest-likelihood parameter sets
    like_out=np.zeros((n,2))
    targets_out=pd.read_csv('targets_sir'+str(chain)+'_1_0.csv',index_col=0)
    params_out=pd.read_csv('params_sir'+str(chain)+'_1_0.csv',index_col=0)
    #calculate covariance & inverse cov of "all" parameter sets (random set because all is too many)
    if (imis==1):
        cov_all=np.cov(np.transpose(params_out)) 
        inv_cov_all=np.linalg.inv(cov_all)
    #set values to 0 for storing sorted parameter sets/output (targets)
    for col in targets_out.columns:
        targets_out[col].values[:]=0
    for col in params_out.columns:
        params_out[col].values[:]=0
    #define min and max bounds on likelihoods to keep (0 and infinity to start)
    #min bound is needed to keep highest likelihood parameter set
    #max bound is needed because we have multiple files - e.g.
        #param sets in file 2 should have lower likelihoods than param sets in file 1
    min_bound=0 
    max_bound=float("inf") 
    #if running IMIS next, initialize blank dataframe to store closest b parameter sets to highest likelihood
    if imis==1: 
        params_close=params_out.copy()
        params_close.insert(0, "Distance", float("inf")) #distance from best param - large to start
        params_close.insert(1, "Like", 0) #to see if distances and likelihoods closely correspond
        #targets_close=targets_out.copy()
        #targets_close.insert(1, "Distance", float("inf")) #distance from best param - large to start
        #targets_close.insert(2, "Like", 0) #to see if distances and likelihoods closely correspond
    for file in range(0, m): #loop through all parameter set files m times
        print(file)
        for i in range(1, arrays+1):
            for j in range(0, iterations):
                #load files
                params=pd.read_csv('params_sir'+str(chain)+'_'+str(i)+'_'+str(j)+'.csv', index_col=0)
                targets=pd.read_csv('targets_sir'+str(chain)+'_'+str(i)+'_'+str(j)+'.csv',index_col=0)
                like=np.loadtxt(open('likelihood_sir'+str(chain)+'_'+str(i)+'_'+str(j)+'.csv', "rb"), delimiter=",")
                #only need to find b closest parameter sets once (per file)
                #after initially identifying highest likelihood param set
                if (imis==1) & (file==1):
                    #calculate sorted distances between params and best set
                    distances_temp=np.sum(np.multiply(np.dot(np.array(params-param_best),inv_cov_all),np.array(params-param_best)),axis=1)
                    distances=np.sqrt(distances_temp)
                    distances=np.column_stack((np.arange(0,n),distances,like)) #add index & likes to keep track
                    sorted_distances=distances[distances[:,1].argsort()] #sort by distances
                    #append distances that are less than max distance in running params_b
                    indices_temp=np.where(sorted_distances[:,1]<np.max(params_close.Distance))[0] #just the first x indices
                    distances_keep=sorted_distances[indices_temp,:]
                    params_keep=params.loc[distances_keep[:,0]].copy() #params corresponding to indices column in distances_keep
                    params_keep.insert(0, "Distance", distances_keep[:,1])
                    params_keep.insert(1, "Like", distances_keep[:,2])
                    #targets_keep=targets.loc[distances_keep[:,0]].copy()
                    #targets_keep.insert(0, "Distance", distances_keep[:,1])
                    #targets_keep.insert(1, "Like", distances_keep[:,2])
                    #keep b lowest distances
                    params_close=params_close.append(params_keep)
                    params_close=params_close.sort_values(by=['Distance'])
                    params_close=params_close.reset_index(drop=True)
                    params_close=params_close.loc[0:b-1]
                    #targets_close=targets_close.append(targets_keep)
                    #targets_close=targets_close.sort_values(by=['Distance'])
                    #targets_close=targets_close.reset_index(drop=True)
                    #targets_close=targets_close.loc[0:b-1]   
                #sort by likelihood
                like_new=np.zeros((n,2))
                like_new[:,1]=like
                like_all=np.concatenate((like_out, like_new))
                like_all[:,0]=np.arange(0,np.shape(like_all)[0]) #reset index
                index=np.where((like_all[:,1]>min_bound)*(like_all[:,1]<max_bound))[0]
                like_all=like_all[index]
                like_all=like_all[like_all[:,1].argsort()][::-1] #sort descending by likelihood
                like_out=like_all[0:n, :] #keeps only the 1st (highest) n elements (or fewer if < n elements)
                #keep params and targets that correspond to index kept in like_scale_out
                params_out_all=params_out.append(params)
                params_out_all=params_out_all.reset_index(drop=True)
                params_out=params_out_all.loc[like_out[:,0]]
                targets_out_all=targets_out.append(targets)
                targets_out_all=targets_out_all.reset_index(drop=True)
                targets_out=targets_out_all.loc[like_out[:,0]]
                #reset indices again for next file
                like_out[:,0]=np.arange(0,np.shape(like_out)[0])
                params_out=params_out.reset_index(drop=True)
                targets_out=targets_out.reset_index(drop=True)
        if np.shape(like_out)[0]==0:
            print('no more non-zero scaled likelihoods')
            print('last file saved is empty')
            break
        #save sorted outputs to file
        np.savetxt('like_sort_sir'+str(chain)+'_'+str(file)+'.csv', like_out, delimiter=",")
        params_out.to_csv('params_sort_sir'+str(chain)+'_'+str(file)+'.csv')
        targets_out.to_csv('targets_sort_sir'+str(chain)+'_'+str(file)+'.csv')
        #if running IMIS, identify highest-likelihood parameter set from 1st round of sorting
        if (imis==1) & (file==0):
            param_best=params_out.loc[0].copy()
        #if running IMIS, output covariance of b closest parameter sets
        #also output closest parameter sets
        if (imis==1) & (file==1):
            weights_close=params_close.Like/np.sum(params_close.Like)
            cov_close=np.cov(np.transpose(params_close.iloc[:,2:].copy()), aweights=weights_close)
            np.savetxt('cov_close_sir'+str(chain)+'.csv', cov_close, delimiter=",")
            params_close.to_csv('params_close_sir'+str(chain)+'.csv')
            #targets_close.to_csv('targets_close_sir'+str(chain)+'.csv')
        #reset blank dataframes and set max bound for next round
        max_bound=np.min(like_out[:,1])
        like_out=np.zeros((n,2))
        targets_out=pd.read_csv('targets_sir'+str(chain)+'_1_0.csv',index_col=0)
        for col in targets_out.columns:
            targets_out[col].values[:]=0
        params_out=pd.read_csv('params_sir'+str(chain)+'_1_0.csv',index_col=0)
        for col in params_out.columns:
            params_out[col].values[:]=0

if __name__=='__main__':
    arrays = 2 #number of arrays run from SIR (the parallelizable variable)
    iterations = 100 #number of runs per array from SIR
    n = 10000 #n from SIR
    chain = 1 #if mixing multiple SIR or IMIS chains
    m = 5 #save 50,000 highest likelihood parameter sets
    imis = 1 #running IMIS next - find b closest parameter sets to highest likelihood
    b = 100000 #find 100,000 closest parameter sets to highest likelihood
    print(arrays)
    print(iterations)
    print(chain)
    print(imis)
    print(b)
    like_sum = calc_like_sum(arrays, iterations, chain)
    sort_like(iterations, arrays, n, chain, m, imis, b)