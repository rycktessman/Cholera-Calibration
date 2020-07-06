#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: sample from the combined posterior from SIR and/or IMIS                  #
# Run this after running SIR1 and SIR2, and if running IMIS, after IMIS             #
# Last Updated: June 17, 2020                                                        #
#####################################################################################

import numpy as np
import pandas as pd
import random

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
    
    
    
m=5
imis=5
chain=1
samples=50000

#loading files - actual targets & pop/mortality
targets=pd.read_csv('calibration_targets4.csv', dtype={'Attribute':str})
targets.set_index('Attribute', inplace=True)

chain_start=1
chain_end=20
imis=5 #number of IMIS rounds run per chain (in addition to initial random searching step)
#initial best parameters from random draws step of IMIS/SIR
#includes scaling to be the same as params_norm
params_sir_best=pd.read_csv('params_sort'+str(chain_start)+'_0.csv', index_col=0) #best params from SIR only (before IMIS steps)
params_sir_best.insert(0, 'File', chain_start)
targets_sir_best=pd.read_csv('targets_sort'+str(chain_start)+'_0.csv', index_col=0)
targets_sir_best.insert(0, 'File', chain_start)
like_scale_temp=np.loadtxt('like_scale_sort'+str(chain_start)+'_0.csv', delimiter=",")
master_sum_temp=np.loadtxt('master_sum_'+str(chain_start)+'.csv', delimiter=",")
like_scale_temp[:,1]=like_scale_temp[:,1]*master_sum_temp
like_sir_best=np.copy(like_scale_temp)
master_sum_sir=0+master_sum_temp
for chain in range(chain_start, chain_end+1):
    print(chain)
    for sorted in range(0, 3):
        if sorted==0 & chain==1:
            next
        print(sorted)
        params_temp=pd.read_csv('params_sort'+str(chain)+'_0.csv', index_col=0)
        params_temp.insert(0, 'File', chain)
        params_sir_best=params_sir_best.append(params_temp)
        targets_temp=pd.read_csv('targets_sort'+str(chain)+'_0.csv', index_col=0)
        targets_temp.insert(0, 'File', chain)
        targets_sir_best=targets_sir_best.append(targets_temp)
        like_scale_temp=np.loadtxt('like_scale_sort'+str(chain)+'_0.csv', delimiter=",")
        master_sum_temp=np.loadtxt('master_sum_'+str(chain)+'.csv', delimiter=",")
        like_scale_temp[:,1]=like_scale_temp[:,1]*master_sum_temp
        like_sir_best=np.append(like_sir_best, like_scale_temp, axis=0)
        master_sum_sir+=master_sum_temp
        del params_temp
        del targets_temp
del like_scale_temp
del master_sum_temp



#sum of likelihoods across all chains - only if IMIS step was done (IMIS > 0)
if imis>0:
    master_sum_all=np.loadtxt('master_sum_all'+str(chain_start)+'.csv', delimiter=",") 
    for chain in range(chain_start+1, chain_end+1):
        master_sum_temp=np.loadtxt('master_sum_all'+str(chain)+'.csv', delimiter=",")
        master_sum_all+=master_sum_temp
        del master_sum_temp

#random unsorted set of 20,000 parameters from SIR step of IMIS = priors
params_prior=pd.read_csv('params_initial'+str(chain_start)+'_1000_0.csv', index_col=0).iloc[0:20000,:]
targets_prior=pd.read_csv('targets_initial'+str(chain_start)+'_1000_0.csv', index_col=0).iloc[0:20000,:]
like_prior=np.loadtxt('likelihood_initial'+str(chain_start)+'_1000_0.csv', delimiter=",")[0:20000]

#add columns so all target and all param files are same shape
params_sir_best.insert(1, 'Indices', "NA")
params_sir_best.insert(2, 'Distance', "NA")
params_sir_best.insert(3, 'Like', like_sir_best[:,1])
targets_sir_best.insert(1, 'Indices', "NA")
targets_sir_best.insert(2, 'Distance', "NA")
targets_sir_best.insert(3, 'Like', like_sir_best[:,1])
params_prior.insert(0, 'File', "1000_0")
params_prior.insert(1, 'Indices', np.arange(0,np.shape(params_prior)[0]))
params_prior.insert(2, 'Distance', "NA")
params_prior.insert(3, 'Like', like_prior)
targets_prior.insert(0, 'File', "1000_0")
targets_prior.insert(1, 'Indices', np.arange(0,np.shape(params_prior)[0]))
targets_prior.insert(2, 'Distance', "NA")
targets_prior.insert(3, 'Like', like_prior)
del like_prior
del like_sir_best

#append params_norm to params_best for master best_params (only if IMIS step done - e.g. imis >0), and sort
if imis>0:
    params_best=params_sir_best.append(params_norm)
    targets_best=targets_sir_best.append(targets_norm)
    params_best.reset_index(inplace=True, drop=True)
    targets_best.reset_index(inplace=True, drop=True)
    params_best.sort_values(by='Like', inplace=True, ascending=False)
    targets_best.sort_values(by='Like', inplace=True, ascending=False)
    params_best.reset_index(inplace=True, drop=True)
    targets_best.reset_index(inplace=True, drop=True)

if imis>0:
    print(np.sum(params_best.Like)/master_sum_all)
print(np.sum(params_sir_best.Like)/master_sum_sir)

#separate IMIS files out by chain
if imis>0:
    params_best['File']=params_best['File'].apply(str)
    params_best['Chain']=params_best['File'].str.split("_",n=1,expand=True)[0]
    params_best['Chain']=pd.to_numeric(params_best['Chain'], downcast='integer')
    params_best1=params_best.loc[params_best['Chain']==1]
    params_best2=params_best.loc[params_best['Chain']==2]
    params_best3=params_best.loc[params_best['Chain']==3]
    params_best4=params_best.loc[params_best['Chain']==4]
    params_best5=params_best.loc[params_best['Chain']==5]
    params_best6=params_best.loc[params_best['Chain']==6]
    params_best7=params_best.loc[params_best['Chain']==7]
    params_best8=params_best.loc[params_best['Chain']==8]
    params_best9=params_best.loc[params_best['Chain']==9]
    params_best10=params_best.loc[params_best['Chain']==10]
    params_best11=params_best.loc[params_best['Chain']==11]
    params_best12=params_best.loc[params_best['Chain']==12]
    params_best13=params_best.loc[params_best['Chain']==13]
    params_best14=params_best.loc[params_best['Chain']==14]
    params_best15=params_best.loc[params_best['Chain']==15]
    params_best16=params_best.loc[params_best['Chain']==16]
    params_best17=params_best.loc[params_best['Chain']==17]
    params_best18=params_best.loc[params_best['Chain']==18]
    params_best19=params_best.loc[params_best['Chain']==19]
    params_best20=params_best.loc[params_best['Chain']==20]
    
    targets_best['File']=targets_best['File'].apply(str)
    targets_best['Chain']=targets_best['File'].str.split("_",n=1,expand=True)[0]
    targets_best['Chain']=pd.to_numeric(targets_best['Chain'], downcast='integer')
    targets_best1=targets_best.loc[targets_best['Chain']==1]
    targets_best2=targets_best.loc[targets_best['Chain']==2]
    targets_best3=targets_best.loc[targets_best['Chain']==3]
    targets_best4=targets_best.loc[targets_best['Chain']==4]
    targets_best5=targets_best.loc[targets_best['Chain']==5]
    targets_best6=targets_best.loc[targets_best['Chain']==6]
    targets_best7=targets_best.loc[targets_best['Chain']==7]
    targets_best8=targets_best.loc[targets_best['Chain']==8]
    targets_best9=targets_best.loc[targets_best['Chain']==9]
    targets_best10=targets_best.loc[targets_best['Chain']==10]
    targets_best11=targets_best.loc[targets_best['Chain']==11]
    targets_best12=targets_best.loc[targets_best['Chain']==12]
    targets_best13=targets_best.loc[targets_best['Chain']==13]
    targets_best14=targets_best.loc[targets_best['Chain']==14]
    targets_best15=targets_best.loc[targets_best['Chain']==15]
    targets_best16=targets_best.loc[targets_best['Chain']==16]
    targets_best17=targets_best.loc[targets_best['Chain']==17]
    targets_best18=targets_best.loc[targets_best['Chain']==18]
    targets_best19=targets_best.loc[targets_best['Chain']==19]
    targets_best20=targets_best.loc[targets_best['Chain']==20]

#weighted file for each chain
if imis>0:
    for i in range(chain_start, chain_end+1):
        eval("params_best"+str(i))['Weights']=eval("params_best"+str(i)).Like/np.sum((eval("params_best"+str(i))).Like)
    params_post1=params_best1.sample(n=50000, replace=True, weights=params_best1.Weights)
    params_post2=params_best2.sample(n=50000, replace=True, weights=params_best2.Weights)
    params_post3=params_best3.sample(n=50000, replace=True, weights=params_best3.Weights)
    params_post4=params_best4.sample(n=50000, replace=True, weights=params_best4.Weights)
    params_post5=params_best5.sample(n=50000, replace=True, weights=params_best5.Weights)
    params_post6=params_best6.sample(n=50000, replace=True, weights=params_best6.Weights)
    params_post7=params_best7.sample(n=50000, replace=True, weights=params_best7.Weights)
    params_post8=params_best8.sample(n=50000, replace=True, weights=params_best8.Weights)
    params_post9=params_best9.sample(n=50000, replace=True, weights=params_best9.Weights)
    params_post10=params_best10.sample(n=50000, replace=True, weights=params_best10.Weights)
    params_post11=params_best11.sample(n=50000, replace=True, weights=params_best11.Weights)
    params_post12=params_best12.sample(n=50000, replace=True, weights=params_best12.Weights)
    params_post13=params_best13.sample(n=50000, replace=True, weights=params_best13.Weights)
    params_post14=params_best14.sample(n=50000, replace=True, weights=params_best14.Weights)
    params_post15=params_best15.sample(n=50000, replace=True, weights=params_best15.Weights)
    params_post16=params_best16.sample(n=50000, replace=True, weights=params_best16.Weights)
    params_post17=params_best17.sample(n=50000, replace=True, weights=params_best17.Weights)
    params_post18=params_best18.sample(n=50000, replace=True, weights=params_best18.Weights)
    params_post19=params_best19.sample(n=50000, replace=True, weights=params_best19.Weights)
    params_post20=params_best20.sample(n=50000, replace=True, weights=params_best20.Weights)
    for i in range(chain_start, chain_end+1):
        eval("params_post"+str(i)).sort_values(by=['Like'], ascending=False, inplace=True)
    targets_post1=targets_best1.loc[params_post1.index]
    targets_post2=targets_best2.loc[params_post2.index]
    targets_post3=targets_best3.loc[params_post3.index]
    targets_post4=targets_best4.loc[params_post4.index]
    targets_post5=targets_best5.loc[params_post5.index]
    targets_post6=targets_best6.loc[params_post6.index]
    targets_post7=targets_best7.loc[params_post7.index]
    targets_post8=targets_best8.loc[params_post8.index]
    targets_post9=targets_best9.loc[params_post9.index]
    targets_post10=targets_best10.loc[params_post10.index]
    targets_post11=targets_best11.loc[params_post11.index]
    targets_post12=targets_best12.loc[params_post12.index]
    targets_post13=targets_best13.loc[params_post13.index]
    targets_post14=targets_best14.loc[params_post14.index]
    targets_post15=targets_best15.loc[params_post15.index]
    targets_post16=targets_best16.loc[params_post16.index]
    targets_post17=targets_best17.loc[params_post17.index]
    targets_post18=targets_best18.loc[params_post18.index]
    targets_post19=targets_best19.loc[params_post19.index]
    targets_post20=targets_best20.loc[params_post20.index]


#generate weighted params_best file (posterior)
if imis>0:
    params_weights=params_best.Like/np.sum(params_best.Like) #confirmed that this equals Likes/master_sum_all
    params_post=params_best.sample(n=50000, replace=True, weights=params_weights)
    params_post=params_post.sort_values(by=['Like'], ascending=False)
    targets_post=targets_best.loc[params_post.index]

#version of IMIS posterior without parameter sets that fall outside the targets
targets_best_fit=targets_best.copy()
targets_best_fit.iloc[:,4:]=targets_best_fit.iloc[:,4:]-targets.loc['low (99)']
targets_best_fit.insert(0,'Flag',0)
targets_best_fit.Flag=1*(np.sum(targets_best_fit.iloc[:,5:]<0, axis=1)>0)
targets_best_fit=targets_best_fit.loc[targets_best_fit.Flag==0]
targets_best_fit.iloc[:,5:]=targets_best_fit.iloc[:,5:]+targets.loc['low (99)']
targets_best_fit.iloc[:,5:]=targets.loc['high (99)']-targets_best_fit.iloc[:,5:]
targets_best_fit.Flag=1*(np.sum(targets_best_fit.iloc[:,5:]<0, axis=1)>0)
targets_best_fit=targets_best_fit.loc[targets_best_fit.Flag==0]
targets_best_fit.iloc[:,5:]=-(targets_best_fit.iloc[:,5:]-targets.loc['high (99)'])
params_best_fit=params_best.iloc[targets_best_fit.index]
params_weights_fit=params_best_fit.Like/np.sum(params_best_fit.Like)
params_post_fit=params_best_fit.sample(n=50000, replace=True, weights=params_weights_fit)
params_post_fit=params_post_fit.sort_values(by=['Like'],ascending=False)
targets_post_fit=targets_best_fit.loc[params_post_fit.index]
targets_post_fit=targets_post_fit.iloc[:,1:] #remove flag column

params_sir_weights=params_sir_best.Like/np.sum(params_sir_best.Like)
params_sir_post=params_sir_best.sample(n=50000, replace=True, weights=params_sir_weights)
params_sir_post=params_sir_post.sort_values(by=['Like'], ascending=False)
targets_sir_post=targets_sir_best.loc[params_sir_post.index]

#fit within target CIs for performance metrics
if imis>0:
    fit=targets_post.copy().iloc[:,4:]
    fit=fit-targets.loc['low (99)']
    fit.insert(0, 'Flag', 0)
    fit.Flag=1*(np.sum(fit.iloc[:,1:13]<0, axis=1)>0)
    fit=fit.loc[fit.Flag==0]
    fit.iloc[:,1:13]=fit.iloc[:,1:13]+targets.loc['low (99)']
    fit.iloc[:,1:13]=targets.loc['high (99)']-fit.iloc[:,1:13]
    fit.Flag=1*(np.sum(fit.iloc[:,1:13]<0, axis=1)>0)
    fit=fit.loc[fit.Flag==0]
    fit_imis=np.shape(fit)[0]/np.shape(targets_post)[0]

fit=targets_sir_post.copy().iloc[:,4:17]
fit=fit-targets.loc['low (99)']
fit.insert(0, 'Flag', 0)
fit.Flag=1*(np.sum(fit.iloc[:,1:13]<0, axis=1)>0)
fit=fit.loc[fit.Flag==0]
fit.iloc[:,1:13]=fit.iloc[:,1:13]+targets.loc['low (99)']
fit.iloc[:,1:13]=targets.loc['high (99)']-fit.iloc[:,1:13]
fit.Flag=1*(np.sum(fit.iloc[:,1:13]<0, axis=1)>0)
fit=fit.loc[fit.Flag==0]
fit_sir=np.shape(fit)[0]/np.shape(targets_sir_post)[0]

if imis>0:
    params_post.to_csv('Params_IMIS_Post.csv') #weighted file to use for CEA
    targets_post.to_csv('Targets_IMIS_Post.csv') #weighted file to use for CEA
    params_post_fit.to_csv('Params_IMIS_Post_Fit.csv') #file only w/ params that fit in target CIs
    targets_post_fit.to_csv('Targets_IMIS_Post_Fit.csv')
    for i in range(chain_start, chain_end+1):
        eval("params_post"+str(i)).to_csv('Params_IMIS_Post'+str(i)+'.csv')
        eval("targets_post"+str(i)).to_csv('Targets_IMIS_Post'+str(i)+'.csv')
params_prior.to_csv('Params_Prior.csv')
targets_prior.to_csv('Targets_Prior.csv')
#params_best.to_csv('Params_Best.csv') #unweighted file for illustration only
#targets_best.to_csv('Targets_Best.csv') #unweighted file for illustration only
params_sir_post.to_csv('Params_SIR_Post.csv')
targets_sir_post.to_csv('Targets_SIR_Post.csv')
np.savetxt('params_sir_wts.csv', params_sir_weights, delimiter=",")

#randomly sample all IMIS so we can export a smaller file
if imis>0:
    params_best_out=params_best.sample(n=50000, replace=False, weights=None)
    params_best_out_indices=params_best_out.index
    targets_best_out=targets_best.loc[params_best_out_indices]
    params_best_out.to_csv('Params_IMIS_No_Wt.csv') #unweighted file of all IMIS params for multi chain comparison
    targets_best_out.to_csv('Targets_IMIS_No_Wt.csv')