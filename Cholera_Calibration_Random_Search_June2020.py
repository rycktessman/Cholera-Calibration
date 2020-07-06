#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: run calibration of a cholera natural history model using Random Search   #
# Last Updated: June 8, 2020                                                        #
#####################################################################################

#load required packages and set default options
import numpy as np
import pandas as pd
from pyDOE import *

#load functions
from cholera_model import cholera_model #runs the model
from sorted_rank import sorted_rank #used to induce correlation in severity parameters
from sample_priors import sample_priors #samples prior parameter sets


#run_rs runs Random Search. Inputs include:
# 1. array: if running in parallel, tracks which instance this is (e.g. 1-1000)
# 2. iterations: loop through the n samples this many times (e.g. 5000)
# 3. n: number of samples to draw per iteration (e.g. 10,000)

def run_rs(array, iterations, n):
    #load data
    BGD_ASM=np.loadtxt(open("BGD_ASM.csv", "rb"), delimiter=",", skiprows=1) #load age-specific background mortality
    BGD_Pop=np.loadtxt(open("BGD_Pop.csv", "rb"), delimiter=",", skiprows=0) #load population age distribution
    targets=pd.read_csv('calibration_targets.csv', dtype={'Attribute':str}) #load targets
    targets.set_index('Attribute', inplace=True)

    #broadcast targets over ages
    max_age=116
    deaths_low=np.zeros((max_age))
    deaths_low[0:10]=targets.iloc[1,0]
    deaths_low[10:30]=targets.iloc[1,1]
    deaths_low[30:100]=targets.iloc[1,2]
    deaths_low[100:140]=targets.iloc[1,3]
    deaths_low[140:max_age-2]=targets.iloc[1,4]
    deaths_high=np.zeros((max_age))
    deaths_high[0:10]=targets.iloc[2,0]
    deaths_high[10:30]=targets.iloc[2,1]
    deaths_high[30:100]=targets.iloc[2,2]
    deaths_high[100:140]=targets.iloc[2,3]
    deaths_high[140:max_age-2]=targets.iloc[2,4]
    
    inc_age_low=np.zeros((max_age))
    inc_age_low[0:2]=targets.iloc[1,5]
    inc_age_low[2:10]=targets.iloc[1,6]
    inc_age_low[10:30]=targets.iloc[1,7]
    inc_age_low[30:max_age-2]=targets.iloc[1,8]
    inc_age_high=np.zeros((max_age))
    inc_age_high[0:2]=targets.iloc[2,5]
    inc_age_high[2:10]=targets.iloc[2,6]
    inc_age_high[10:30]=targets.iloc[2,7]
    inc_age_high[30:max_age-2]=targets.iloc[2,8]
    
    sev_inc_age_low=np.zeros((max_age))
    sev_inc_age_low[2:10]=targets.iloc[1,9]
    sev_inc_age_low[10:30]=targets.iloc[1,10]
    sev_inc_age_low[30:max_age-2]=targets.iloc[1,11]
    sev_inc_age_high=np.zeros((max_age))
    sev_inc_age_high[2:10]=targets.iloc[2,9]
    sev_inc_age_high[10:30]=targets.iloc[2,10]
    sev_inc_age_high[30:max_age-2]=targets.iloc[2,11]
    
    for k in range(0, iterations): 
        print(k)
        params=sample_priors(n) #sample parameter sets to be used in calibration function  
        results=cholera_model(params, BGD_ASM, BGD_Pop, n)  #calculate model outputs for each parameter set  
        
        #eliminating sets that aren't in deaths CIs (IHME)
        #broadcast results by age
        deaths_sim=np.zeros((max_age,n))
        deaths_sim[0:10,:]=results['%Deaths_04']
        deaths_sim[10:30,:]=results['%Deaths_514']
        deaths_sim[30:100,:]=results['%Deaths_1549']
        deaths_sim[100:140,:]=results['%Deaths_5069']
        deaths_sim[140:max_age-2,:]=results['%Deaths_70+']

        deaths_sim=pd.DataFrame(deaths_sim.T-deaths_low)
        deaths_sim=deaths_sim[(deaths_sim>=0).all(1)]
        deaths_sim=deaths_sim+deaths_low
        deaths_sim=deaths_sim-deaths_high
        deaths_sim=deaths_sim[(deaths_sim<=0).all(1)]
        deaths_sim=deaths_sim+deaths_high
        
        #eliminating sets that aren't in facility incidence by age CIs (meta-analysis)
        inc_age_sim=np.zeros((max_age,n))
        inc_age_sim[0:2,:]=results['HospInc_<1']
        inc_age_sim[2:10,:]=results['HospInc_14']
        inc_age_sim[10:30,:]=results['HospInc_514']
        inc_age_sim[30:max_age-2,:]=results['HospInc_15+']

        inc_age_sim=pd.DataFrame(inc_age_sim.T-inc_age_low)
        inc_age_sim=inc_age_sim[(inc_age_sim>=0).all(1)]
        inc_age_sim=inc_age_sim+inc_age_low
        inc_age_sim=inc_age_sim-inc_age_high
        inc_age_sim=inc_age_sim[(inc_age_sim<=0).all(1)]
        inc_age_sim=inc_age_sim+inc_age_high
        
        #eliminating sets that aren't in severe facility incidence CIs (meta-analysis)
        sev_inc_age_sim=np.zeros((max_age,n))
        sev_inc_age_sim[2:10,:]=results['SevHospInc_14']
        sev_inc_age_sim[10:30,:]=results['SevHospInc_514']
        sev_inc_age_sim[30:max_age-2,:]=results['SevHospInc_15+']

        sev_inc_age_sim=pd.DataFrame(sev_inc_age_sim.T-sev_inc_age_low)
        sev_inc_age_sim=sev_inc_age_sim[(sev_inc_age_sim>=0).all(1)]
        sev_inc_age_sim=sev_inc_age_sim+sev_inc_age_low
        sev_inc_age_sim=sev_inc_age_sim-sev_inc_age_high
        sev_inc_age_sim=sev_inc_age_sim[(sev_inc_age_sim<=0).all(1)]
        sev_inc_age_sim=sev_inc_age_sim+sev_inc_age_high
        
        #saving small subset of priors output - only do this once per entire calibration
        if (k==0) & (array==1):
            priors_all=params.iloc[0:5000,:]
            priors_all.to_csv('params_prior_rs.csv')
            del priors_all
            targets_all=results.iloc[0:5000,:]
            targets_all.to_csv('targets_prior_rs.csv')
            del targets_all
        
        #eliminating sets that don't fit all 3 types of targets
        death_index=deaths_sim.index.tolist()
        inc_age_index=inc_age_sim.index.tolist()
        sev_inc_age_index=sev_inc_age_sim.index.tolist()
        index1=set(death_index).intersection(inc_age_index)
        all_index=index1.intersection(sev_inc_age_index)
        #no need to save/export if it's empty
        if (len(all_index)==0):
            continue
        params_rs=params.take(list(all_index))
        targets_rs=results.take(list(all_index))
       
        #exporting to file
        params_rs.to_csv('params_rs'+str(k)+'_'+str(array)+'.csv')
        targets_rs.to_csv('targets_rs'+str(k)+'_'+str(array)+'.csv')
        print('done')


if __name__=='__main__':
    array = 1 #parallelizable variable - e.g. : int(sys.argv[1])
    iterations=5000
    n=10000
    print(array)
    print(iterations)
    print(n)
    #total draws=1000*5000*10000 = 50 billion
    
    run_rs(array, iterations, n)
    
    
    
        
    
