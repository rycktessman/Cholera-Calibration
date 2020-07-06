#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: cholera natural history model function used with calibration scripts     #
# Last Updated: June 8, 2020                                                        #
#####################################################################################

import numpy as np
import pandas as pd

# Function inputs:
# 1. params - dataframe of parameter set samples to be run through the model
# 2. BGD_ASM - background mortality rates (age-stratified) loaded from data
# 3. BGD_Pop - Bangladesh population-age structure loaded from data
# 4. n - number of times to run the model (defined in calibration script)

def cholera_model(params, BGD_ASM, BGD_Pop, n):
    
    #natural history model
    cyc_len=0.5 
    T=3 #number of cycles - just running for 1 year for now
    max_age=int(116/cyc_len) #running for all ages for now
    poprisk=1 #% of urban pop. at risk for cholera
    
    ASM=np.column_stack((BGD_ASM[:,1], BGD_ASM[:,1])).flatten()
    BGD_ASM2=np.zeros((np.shape(ASM)[0],2))
    BGD_ASM2[:,0]=np.arange(0,max_age,1)*cyc_len
    BGD_ASM2[:,1]=ASM
    pop_size=BGD_Pop[:,1]
    params=np.array(params)
    
    cases_all=np.zeros((max_age,n))
    cases_hosp_6mo_1=np.zeros((max_age,n))
    cases_hosp_6mo_2=np.zeros((max_age,n))
    sev_cases_hosp_6mo_1=np.zeros((max_age,n))
    sev_cases_hosp_6mo_2=np.zeros((max_age,n))
    cholera_deaths_all=np.zeros((max_age,n))
    other_deaths_all=np.zeros((max_age,n))
    total_6mo_1=np.zeros((max_age,n))
    total_6mo_2=np.zeros((max_age,n))
    
    for i in range(0,n):
        #parameter arrays by age for model
        p_cholera=np.zeros((max_age))
        p_cholera[0:2]=params[i,0]
        p_cholera[2:10]=params[i,1] 
        p_cholera[10:30]=params[i,2]
        p_cholera[30:max_age]=params[i,3]
        p_die_severe_nt=np.zeros((max_age))
        p_die_severe_nt[0:2]=params[i,4]
        p_die_severe_nt[2:10]=params[i,5]
        p_die_severe_nt[10:30]=params[i,6]
        p_die_severe_nt[30:max_age]=params[i,7]
        p_die_severe_t=np.zeros((max_age))
        p_die_severe_t[0:2]=params[i,8]
        p_die_severe_t[2:10]=params[i,9]
        p_die_severe_t[10:30]=params[i,10]
        p_die_severe_t[30:max_age]=params[i,11]
        p_severe=np.zeros((max_age))
        p_severe[0:2]=params[i,12]
        p_severe[2:10]=params[i,13]
        p_severe[10:30]=params[i,14]
        p_severe[30:max_age]=params[i,15]
        p_treat_mild=np.zeros((max_age))
        p_treat_mild[0:2]=params[i,16]
        p_treat_mild[2:10]=params[i,17]
        p_treat_mild[10:30]=params[i,18]
        p_treat_mild[30:max_age]=params[i,19]
        p_treat_sev=np.zeros((max_age))
        p_treat_sev[0:2]=params[i,20]
        p_treat_sev[2:10]=params[i,21]
        p_treat_sev[10:30]=params[i,22]
        p_treat_sev[30:max_age]=params[i,23]  
               
        #keep track of outputs across simulations
        cases=np.zeros((max_age,T))
        cases_hosp=np.zeros((max_age,T))
        sev_cases_hosp=np.zeros((max_age,T))
        num_dead=np.zeros((max_age,T))
        num_alive=np.zeros((max_age,T)) #tracks number ppl alive at start of each period
        num_alive[0:max_age,0]=pop_size[0:max_age] 
        num_alive[:,1]=num_alive[:,0]
        cholera_deaths=np.zeros((max_age,T))
        other_deaths=np.zeros((max_age,T))
        total_alive=np.copy(num_alive)
        
        for t in range(1,T): #for calibration, run for 1 year (for policy simulation e.g., run for longer)
            other_deaths[:,t]+= BGD_ASM2[:,1]*num_alive[:,t].T
            total_alive[:,t]=(1-BGD_ASM2[:,1])*num_alive[:,t].T 
            cases[:,t]+=(p_cholera.T*total_alive[:,t])*poprisk 
            severe=p_severe.T*cases[:,t]
            mild=(1-p_severe.T)*cases[:,t]
            severe_treat=p_treat_sev*severe
            severe_untreat=(1-p_treat_sev)*severe
            mild_treat=p_treat_mild*mild
            mild_untreat=(1-p_treat_mild)*mild
            cases_hosp[:,t]+=severe_treat+mild_treat
            sev_cases_hosp[:,t]+=severe_treat
            cholera_deaths[:,t]+=p_die_severe_t*severe_treat+p_die_severe_nt*severe_untreat
            if t<T-1:
                num_alive[:,t+1]=np.roll(total_alive[:,t]-cholera_deaths[:,t],1, axis=0) #number remaining alive at beg. next period by current age group
        #saving output and params from each parameter set
        cases_hosp_6mo_1[:,i]=cases_hosp[:,1]
        cases_hosp_6mo_2[:,i]=cases_hosp[:,2]
        sev_cases_hosp_6mo_1[:,i]=sev_cases_hosp[:,1]
        sev_cases_hosp_6mo_2[:,i]=sev_cases_hosp[:,2]
        cholera_deaths_all[:,i]=cholera_deaths[:,1] #adding up deaths in period 1 only (because ages change)
        other_deaths_all[:,i]=other_deaths[:,1] #adding up deaths in period 1 only (because ages change)
        total_6mo_1[:,i]=total_alive[:,1]
        total_6mo_2[:,i]=total_alive[:,2]
    
  
    #results = output to compare w/ calibration targets
    results=pd.DataFrame({'%Deaths_04': np.sum(cholera_deaths_all[0:10,:],0)/np.sum((cholera_deaths_all[0:10,:]+other_deaths_all[0:10,:]),0)}, index=np.arange(0,n))
    results['%Deaths_514']=np.sum(cholera_deaths_all[10:30,:],0)/np.sum((cholera_deaths_all[10:30,:]+other_deaths_all[10:30,:]),0)
    results['%Deaths_1549']=np.sum(cholera_deaths_all[30:100,:],0)/np.sum((cholera_deaths_all[30:100,:]+other_deaths_all[30:100,:]),0)
    results['%Deaths_5069']=np.sum(cholera_deaths_all[100:140,:],0)/np.sum((cholera_deaths_all[100:140,:]+other_deaths_all[100:140,:]),0)
    results['%Deaths_70+']=np.sum(cholera_deaths_all[140:max_age-2,:],0)/np.sum((cholera_deaths_all[140:max_age-2,:]+other_deaths_all[140:max_age-2,:]),0)
    results['HospInc_<1']=np.sum(cases_hosp_6mo_1[0:2,:]+cases_hosp_6mo_2[0:2,:],0)/(np.sum(total_6mo_1[0:2,:]+total_6mo_2[0:2,:],0)/2)
    results['HospInc_14']=np.sum(cases_hosp_6mo_1[2:10,:]+cases_hosp_6mo_2[2:10,:],0)/(np.sum(total_6mo_1[2:10,:]+total_6mo_2[2:10,:],0)/2)
    results['HospInc_514']=np.sum(cases_hosp_6mo_1[10:30,:]+cases_hosp_6mo_2[10:30,:],0)/(np.sum(total_6mo_1[10:30,:]+total_6mo_2[10:30,:],0)/2)
    results['HospInc_15+']=np.sum(cases_hosp_6mo_1[30:max_age-2,:]+cases_hosp_6mo_2[30:max_age-2,:],0)/(np.sum(total_6mo_1[30:max_age-2,:]+total_6mo_2[30:max_age-2,:],0)/2)
    results['SevHospInc_14']=np.sum(sev_cases_hosp_6mo_1[2:10,:]+sev_cases_hosp_6mo_2[2:10,:],0)/(np.sum(total_6mo_1[2:10,:]+total_6mo_2[2:10,:],0)/2)
    results['SevHospInc_514']=np.sum(sev_cases_hosp_6mo_1[10:30,:]+sev_cases_hosp_6mo_2[10:30,:],0)/(np.sum(total_6mo_1[10:30,:]+total_6mo_2[10:30,:],0)/2)
    results['SevHospInc_15+']=np.sum(sev_cases_hosp_6mo_1[30:max_age-2,:]+sev_cases_hosp_6mo_2[30:max_age-2,:],0)/(np.sum(total_6mo_1[30:max_age-2,:]+total_6mo_2[30:max_age-2,:],0)/2)

    return(results)