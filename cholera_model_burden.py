##############################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                            #
# Purpose: cholera natural history model function used to estimate burden after calibration  #
# Last Updated: June 18, 2020                                                                #
##############################################################################################

import numpy as np
import pandas as pd

# Function inputs:
# 1. params - dataframe of parameter set samples to be run through the model
# 2. BGD_ASM - background mortality rates (age-stratified) loaded from data
# 3. BGD_Pop - Bangladesh population-age structure loaded from data
# 4. n - number of times to run the model (defined in calibration script)

def cholera_model_burden(params, BGD_ASM, BGD_Pop, n):
    
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
    sev_cases_all=np.zeros((max_age,n))
    cases_hosp_all=np.zeros((max_age,n))
    sev_cases_hosp_all=np.zeros((max_age,n))
    cholera_deaths_all=np.zeros((max_age,n))
    cholera_deaths_hosp_all=np.zeros((max_age,n))
    total_deaths_all=np.zeros((max_age,n))
    total_alive_all=np.zeros((max_age,n))
    
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
        sev_cases=np.zeros((max_age,T)) #NEW
        cases_hosp=np.zeros((max_age,T))
        sev_cases_hosp=np.zeros((max_age,T))
        num_dead=np.zeros((max_age,T))
        num_alive=np.zeros((max_age,T)) #tracks number ppl alive at start of each period
        num_alive[0:max_age,0]=pop_size[0:max_age] 
        num_alive[:,1]=num_alive[:,0]
        cholera_deaths=np.zeros((max_age,T))
        cholera_deaths_hosp=np.zeros((max_age,T)) #NEW
        other_deaths=np.zeros((max_age,T))
        total_alive=np.copy(num_alive)
        
        for t in range(1,T): #for calibration, run for 1 year (for policy simulation e.g., run for longer)
            other_deaths[:,t]+= BGD_ASM2[:,1]*num_alive[:,t].T
            total_alive[:,t]=(1-BGD_ASM2[:,1])*num_alive[:,t].T 
            cases[:,t]+=(p_cholera.T*total_alive[:,t])*poprisk 
            sev_cases[:,t]+=p_severe.T*cases[:,t]
            severe=p_severe.T*cases[:,t]
            mild=(1-p_severe.T)*cases[:,t]
            severe_treat=p_treat_sev*severe
            severe_untreat=(1-p_treat_sev)*severe
            mild_treat=p_treat_mild*mild
            mild_untreat=(1-p_treat_mild)*mild
            cases_hosp[:,t]+=severe_treat+mild_treat
            sev_cases_hosp[:,t]+=severe_treat
            cholera_deaths[:,t]+=p_die_severe_t*severe_treat+p_die_severe_nt*severe_untreat
            cholera_deaths_hosp[:,t]+=p_die_severe_t*severe_treat
            if t<T-1:
                num_alive[:,t+1]=np.roll(total_alive[:,t]-cholera_deaths[:,t],1, axis=0) #number remaining alive at beg. next period by current age group
        
        #saving output and params from each parameter set
        cases_all[:,i]=cases[:,1]+cases[:,2]
        sev_cases_all[:,i]=sev_cases[:,1]+sev_cases[:,2]
        cases_hosp_all[:,i]=cases_hosp[:,1]+cases_hosp[:,2]
        sev_cases_hosp_all[:,i]=sev_cases_hosp[:,1]+sev_cases_hosp[:,2]
        cholera_deaths_all[:,i]=cholera_deaths[:,1] #adding up deaths in period 1 only (because ages change)
        cholera_deaths_hosp_all[:,i]=cholera_deaths_hosp[:,1]
        total_deaths_all[:,i]=cholera_deaths[:,1]+other_deaths[:,1]
        total_alive_all[:,i]=(total_alive[:,1]+total_alive[:,2])/2

    
  
    #results = output on burden
    results=pd.DataFrame({'Cases_0':np.sum(cases_all[0:2,:],0)}, index=np.arange(0,n))
    results['Cases_1']=np.sum(cases_all[2:10,:],0)
    results['Cases_2']=np.sum(cases_all[10:30,:],0)
    results['Cases_3']=np.sum(cases_all[30:max_age-2,:],0)
    results['Cases']=results['Cases_0']+results['Cases_1']+results['Cases_2']+results['Cases_3']
    results['HospCases']=np.sum(cases_hosp_all,0)
    results['NonHospCases']=results['Cases']-results['HospCases']
    results['Sev_0']=np.sum(sev_cases_all[0:2,:],0)
    results['Sev_1']=np.sum(sev_cases_all[2:10,:],0)
    results['Sev_2']=np.sum(sev_cases_all[10:30,:],0)
    results['Sev_3']=np.sum(sev_cases_all[30:max_age-2,:],0)
    results['Sev']=results['Sev_0']+results['Sev_1']+results['Sev_2']+results['Sev_3']
    results['HospSev']=np.sum(sev_cases_hosp_all,0)
    results['NonHospSev']=np.sum(sev_cases_all,0)-results['HospSev']
    results['TotalAlive_0']=np.sum(total_alive_all[0:2,:],0)
    results['TotalAlive_1']=np.sum(total_alive_all[2:10,:],0)
    results['TotalAlive_2']=np.sum(total_alive_all[10:30,:],0)
    results['TotalAlive_3']=np.sum(total_alive_all[30:max_age-2,:],0)
    results['Incidence']=np.sum(cases_all,0)/np.sum(total_alive_all,0) #Annual
    results['Sev_Incidence']=np.sum(sev_cases_all,0)/np.sum(total_alive_all,0) #Annual
    results['CholeraDeaths_04']=np.sum(cholera_deaths_all[0:10,:],0)*2 #Annual
    results['CholeraDeaths_514']=np.sum(cholera_deaths_all[10:30,:],0)*2 
    results['CholeraDeaths_1549']=np.sum(cholera_deaths_all[30:100,:],0)*2 
    results['CholeraDeaths_5069']=np.sum(cholera_deaths_all[100:140,:],0)*2 
    results['CholeraDeaths_70+']=np.sum(cholera_deaths_all[140:max_age-2,:],0)*2 
    results['CholeraDeaths_All']=np.sum(cholera_deaths_all,0)*2 
    results['CholeraDeaths_Hosp']=np.sum(cholera_deaths_hosp_all,0)*2
    results['CholeraDeaths_NoHosp']=results['CholeraDeaths_All']-results['CholeraDeaths_Hosp']
    results['TotalDeaths_04']=np.sum(total_deaths_all[0:10,:],0)*2  #Annual
    results['TotalDeaths_514']=np.sum(total_deaths_all[10:30,:],0)*2 
    results['TotalDeaths_1549']=np.sum(total_deaths_all[30:100,:],0)*2 
    results['TotalDeaths_5069']=np.sum(total_deaths_all[100:140,:],0)*2 
    results['TotalDeaths_70+']=np.sum(total_deaths_all[140:max_age-2,:],0)*2 
    results['%DeathsCholera']=np.sum(cholera_deaths_all,0)/np.sum(total_deaths_all,0)
    results['CholeraDeaths_01']=np.sum(cholera_deaths_all[0:2,:],0)*2 #Annual
    results['CholeraDeaths_14']=np.sum(cholera_deaths_all[2:10,:],0)*2
    results['CholeraDeaths_15+']=np.sum(cholera_deaths_all[30:max_age-2,:],0)*2

    return(results)