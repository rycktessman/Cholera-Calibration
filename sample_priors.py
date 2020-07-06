#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: samples from prior distributions used with calibration scripts           #
# Last Updated: June 8, 2020                                                        #
#####################################################################################

import numpy as np
import pandas as pd
from pyDOE import *
from sorted_rank import sorted_rank

#function inputs: n=number of samples to draw
def sample_priors(n):
    uniforms=lhs(24, samples=n) #creates LHS uniform[0,1] for each parameter (24 parameters in total)
    pCholera_0=uniforms[:,0]*0.07 #incidence ages 0-1
    pCholera_1=uniforms[:,1]*0.1 #incidence ages 1-4
    pCholera_2=uniforms[:,2]*0.025 #incidence ages 5-14
    pCholera_3=uniforms[:,3]*0.02 #incidence ages 15+
    
    pSevere_0=uniforms[:,4]*0.5 #severity ages 0-1
    pSevere_1=uniforms[:,5]*0.5 #severity ages 1-4
    pSevere_2=uniforms[:,6]*0.5 #severity ages 5-14
    pSevere_3=uniforms[:,7]*0.5 #severity ages 15+
    #induce correlation across severity samples by reordering using sorted_rank function
    pSevere=np.transpose(np.array([pSevere_0, pSevere_1, pSevere_2, pSevere_3]))
    rho=.5
    mean=np.zeros((4))
    cov=np.matrix([[1, rho, rho, rho],
                   [rho, 1, rho, rho],
                   [rho, rho, 1, rho],
                   [rho, rho, rho, 1]])
    eig_values=np.linalg.eig(cov)[0]
    eig_vecs=np.linalg.eig(cov)[1]
    eig_values[eig_values<=0]=0.0001
    cov_pd=eig_vecs * np.diag(eig_values) * np.linalg.inv(eig_vecs)
    #Cholesky decomposition if needed:
    #chol=np.linalg.cholesky(cov_pd)
    #norms=np.random.multivariate_normal(mean, np.eye(4,4),n)
    #norms=norms*chol
    norms=np.random.multivariate_normal(mean, cov_pd, n)
    pSevere_sorted=sorted_rank(pSevere, norms)
    pSevere_0=pSevere_sorted[:,0]
    pSevere_1=pSevere_sorted[:,1]
    pSevere_2=pSevere_sorted[:,2]
    pSevere_3=pSevere_sorted[:,3]
    
    pTreatSC_0=uniforms[:,8] #hospitalization for severe cases ages 0-1
    pTreatSC_1=uniforms[:,9] #hospitalization for severe cases ages 1-4
    pTreatSC_2=uniforms[:,10] #hospitalization for severe cases ages 5-14
    pTreatSC_3=uniforms[:,11] #hospitalization for severe cases ages 15+
    pTreatMC_0=uniforms[:,12]*pTreatSC_0 #hospitalization for mild cases ages 0-1
    pTreatMC_1=uniforms[:,13]*pTreatSC_1 #hospitalization for mild cases ages 1-4
    pTreatMC_2=uniforms[:,14]*pTreatSC_2 #hospitalization for mild cases ages 5-14
    pTreatMC_3=uniforms[:,15]*pTreatSC_3 #hospitalization for mild cases ages 15+
    pDieSevereT_0=uniforms[:,16]*0.1 #case fatality for treated (at hospital) severe cases ages 0-1
    pDieSevereT_1=uniforms[:,17]*0.1 #case fatality for treated (at hospital) severe cases ages 1-4
    pDieSevereT_2=uniforms[:,18]*0.1 #case fatality for treated (at hospital) severe cases ages 5-14
    pDieSevereT_3=uniforms[:,19]*0.1 #case fatality for treated (at hospital) severe cases ages 15+
    pDieSevereNT_0=(uniforms[:,20]*0.7)+.3 #case fatality for non-hospitalization severe cases ages 0-1
    pDieSevereNT_1=(uniforms[:,21]*0.7)+.3 #case fatality for non-hospitalization severe cases ages 1-4
    pDieSevereNT_2=(uniforms[:,22]*0.7)+.3 #case fatality for non-hospitalization severe cases ages 5-14
    pDieSevereNT_3=(uniforms[:,23]*0.7)+.3 #case fatality for non-hospitalization severe cases ages 15+
    
    params=pd.DataFrame({'pCholera_0': pCholera_0, 'pCholera_1': pCholera_1, 'pCholera_2': pCholera_2, 'pCholera_3': pCholera_3,
                               'pDieSevereNT_0': pDieSevereNT_0, 'pDieSevereNT_1': pDieSevereNT_1, 'pDieSevereNT_2': pDieSevereNT_2, 
                               'pDieSevereNT_3': pDieSevereNT_3, 'pDieSevereT_0': pDieSevereT_0, 'pDieSevereT_1': pDieSevereT_1, 
                               'pDieSevereT_2': pDieSevereT_2, 'pDieSevereT_3': pDieSevereT_3, 'pSevere_0': pSevere_0, 'pSevere_1': pSevere_1,
                               'pSevere_2': pSevere_2, 'pSevere_3': pSevere_3, 'pTreatMC_0': pTreatMC_0, 'pTreatMC_1': pTreatMC_1,
                               'pTreatMC_2': pTreatMC_2, 'pTreatMC_3': pTreatMC_3,'pTreatSC_0': pTreatSC_0, 'pTreatSC_1': pTreatSC_1,
                               'pTreatSC_2': pTreatSC_2, 'pTreatSC_3': pTreatSC_3,})
    return(params)