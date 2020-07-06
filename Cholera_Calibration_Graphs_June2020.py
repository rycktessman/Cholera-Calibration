##########################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                        #
# Purpose: generate graphs comparing combined output from multiple types of calibration  #
# Use this script after running calibration                                              #
# Last Updated: June 18, 2020                                                            #
##########################################################################################

#load packages 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

#load data from file
#Bangladesh background mortality and population age distribution
BGD_ASM=np.loadtxt(open("BGD_ASM.csv", "rb"), delimiter=",", skiprows=1)
BGD_Pop=np.loadtxt(open("BGD_Pop.csv", "rb"), delimiter=",", skiprows=0)
#calibration targets
targets=pd.read_csv('calibration_targets.csv', dtype={'Attribute':str})
targets.set_index('Attribute', inplace=True)
#posterior parameters and model output (targets) from random search, SIR, and IMIS
params_imis_post=pd.read_csv('params_post_imis.csv', index_col=0)
targets_imis_post=pd.read_csv('targets_post_imis.csv', index_col=0)
params_sir_post=pd.read_csv('params_post_sir.csv', index_col=0)
targets_sir_post=pd.read_csv('targets_post_sir.csv', index_col=0)
params_rs_post=pd.read_csv('params_post_rs.csv', index_col=0)
targets_rs_post=pd.read_csv('targets_post_rs.csv', index_col=0)
#parameters, model output (labeled as targets) and likelihoods of naive parameters
params_naive=pd.read_csv('naive_params.csv',index_col=0)
targets_naive=pd.read_csv('naive_targets.csv',index_col=0)
like_naive=np.loadtxt(open('naive_like.csv', "rb"), delimiter=",", skiprows=0)
#prior parameter distributions and corresponding model output
params_prior=pd.read_csv('params_prior.csv', index_col=0)
targets_prior=pd.read_csv('targets_prior.csv', index_col=0)
#burden estimates (from Cholera_Calibration_Burden_June2020.py)
burden_imis=pd.read_csv('cholera_burden_imis.csv', index_col=0)
burden_sir=pd.read_csv('cholera_burden_sir.csv', index_col=0)
burden_rs=pd.read_csv('cholera_burden_rs.csv', index_col=0)
burden_naive=pd.read_csv('cholera_burden_naive.csv', index_col=0)
burden_prior=pd.read_csv('cholera_burden_prior.csv', index_col=0)

#add source and like to make format of naive params and targets consistent with others
params_naive.insert(0, "Source", "Naive")
params_naive.insert(1, "Like", like_naive)
targets_naive.insert(0, "Source", "Naive")
targets_naive.insert(1, "Like", like_naive)
del like_naive

#specify parameters and labels
cyc_len=0.5 
max_age=int(116/cyc_len) #running for all ages for now
param_list=list(['Incidence <1', 'Incidence 1-4', 'Incidence 5-14', 'Incidence 15+',
                      'Case Fatality w/out Tx <1', 'Case Fatality w/out Tx 1-4',
                      'Case Fatality w/out Tx 5-14', 'Case Fatality w/out Tx 15+',
                      'Case Fatality w/ Tx <1', 'Case Fatality w/ Tx 1-4',
                      'Case Fatality w/ Tx 5-14', 'Case Fatality w/ Tx 15+', 
                      'Severity <1', 'Severity 1-4', 'Severity 5-14', 'Severity 15+',
                      'Severe Hospitalization <1', 'Severe Hospitalization 1-4',
                      'Severe Hospitalization 5-14', 'Severe Hospitalization 15+',
                      'Mild Hospitalization <1','Mild Hospitalization 1-4', 
                      'Mild Hospitalization 5-14', 'Mild Hospitalization 15+'])
param_list_long=list(['Incidence ages 0-1', 'Incidence ages 1-4', 'Incidence ages 5-14', 'Incidence ages 15+',
                      'Fatality w/out Tx ages 0-1', 'Fatality w/out Tx ages 1-4',
                      'Fatality w/out Tx ages 5-14', 'Fatality w/out Tx ages 15+',
                      'Fatality w/ Tx ages 0-1', 'Fatality w/ Tx ages 1-4',
                      'Fatality w/ Tx ages 5-14', 'Fatality w/ Tx ages 15+',
                      'Severity ages 0-1', 'Severity ages 1-4', 'Severity ages 5-14', 'Severity ages 15+',
                      'Mild Tx ages 0-1', 'Mild Tx ages 1-4', 'Mild Tx ages 5-14', 'Mild Tx ages 15+',
                      'Severe Tx ages 0-1', 'Severe Tx ages 1-4', 'Severe Tx ages 5-14', 'Severe Tx ages 15+'])
target_list=list(['Cholera Deaths 0-4 (%)', 'Cholera Deaths 5-14 (%)', 
                      'Cholera Deaths 15-49 (%)', 'Cholera Deaths 50-69 (%)', 
                      'Cholera Deaths 70+ (%)', 'Observed Incidence <1', 
                      'Observed Incidence 1-4', 'Observed Incidence 5-14', 
                      'Observed Incidence 15+', 'Observed Sev. Incidence 1-4',
                      'Observed Sev. Incidence 5-14', 'Observed Sev. Incidence 15+'])

#broadcasting target outputs by each individual age
#SIR posteriors
deaths_sir_post=np.zeros((max_age,np.shape(params_sir_post)[0]))
deaths_sir_post[0:10,:]=targets_sir_post['%Deaths_04']
deaths_sir_post[10:30,:]=targets_sir_post['%Deaths_514']
deaths_sir_post[30:100,:]=targets_sir_post['%Deaths_1549']
deaths_sir_post[100:140,:]=targets_sir_post['%Deaths_5069']
deaths_sir_post[140:max_age-2,:]=targets_sir_post['%Deaths_70+']
inc_sir_post=np.zeros((max_age,np.shape(params_sir_post)[0]))
inc_sir_post[0:2,:]=targets_sir_post['HospInc_<1']
inc_sir_post[2:10,:]=targets_sir_post['HospInc_14']
inc_sir_post[10:30,:]=targets_sir_post['HospInc_514']
inc_sir_post[30:max_age-2,:]=targets_sir_post['HospInc_15+']
sev_inc_sir_post=np.zeros((max_age,np.shape(params_sir_post)[0]))
sev_inc_sir_post[2:10,:]=targets_sir_post['SevHospInc_14']
sev_inc_sir_post[10:30,:]=targets_sir_post['SevHospInc_514']
sev_inc_sir_post[30:max_age-2,:]=targets_sir_post['SevHospInc_15+']
#IMIS posteriors
deaths_imis_post=np.zeros((max_age,np.shape(params_imis_post)[0]))
deaths_imis_post[0:10,:]=targets_imis_post['%Deaths_04']
deaths_imis_post[10:30,:]=targets_imis_post['%Deaths_514']
deaths_imis_post[30:100,:]=targets_imis_post['%Deaths_1549']
deaths_imis_post[100:140,:]=targets_imis_post['%Deaths_5069']
deaths_imis_post[140:max_age-2,:]=targets_imis_post['%Deaths_70+']
inc_imis_post=np.zeros((max_age,np.shape(params_imis_post)[0]))
inc_imis_post[0:2,:]=targets_imis_post['HospInc_<1']
inc_imis_post[2:10,:]=targets_imis_post['HospInc_14']
inc_imis_post[10:30,:]=targets_imis_post['HospInc_514']
inc_imis_post[30:max_age-2,:]=targets_imis_post['HospInc_15+']
sev_inc_imis_post=np.zeros((max_age,np.shape(params_imis_post)[0]))
sev_inc_imis_post[2:10,:]=targets_imis_post['SevHospInc_14']
sev_inc_imis_post[10:30,:]=targets_imis_post['SevHospInc_514']
sev_inc_imis_post[30:max_age-2,:]=targets_imis_post['SevHospInc_15+']
#RS posteriors
deaths_rs_post=np.zeros((max_age,np.shape(params_rs_post)[0]))
deaths_rs_post[0:10,:]=targets_rs_post['%Deaths_04']
deaths_rs_post[10:30,:]=targets_rs_post['%Deaths_514']
deaths_rs_post[30:100,:]=targets_rs_post['%Deaths_1549']
deaths_rs_post[100:140,:]=targets_rs_post['%Deaths_5069']
deaths_rs_post[140:max_age-2,:]=targets_rs_post['%Deaths_70+']
inc_rs_post=np.zeros((max_age,np.shape(params_rs_post)[0]))
inc_rs_post[0:2,:]=targets_rs_post['HospInc_<1']
inc_rs_post[2:10,:]=targets_rs_post['HospInc_14']
inc_rs_post[10:30,:]=targets_rs_post['HospInc_514']
inc_rs_post[30:max_age-2,:]=targets_rs_post['HospInc_15+']
sev_inc_rs_post=np.zeros((max_age,np.shape(params_rs_post)[0]))
sev_inc_rs_post[2:10,:]=targets_rs_post['SevHospInc_14']
sev_inc_rs_post[10:30,:]=targets_rs_post['SevHospInc_514']
sev_inc_rs_post[30:max_age-2,:]=targets_rs_post['SevHospInc_15+']
#Naive params
deaths_naive=np.zeros((max_age,np.shape(params_naive)[0]))
deaths_naive[0:10,:]=targets_naive['%Deaths_04']
deaths_naive[10:30,:]=targets_naive['%Deaths_514']
deaths_naive[30:100,:]=targets_naive['%Deaths_1549']
deaths_naive[100:140,:]=targets_naive['%Deaths_5069']
deaths_naive[140:max_age-2,:]=targets_naive['%Deaths_70+']
inc_naive=np.zeros((max_age,np.shape(params_naive)[0]))
inc_naive[0:2,:]=targets_naive['HospInc_<1']
inc_naive[2:10,:]=targets_naive['HospInc_14']
inc_naive[10:30,:]=targets_naive['HospInc_514']
inc_naive[30:max_age-2,:]=targets_naive['HospInc_15+']
sev_inc_naive=np.zeros((max_age,np.shape(params_naive)[0]))
sev_inc_naive[2:10,:]=targets_naive['SevHospInc_14']
sev_inc_naive[10:30,:]=targets_naive['SevHospInc_514']
sev_inc_naive[30:max_age-2,:]=targets_naive['SevHospInc_15+']
#Priors
deaths_prior=np.zeros((max_age,np.shape(params_prior)[0]))
deaths_prior[0:10,:]=targets_prior['%Deaths_04']
deaths_prior[10:30,:]=targets_prior['%Deaths_514']
deaths_prior[30:100,:]=targets_prior['%Deaths_1549']
deaths_prior[100:140,:]=targets_prior['%Deaths_5069']
deaths_prior[140:max_age-2,:]=targets_prior['%Deaths_70+']
inc_prior=np.zeros((max_age,np.shape(params_prior)[0]))
inc_prior[0:2,:]=targets_prior['HospInc_<1']
inc_prior[2:10,:]=targets_prior['HospInc_14']
inc_prior[10:30,:]=targets_prior['HospInc_514']
inc_prior[30:max_age-2,:]=targets_prior['HospInc_15+']
sev_inc_prior=np.zeros((max_age,np.shape(params_prior)[0]))
sev_inc_prior[2:10,:]=targets_prior['SevHospInc_14']
sev_inc_prior[10:30,:]=targets_prior['SevHospInc_514']
sev_inc_prior[30:max_age-2,:]=targets_prior['SevHospInc_15+']
#targets themselves (from data)
deaths_target=np.zeros((max_age,3))
deaths_target[0:10,0]=targets['%Deaths_04'].loc['mean']
deaths_target[10:30,0]=targets['%Deaths_514'].loc['mean']
deaths_target[30:100,0]=targets['%Deaths_1549'].loc['mean']
deaths_target[100:140,0]=targets['%Deaths_5069'].loc['mean']
deaths_target[140:max_age-2,0]=targets['%Deaths_70+'].loc['mean']
deaths_target[0:10,1]=targets['%Deaths_04'].loc['low (99)']
deaths_target[10:30,1]=targets['%Deaths_514'].loc['low (99)']
deaths_target[30:100,1]=targets['%Deaths_1549'].loc['low (99)']
deaths_target[100:140,1]=targets['%Deaths_5069'].loc['low (99)']
deaths_target[140:max_age-2,1]=targets['%Deaths_70+'].loc['low (99)']
deaths_target[0:10,2]=targets['%Deaths_04'].loc['high (99)']
deaths_target[10:30,2]=targets['%Deaths_514'].loc['high (99)']
deaths_target[30:100,2]=targets['%Deaths_1549'].loc['high (99)']
deaths_target[100:140,2]=targets['%Deaths_5069'].loc['high (99)']
deaths_target[140:max_age-2,2]=targets['%Deaths_70+'].loc['high (99)']
inc_target=np.zeros((max_age,3))
inc_target[0:2,0]=targets['HospInc_<1'].loc['mean']
inc_target[2:10,0]=targets['HospInc_14'].loc['mean']
inc_target[10:30,0]=targets['HospInc_514'].loc['mean']
inc_target[30:max_age-2,0]=targets['HospInc_15+'].loc['mean']
inc_target[0:2,1]=targets['HospInc_<1'].loc['low (99)']
inc_target[2:10,1]=targets['HospInc_14'].loc['low (99)']
inc_target[10:30,1]=targets['HospInc_514'].loc['low (99)']
inc_target[30:max_age-2,1]=targets['HospInc_15+'].loc['low (99)']
inc_target[0:2,2]=targets['HospInc_<1'].loc['high (99)']
inc_target[2:10,2]=targets['HospInc_14'].loc['high (99)']
inc_target[10:30,2]=targets['HospInc_514'].loc['high (99)']
inc_target[30:max_age-2,2]=targets['HospInc_15+'].loc['high (99)']
sev_inc_target=np.zeros((max_age,3))
sev_inc_target[2:10,0]=targets['SevHospInc_14'].loc['mean']
sev_inc_target[10:30,0]=targets['SevHospInc_514'].loc['mean']
sev_inc_target[30:max_age-2,0]=targets['SevHospInc_15+'].loc['mean']
sev_inc_target[2:10,1]=targets['SevHospInc_14'].loc['low (99)']
sev_inc_target[10:30,1]=targets['SevHospInc_514'].loc['low (99)']
sev_inc_target[30:max_age-2,1]=targets['SevHospInc_15+'].loc['low (99)']
sev_inc_target[2:10,2]=targets['SevHospInc_14'].loc['high (99)']
sev_inc_target[10:30,2]=targets['SevHospInc_514'].loc['high (99)']
sev_inc_target[30:max_age-2,2]=targets['SevHospInc_15+'].loc['high (99)']

#calculate means and 95% CIs of parameters and burden output, and save to file
def calc_stats(data, title, col_start):
    means=data.iloc[:,col_start:].mean()
    lbs=data.iloc[:,col_start:].quantile(.025)
    ubs=data.iloc[:,col_start:].quantile(.975)
    CIs=pd.DataFrame({'Mean': means, 'LB': lbs, 'UB': ubs})
    CIs.to_csv('stats_' + title + '.csv')
calc_stats(params_imis_post, "params_imis", 2)
calc_stats(params_sir_post, "params_sir", 2)
calc_stats(params_rs_post, "params_rs", 2)
calc_stats(params_naive, "params_naive", 2)
calc_stats(params_prior, "params_prior", 2)
calc_stats(burden_imis, "burden_imis", 0)
calc_stats(burden_sir, "burden_sir", 0)
calc_stats(burden_rs, "burden_rs", 0)
calc_stats(burden_naive, "burden_naive", 0)
calc_stats(burden_prior, "burden_prior", 0)

#graphs
color_prior='black'
color_imis='purple'
color_sir='orangered'
color_rs='cadetblue'
color_naive='gray'
line_width=3
line_width_imis=4
style_prior='dashed'
style_imis='solid'
style_sir='solid'
style_rs='solid'
style_naive='dotted'
text_size=36
ticks_size=30
legend_size=30
bw_rs=dict(zip((list(params_prior)[2:]),[2,2,1,1.5,7,7,7,7,1,1,1,1,5,5,5,5,7,7,7,7,5,5,5,5]))
bw_imis=dict(zip((list(params_prior)[2:]),[2,2,1,1.5,7,7,7,7,1,1,1,1,5,5,5,5,7,7,7,7,5,5,5,5]))
bw_sir=dict(zip((list(params_prior)[2:]),[2,2,1,1.5,7,7,7,7,1,1,1,1,5,5,5,5,7,7,7,7,5,5,5,5]))
bw_prior=dict(zip((list(params_prior)[2:]),[2,2,1,1.5,7,7,7,7,1,1,1,1,5,5,5,5,7,7,7,7,5,5,5,5]))
bw_naive=dict(zip((list(params_prior)[2:]),[2,2,1,1.5,7,7,7,7,1,1,1,1,5,5,5,5,7,7,7,7,5,5,5,5]))
mults=dict(zip((list(params_prior)[2:]),[1000,1000,1000,1000,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]))
x_hi=dict(zip((list(params_prior)[2:]),[70,100,25,25,100,100,100,100,10,10,10,10,50,50,50,50,100,100,100,100,100,100,100,100]))
x_lo=dict(zip((list(params_prior)[2:]),[0,0,0,0,30,30,30,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
y_hi=dict(zip((list(params_prior)[2:]),[.35,.35,.35,.35,.08,.08,.08,.08,.35,.35,.35,.35,.08,.08,.08,.08,.08,.08,.08,.08,.08,.08,.08,.08]))
xlabels=dict(zip((list(params_prior)[2:]), ['Cases per 1000','Cases per 1000','Cases per 1000','Cases per 1000', 'Case Fatality (%)', 'Case Fatality (%)','Case Fatality (%)','Case Fatality (%)',
                 'Case Fatality (%)', 'Case Fatality (%)', 'Case Fatality (%)', 'Case Fatality (%)', '% Cases that are Severe','% Cases that are Severe','% Cases that are Severe','% Cases that are Severe',
                 'Mild Case Hosp. (%)','Mild Case Hosp. (%)','Mild Case Hosp. (%)','Mild Case Hosp. (%)',
                 'Severe Case Hosp. (%)','Severe Case Hosp. (%)','Severe Case Hosp. (%)','Severe Case Hosp. (%)']))
titles=dict(zip((list(params_prior)[2:]), param_list_long))

#Parameter distribution densities
j=0
for i in list(params_prior)[2:]:
    print(j)
    fig=plt.figure(figsize=(8,7))
    plt.xlim((x_lo[i],x_hi[i]))
    plt.ylim((0,y_hi[i]))
    sns.kdeplot(params_prior[i]*mults[i], bw=bw_prior[i], color=color_prior, linestyle=style_prior, lw=line_width)
    sns.kdeplot(params_imis_post[i]*mults[i], bw=bw_imis[i], color=color_imis, linestyle=style_imis, lw=line_width_imis)
    sns.kdeplot(params_sir_post[i]*mults[i], bw=bw_sir[i], color=color_sir, linestyle=style_sir, lw=line_width)
    sns.kdeplot(params_rs_post[i]*mults[i], bw=bw_rs[i], color=color_rs, linestyle=style_rs, lw=line_width)
    sns.kdeplot(params_naive[i]*mults[i], bw=bw_naive[i], color=color_naive, linestyle=style_naive, lw=line_width)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    custom_lines = [Line2D([0], [0], color=color_prior, lw=line_width, linestyle=style_prior),
                    Line2D([0], [0], color=color_imis, lw=line_width, linestyle=style_imis),
                    Line2D([0], [0], color=color_sir, lw=line_width, linestyle=style_sir),
                    Line2D([0], [0], color=color_rs, lw=line_width, linestyle=style_rs),
                    Line2D([0], [0], color=color_naive, lw=line_width, linestyle=style_naive)]
    if j==1:
        plt.legend(custom_lines, ['Priors', 'IMIS Posteriors', 'SIR Posteriors', 
                                  'RS Posteriors', 'Naive Parameters'], fontsize=legend_size)
    else:
        plt.legend().set_visible(False)
    plt.title(titles[i], fontsize=text_size)
    plt.xlabel(xlabels[i], fontsize=text_size)
    plt.ylabel('Density', fontsize=text_size)
    fig.savefig(i+'Density.png', dpi=500, bbox_inches='tight')
    j+=1
    

#Burden distribution densities
j=0
burden_subset=list(burden_imis)[0:14]+list(burden_imis)[34:36]+[list(burden_imis)[21]] + [list(burden_imis)[36]]+list(burden_imis)[25:28]
bw_imis=dict(zip((burden_subset),[6,15,15,50,.05,.005,.1,1,4,1,2,5,5,1,.25,.25,.25,.25,.25,.25,.25]))
bw_sir=dict(zip((burden_subset), [6,15,15,50,.05,.005,.1,1,4,1,2,5,5,1,.25,.25,.25,.25,.25,.25,.25]))
bw_rs=dict(zip((burden_subset), [6,15,15,50,.05,.005,.1,1,4,1,2,5,5,1,.25,.25,.25,.25,.25,.25,.25]))
bw_prior=dict(zip((burden_subset), [6,15,15,50,.05,.005,.1,1,4,1,2,5,5,1,.25,.25,.25,.25,.25,.25,.25]))
bw_naive=dict(zip((burden_subset), [6,15,15,50,.05,.005,.1,1,4,1,2,5,5,1,.25,.25,.25,.25,.25,.25,.25]))
x_hi=dict(zip((burden_subset),[120,600,400,1100,1.8,0.3,1.8,15,60,60,120,200,120,60,10,15,10,15,30,25,25]))
axis_size=32
ticks_size=28
titles=dict(zip((burden_subset), ['Cases Ages <1', 'Cases Ages 1-4', 'Cases Ages 5-14', 'Cases Ages 15+' ,
                'Total Cases', 'Hospitalized Cases', 'Non-Hosp. Cases',
                'Severe Cases Ages <1', 'Severe Cases Ages 1-4', 'Severe Cases Ages 5-14', 'Severe Cases Ages 15+',
                'Total Severe Cases', 'Hospitalized Sev. Cases', 'Non-Hosp. Sev. Cases',
                'Cholera Deaths Ages <1', 'Cholera Deaths Ages 1-4', 'Cholera Deaths Ages 5-14', 'Cholera Deaths Ages 15+',
                'Total Cholera Deaths', 'Hospitalized Cholera Deaths', 'Non-Hosp. Cholera Deaths']))
xlabels=dict(zip((burden_subset), ['Cases (annual, thousands)','Cases (annual, thousands)','Cases (annual, thousands)',
                 'Cases (annual, thousands)','Cases (annual, millions)','Cases (annual, millions)','Cases (annual, millions)',
                 'Severe Cases (annual, thousands)','Severe Cases (annual, thousands)','Severe Cases (annual, thousands)',
                 'Severe Cases (annual, thousands)','Severe Cases (annual, thousands)','Severe Cases (annual, thousands)','Severe Cases (annual, thousands)',
                 'Deaths (annual, thousands)','Deaths (annual, thousands)','Deaths (annual, thousands)',
                 'Deaths (annual, thousands)','Deaths (annual, thousands)','Deaths (annual, thousands)','Deaths (annual, thousands)']))
mults=dict(zip((burden_subset), [1/1000,1/1000,1/1000,1/1000,1/1000000,1/1000000,1/1000000,
               1/1000,1/1000,1/1000,1/1000,1/1000,1/1000,1/1000,
               1/1000,1/1000,1/1000,1/1000,1/1000,1/1000,1/1000,]))
for i in burden_subset:
    fig=plt.figure(figsize=(8,7))
    plt.xlim((0,x_hi[i]))
    sns.kdeplot(burden_prior[i]*mults[i], color=color_prior, lw=line_width, linestyle=style_prior, bw=bw_prior[i])
    sns.kdeplot(burden_naive[i]*mults[i], color=color_naive, lw=line_width, linestyle=style_naive, bw=bw_naive[i])
    sns.kdeplot(burden_sir[i]*mults[i], color=color_sir, lw=line_width, linestyle=style_sir, bw=bw_sir[i])
    sns.kdeplot(burden_rs[i]*mults[i], color=color_rs, lw=line_width, linestyle=style_rs, bw=bw_rs[i])
    sns.kdeplot(burden_imis[i]*mults[i], color=color_imis, lw=line_width, linestyle=style_imis, bw=bw_imis[i])
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    custom_lines = [Line2D([0], [0], color=color_prior, lw=line_width, linestyle=style_prior),
                    Line2D([0], [0], color=color_imis, lw=line_width, linestyle=style_imis),
                    Line2D([0], [0], color=color_sir, lw=line_width, linestyle=style_sir),
                    Line2D([0], [0], color=color_rs, lw=line_width, linestyle=style_rs),
                    Line2D([0], [0], color=color_naive, lw=line_width, linestyle=style_naive)]
    if j==0:
         plt.legend(custom_lines, ['Priors', 'IMIS Posteriors', 'SIR Posteriors', 
                                  'RS Posteriors', 'Naive Parameters'], fontsize=legend_size)
    else:
        plt.legend().set_visible(False)
    plt.title(titles[i], fontsize=text_size)
    plt.xlabel(xlabels[i], fontsize=axis_size)
    plt.ylabel('Density', fontsize=axis_size)
    fig.savefig(i+'Burden_All.png', dpi=500, bbox_inches='tight')
    j+=1

#Model output vs. targets
ticks_size=24
title_size=34
line_width=.002

def target_plot(data, targets, mult, color, title, ylabel, filename):
    fig=plt.figure(figsize=(8,6.5))
    plt.plot(BGD_Pop[:,0], (mult*np.array(data)), color=color, linewidth=line_width)
    plt.plot(BGD_Pop[:,0], mult*targets, color='0.1')
    plt.title(title, fontsize=text_size)
    plt.xlabel("Age", fontsize=ticks_size)
    plt.ylabel(ylabel, fontsize=ticks_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    fig.savefig('targets_'+filename, dpi=500, bbox_inches='tight')

#deaths target graphs
target_plot(deaths_imis_post, deaths_target, 100, color_imis, "% Deaths from Cholera", 
            "% deaths caused by cholera", 'deaths_imis')
target_plot(deaths_sir_post, deaths_target, 100, color_sir, "% Deaths from Cholera", 
            "% deaths caused by cholera", 'deaths_sir')
target_plot(deaths_rs_post, deaths_target, 100, color_rs, "% Deaths from Cholera", 
            "% deaths caused by cholera", 'deaths_rs')
target_plot(deaths_naive, deaths_target, 100, color_naive, "% Deaths from Cholera", 
            "% deaths caused by cholera", 'deaths_naive')
target_plot(deaths_prior, deaths_target, 100, color_prior, "% Deaths from Cholera", 
            "% deaths caused by cholera", 'deaths_prior')
#all incidence target graphs
target_plot(inc_imis_post, inc_target, 1000, color_imis, "Observed Cholera Incidence", 
            "hosp. cases per 1000", 'inc_imis')
target_plot(inc_sir_post, inc_target, 1000, color_sir, "Observed Cholera Incidence", 
            "hosp. cases per 1000", 'inc_sir')
target_plot(inc_rs_post, inc_target, 1000, color_rs, "Observed Cholera Incidence", 
            "hosp. cases per 1000", 'inc_rs')
target_plot(inc_naive, inc_target, 1000, color_naive, "Observed Cholera Incidence", 
            "hosp. cases per 1000", 'inc_naive')
target_plot(inc_prior, inc_target, 1000, color_prior, "Observed Cholera Incidence", 
            "hosp. cases per 1000", 'inc_prior')
#severe incidence target graphs
target_plot(sev_inc_imis_post, sev_inc_target, 1000, color_imis, "Observed Severe Cholera Inc.", 
            "hosp. cases per 1000", 'sev_inc_imis')
target_plot(sev_inc_sir_post, sev_inc_target, 1000, color_sir, "Observed Severe Cholera Inc.", 
            "hosp. cases per 1000", 'sev_inc_sir')
target_plot(sev_inc_rs_post, sev_inc_target, 1000, color_rs, "Observed Severe Cholera Inc.", 
            "hosp. cases per 1000", 'sev_inc_rs')
target_plot(sev_inc_naive, sev_inc_target, 1000, color_naive, "Observed Severe Cholera Inc.", 
            "hosp. cases per 1000", 'sev_inc_naive')
target_plot(sev_inc_prior, sev_inc_target, 1000, color_prior, "Observed Severe Cholera Inc.", 
            "hosp. cases per 1000", 'sev_inc_prior')


#HEATMAPS OF CORRELATIONS
table=pd.DataFrame()
table['Mean']=params_imis_post.mean(0)
table['Median']=params_imis_post.median(0)
table['25']=params_imis_post.quantile(q=0.25, axis=0)
table['75']=params_imis_post.quantile(q=0.75, axis=0)
table=table.iloc[1:] #remove likelihood row

def upper_tri_mask(A):
    m=A.shape[0]
    r=np.arange(m)
    mask=r[:,None]<r
    return np.multiply(A,mask)

text_size=22

corr=params_imis_post.corr().iloc[2:26,2:26] #remove likelihood, index, and chain columns/rows
sns.set(style="white")
corr2=upper_tri_mask(np.matrix(corr))
#all age groups
fig=plt.figure(figsize=((10,7.5)))
heatmap=sns.heatmap(corr2, cmap='bwr', center=0, square=True, xticklabels=param_list_long, yticklabels=param_list_long)
plt.title("IMIS Correlations", fontsize=text_size)
fig.savefig('heatmap_imis.png', dpi=500, bbox_inches='tight')

#SIR HEATMAPS
sns.set(style="white")
corr=params_sir_post.corr().iloc[4:,4:] #remove likelihood, index, and chain columns/rows
corr2=upper_tri_mask(np.matrix(corr))
#all age groups
fig=plt.figure(figsize=((10,7.5)))
heatmap=sns.heatmap(corr2, cmap='bwr', center=0, square=True, xticklabels=param_list_long, yticklabels=param_list_long)
plt.title("SIR Correlations", fontsize=text_size)
fig.savefig('heatmap_sir.png', dpi=500, bbox_inches='tight')

#RANDOM SEARCH HEATMAPS
corr=params_rs_post.corr() #no columns/rows to remove
corr2=upper_tri_mask(np.matrix(corr))
#all age groups
fig=plt.figure(figsize=((10,7.5)))
heatmap=sns.heatmap(corr2, cmap='bwr', center=0, square=True, xticklabels=param_list_long, yticklabels=param_list_long)
plt.title("RS Correlations", fontsize=text_size)
fig.savefig('heatmap_rs.png', dpi=500, bbox_inches='tight')


