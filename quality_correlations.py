# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: FC
#     language: python
#     name: fc
# ---

import pandas as pd
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

rename_dict = {'RelativeRMS-rfMRI_REST1_RL' : 'relative_ses_1_run_1', 
             'RelativeRMS-rfMRI_REST1_LR' : 'relative_ses_1_run_2',
              'RelativeRMS-rfMRI_REST2_RL' : 'relative_ses_2_run_1', 
             'RelativeRMS-rfMRI_REST2_LR' : 'relative_ses_2_run_2', 
               
               'AbsoluteRMS-rfMRI_REST1_RL' : 'absolute_ses_1_run_1', 
             'AbsoluteRMS-rfMRI_REST1_LR' : 'absolute_ses_1_run_2',
              'AbsoluteRMS-rfMRI_REST2_RL' : 'absolute_ses_2_run_1', 
             'AbsoluteRMS-rfMRI_REST2_LR' : 'absolute_ses_2_run_2'
              }

# +
qa_df = pd.read_csv('temp_qa_vals.csv', index_col=0).sort_index()
qa_df.index = [f'sub-{i}' for i in qa_df.index] 

qa_df = qa_df.rename(rename_dict, axis=1) 
# -

unrestricted_df = pd.read_csv('unrestricted_mphagen_1_27_2022_20_50_7.csv')
unrestricted_df.index = [f'sub-{i}' for i in unrestricted_df['Subject']]

restricted_df = pd.read_csv('RESTRICTED_arokem_1_31_2022_23_26_45.csv')
restricted_df.index = [f'sub-{i}' for i in restricted_df['Subject']]

# +
date_string='2023-11-07'
with open(f'results/{date_string}_lasso_dict.pkl', 'rb') as l:
        lasso_dict = pickle.load(l)
        
with open(f'results/{date_string}_uoi_dict.pkl', 'rb') as u:
        uoi_dict = pickle.load(u)
        
with open(f'results/{date_string}_pearson_dict.pkl', 'rb') as f:
        pearson_dict = pickle.load(f)
# -

rms_type = 'relative' 

fc_type = 'lasso' 

res_dict = {} 
for scan in ['ses-1_run-1', 'ses-1_run-2', 'ses-2_run-1', 'ses-2_run-2']: 
    temp_df = pd.DataFrame()
    if fc_type == 'pearson': 
        fc_dict = pearson_dict
    if fc_type == 'uoi':
        fc_dict = uoi_dict
    if fc_type == 'lasso': 
        fc_dict = lasso_dict
    for key in fc_dict.keys(): 
        try: 
            temp_df[key] = fc_dict[key][scan].ravel()
        except KeyError: 
            print(f'{scan} not in {key}') 
    adj_df = temp_df.T.sort_index().add_prefix('node_')
    
    
    scan = scan.replace('-', '_') #statsmodels doesn't like - 
    res_dict[scan] = {} 

    adj_df = pd.concat([adj_df,
                     qa_df[f'{rms_type}_{scan}'], 
                     restricted_df['Age_in_Yrs'], 
                     unrestricted_df['Gender']], 
                    axis=1).dropna()
    
    adj_df.columns = adj_df.columns.astype(str)
    adj_df['Gender'] = pd.get_dummies(adj_df['Gender'], dtype=float)['F']
    
    
    for idx in range(1, 9999):
        res = smf.ols(f'{rms_type}_{scan} ~ node_{idx} + Age_in_Yrs + Gender', data=adj_df).fit()
        res_dict[scan][f'node_{idx}'] = {'pval' : (res.f_pvalue), 'corr' : np.sqrt(res.rsquared )} 

res_dict

sig_corr = {}
for key in res_dict[scan].keys(): 
    if res_dict[scan][key]['pval'] < .05: 
        sig_corr[scan].append(res_dict[scan][key]['corr']) 
