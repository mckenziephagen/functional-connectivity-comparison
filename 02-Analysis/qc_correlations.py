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

import sys

#janky, but ok
sys.path.append("/global/u1/m/mphagen/functional_connectivity_comparison/fmriprep-denoise-benchmark/fmriprep_denoise")
from features.quality_control_connectivity import qcfc, partial_correlation


# +
import pandas as pd
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import fmriprep_denoise

from scipy import stats, linalg

# -

import scipy

rename_dict = {'RelativeRMS-rfMRI_REST1_RL' : 'relative_ses-1_run-1', 
             'RelativeRMS-rfMRI_REST1_LR' : 'relative_ses-1_run-2',
              'RelativeRMS-rfMRI_REST2_RL' : 'relative_ses-2_run-1', 
             'RelativeRMS-rfMRI_REST2_LR' : 'relative_ses-2_run-2', 
               
               'AbsoluteRMS-rfMRI_REST1_RL' : 'absolute_ses-1_run-1', 
             'AbsoluteRMS-rfMRI_REST1_LR' : 'absolute_ses-1_run-2',
              'AbsoluteRMS-rfMRI_REST2_RL' : 'absolute_ses-2_run-1', 
             'AbsoluteRMS-rfMRI_REST2_LR' : 'absolute_ses-2_run-2'
              }

# +
qa_df = pd.read_csv('../temp_qa_vals.csv', index_col=0).sort_index()
qa_df.index = [f'sub-{i}' for i in qa_df.index] 

qa_df = qa_df.rename(rename_dict, axis=1) 
# -

unrestricted_df = pd.read_csv('../Data/unrestricted_mphagen_1_27_2022_20_50_7.csv')
unrestricted_df.index = [f'sub-{i}' for i in unrestricted_df['Subject']]

restricted_df = pd.read_csv('../Data/RESTRICTED_arokem_1_31_2022_23_26_45.csv')
restricted_df.index = [f'sub-{i}' for i in restricted_df['Subject']]

# +
date_string='2024-04-29'
with open(f'../results/{date_string}_lasso-bic_dict.pkl', 'rb') as l:
        lasso_dict = pickle.load(l)
        
with open(f'../results/{date_string}_uoi_dict.pkl', 'rb') as u:
        uoi_dict = pickle.load(u)
        
with open(f'../results/{date_string}_pearson_dict.pkl', 'rb') as f:
        pearson_dict = pickle.load(f)


# -

def create_df(mat_dict, ses_string, rms='absolute'): 
    df = pd.DataFrame()
    for sub_id in pearson_dict.keys(): 
        df = df.join(pd.DataFrame({f'{sub_id}': mat_dict[sub_id][ses_string].flatten()} ),
                how='outer')
        
    df = df.T.join([qa_df[[f'{rms}_{ses_string}']], 
                     unrestricted_df[['Gender']].replace({'M':0, 'F':1})]).rename({f'{rms}_{ses_string}': 'mean_framewise_displacement'}, 
                                       axis='columns')

    return(df) 


# +
# for sub_id in pearson_dict.keys(): 
#    # for ses_id in lasso_dict[sub_id].keys():
#     ses_1_run_1_df = ses_1_run_1_df.join(pd.DataFrame({f'{sub_id}': pearson_dict[sub_id]['ses-1_run-1'].flatten()} ),
#                 how='outer')
    
# ses_1_run_1_df = ses_1_run_1_df.T

# ses_1_run_1_df = ses_1_run_1_df.join([qa_df[['absolute_ses_1_run_1']], 
#                      unrestricted_df[['Gender']].replace({'M':0, 'F':1})])

# +
def wrap_qcfc(df): 
    covarates = df[['Gender']]
    movement = df[['mean_framewise_displacement']]
    connectomes = df.drop(['mean_framewise_displacement', 'Gender'], axis=1)

    qcfc_results = qcfc(movement=movement, connectomes=connectomes, covarates=covarates)
    return(qcfc_results) 

def unpack_qcfc(results_obj): 
    pval_list = []
    sig_corr_list = [] 
    _ =[pval_list.append(i['pvalue']) for i in results_obj]
    _ =[ sig_corr_list.append(i['correlation']) for i in results_obj if i['pvalue'] <.05 ]
    return(pval_list, sig_corr_list) 


# -

lasso_df = create_df(lasso_dict, ses_string='ses-1_run-2')
lasso_results = wrap_qcfc(lasso_df) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)
