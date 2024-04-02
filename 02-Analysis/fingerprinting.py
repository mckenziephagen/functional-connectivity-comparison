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

import pickle

import numpy as np
import matplotlib.pyplot as plt

# +
date_string='2023-11-07'
with open(f'results/{date_string}_lasso_dict.pkl', 'rb') as l:
        lasso_dict = pickle.load(l)
        
with open(f'results/{date_string}_uoi_dict.pkl', 'rb') as u:
        uoi_dict = pickle.load(u)
        
with open(f'results/{date_string}_pearson_dict.pkl', 'rb') as f:
        pearson_dict = pickle.load(f)
# -

pearson_ses_dict = {} 
for sub_id in pearson_dict.keys(): 
    if sub_id not in pearson_ses_dict.keys(): 
        pearson_ses_dict[sub_id] = {} 
    try: 
        pearson_ses_dict[sub_id]['ses-1'] = np.median(np.array([pearson_dict[sub_id]['ses-1_run-1'].ravel(),
                                                    pearson_dict[sub_id]['ses-1_run-2'].ravel()]), axis=0)
    except KeyError: 
        pass
    
    try:
        pearson_ses_dict[sub_id]['ses-2'] = np.median(np.array([pearson_dict[sub_id]['ses-2_run-1'].ravel(),
                                                    pearson_dict[sub_id]['ses-2_run-2'].ravel()]), axis=0)
    except KeyError:
        pass
    

lasso_ses_dict = {}
for sub_id in lasso_dict.keys(): 
    if sub_id not in lasso_ses_dict.keys(): 
        lasso_ses_dict[sub_id] = {} 
    try: 
        lasso_ses_dict[sub_id]['ses-1'] = np.median(np.array([lasso_dict[sub_id]['ses-1_run-1'].ravel(),
                                                    lasso_dict[sub_id]['ses-1_run-2'].ravel()]), axis=0)
    except KeyError: 
        pass
    
    try:
        lasso_ses_dict[sub_id]['ses-2'] = np.median(np.array([lasso_dict[sub_id]['ses-2_run-1'].ravel(),
                                                    lasso_dict[sub_id]['ses-2_run-2'].ravel()]), axis=0)
    except KeyError:
        pass
    

uoi_ses_dict = {}
for sub_id in uoi_dict.keys(): 
    if sub_id not in uoi_ses_dict.keys(): 
        uoi_ses_dict[sub_id] = {} 
    try: 
        uoi_ses_dict[sub_id]['ses-1'] = np.median(np.array([uoi_dict[sub_id]['ses-1_run-1'].ravel(),
                                                    uoi_dict[sub_id]['ses-1_run-2'].ravel()]), axis=0)
    except KeyError: 
        pass
    
    try:
        uoi_ses_dict[sub_id]['ses-2'] = np.median(np.array([uoi_dict[sub_id]['ses-2_run-1'].ravel(),
                                                    uoi_dict[sub_id]['ses-2_run-2'].ravel()]), axis=0)
    except KeyError:
        pass
    

pearson_corr_dict = {} 
for sub_id in pearson_ses_dict.keys(): 
    if sub_id not in pearson_corr_dict.keys(): 
        pearson_corr_dict[sub_id] = {}

    for sub_id2 in pearson_ses_dict.keys(): 
        try: 
            pearson_corr_dict[sub_id][sub_id2] = (np.corrcoef(pearson_ses_dict[sub_id]['ses-1'], pearson_ses_dict[sub_id2]['ses-2'])[0,1])
        except KeyError: 
            pass

lasso_corr_dict = {}
for sub_id in lasso_ses_dict.keys(): 
    if sub_id not in lasso_corr_dict.keys(): 
        lasso_corr_dict[sub_id] = {}

    for sub_id2 in lasso_ses_dict.keys(): 
        try: 
            lasso_corr_dict[sub_id][sub_id2] = (np.corrcoef(lasso_ses_dict[sub_id]['ses-1'], lasso_ses_dict[sub_id2]['ses-2'])[0,1])
        except KeyError: 
            pass

uoi_corr_dict = {}
for sub_id in uoi_ses_dict.keys(): 
    if sub_id not in uoi_corr_dict.keys(): 
        uoi_corr_dict[sub_id] = {}

    for sub_id2 in uoi_ses_dict.keys(): 
        try: 
            uoi_corr_dict[sub_id][sub_id2] = (np.corrcoef(uoi_ses_dict[sub_id]['ses-1'], uoi_ses_dict[sub_id2]['ses-2'])[0,1])
        except KeyError: 
            pass

fp_list = []
for key, value in pearson_corr_dict.items(): 
    fp_list.append( (key == (max(pearson_corr_dict[key],key=pearson_corr_dict[key].get))))

sum(fp_list) / len(pearson_corr_dict) 

fp_list = []
for key, value in lasso_corr_dict.items(): 
    fp_list.append( (key == (max(lasso_corr_dict[key],key=lasso_corr_dict[key].get))))

sum(fp_list) / len(lasso_corr_dict) 

fp_list = []
for key, value in uoi_corr_dict.items(): 
    fp_list.append( (key == (max(uoi_corr_dict[key],key=uoi_corr_dict[key].get))))

#this is a little suspicious
sum(fp_list) / len(uoi_corr_dict) 

models = ['Pearson', 'LASSO', 'UoI']

plt.bar(models, accuracies)
plt.ylim(0,1)
plt.title('"Fingerprinting" Accuracy') 


