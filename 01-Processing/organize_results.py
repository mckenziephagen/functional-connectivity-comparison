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

import os.path as op
from glob import glob
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# +
fc_data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity'


pearson_results_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_schaefer*', '*', '*correlation*')
lasso_results_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_schaefer*', '*', '*fc-lasso*model*')
uoi_results_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_schaefer*', '*', '*uoi-*model*')
# -

uoi_r2_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_schaefer*', '*', '*uoi-*r2*')

pearson_results_files = glob(pearson_results_path) 
lasso_result_files = glob(lasso_results_path)
pyuoi_result_files = glob(uoi_results_path) 

pyuoi_r2_files = glob(uoi_r2_path)

len(pearson_results_files)

len(lasso_result_files) 

lasso_dict = {} 
for idx, file in enumerate(lasso_result_files): 
    with open(file, 'rb') as f:
         mat = pickle.load(f)    
    ses_id = '_'.join(file.split('/')[-1].split('_')[-4:-2])
    sub_id = file.split('/')[-2]

    if sub_id not in lasso_dict.keys(): 
        lasso_dict[sub_id] = {} 
    #average five folds together
    lasso_dict[sub_id].update({ses_id: np.median(np.array([*mat.values()]), axis=0)})
#now I have a dictionary of lasso fc matrices averaged over five folds  
with open(op.join('results', f'{str(datetime.date.today())}_lasso_dict.pkl'), 'wb') as f:
        pickle.dump(lasso_dict, f)

pearson_dict = {} 
for idx, file in enumerate(pearson_results_files): 
    with open(file, 'rb') as f:
         mat = pickle.load(f)    
    ses_id = '_'.join(file.split('/')[-1].split('_')[-3:-1])
    sub_id = file.split('/')[-2]

    if sub_id not in pearson_dict.keys(): 
        pearson_dict[sub_id] = {} 
    #average five folds together
    pearson_dict[sub_id].update({ses_id: mat})
with open(op.join('results', f'{str(datetime.date.today())}_pearson_dict.pkl'), 'wb') as f:
            pickle.dump(pearson_dict, f)


# +
uoi_dict = {} 
for idx, file in enumerate(pyuoi_result_files): 
    with open(file, 'rb') as f:
         mat = pickle.load(f)    
    ses_id = '_'.join(file.split('/')[-1].split('_')[-4:-2])
    sub_id = file.split('/')[-2]

    if sub_id not in uoi_dict.keys(): 
        uoi_dict[sub_id] = {} 
    #average five folds together
    uoi_dict[sub_id].update({ses_id: np.median(np.array([*mat.values()]), axis=0)})
    
with open(op.join('results', f'{str(datetime.date.today())}_uoi_dict.pkl'), 'wb') as f:
        pickle.dump(uoi_dict, f)
# -

pearson_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub'] )
for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': pearson_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner, 
                            'pos': list(range(1,10001,1))})
        pearson_icc_df = pd.concat([pearson_icc_df, temp_df])
pearson_icc_df.to_csv('pearson_icc_df.csv')

lasso_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub', 'pos'] )
for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': lasso_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner, 
                              'pos': list(range(1,10001,1))})
        lasso_icc_df = pd.concat([lasso_icc_df, temp_df])
lasso_icc_df.to_csv('lasso_icc_df.csv')

uoi_dict[outer][inner].shape

# +
uoi_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub'] )
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': uoi_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner})
        uoi_icc_df = pd.concat([uoi_icc_df, temp_df])
        
uoi_icc_df.to_csv('uoi_icc_df.csv')
# -

len(uoi_dict)

lasso_dict.keys() 

len(pearson_dict)

r2_mean_list = []
for i in pyuoi_r2_files: 
    with open(i, 'rb') as f:
         mat = pickle.load(f)  
    r2_df = pd.json_normalize(mat).filter(like='test')
    
    r2_mean_list.append(np.mean(r2_df.explode(list(r2_df.columns))))

#I shoudl investigate those lower ones and cut them
plt.hist(r2_mean_list)
plt.title('Average Union of Intersections Model Accuracy Per Scan') 

plt.hist(np.mean(r2_df.explode(list(r2_df.columns)), axis=1))


