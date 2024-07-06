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

# Takes output from `calc_fc.py` and organizes it into useable dataframes and dictionaries for downstream analyses. 

import os
import os.path as op
from glob import glob
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse

date_str = str(datetime.date.today())

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lasso-bic') 
parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/ds000228-fmriprep') 


#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])

model_str = args.model
fc_data_path = args.fc_data_path

# +
organized_result_path = op.join(fc_data_path, 
                                'results', 
                                date_str)

os.makedirs(organized_result_path,exist_ok=True) 


fc_matrices_path = op.join(fc_data_path, 
                           'derivatives', 
                           'fc-matrices_schaefer-100')

pearson_model_path = op.join(fc_matrices_path, 
                             '*',
                             '*fc-correlation.pkl')

lasso_model_path = op.join(fc_matrices_path, 
                             '*', 
                             'lasso-bic', 
                             '*fc-lasso-bic_model.pkl')

uoi_model_path = op.join(fc_matrices_path, 
                           '*',
                           '*fc-uoi-lasso_model.pkl')
# +
uoi_r2_path = op.join(fc_matrices_path,
                       '*', 
                       '*fc-uoi-lasso_r2.pkl')

lasso_r2_path = op.join(fc_matrices_path,
                        '*', 
                        'lasso-bic',
                        '*fc-lasso-bic_r2.pkl')
# -

pearson_files = glob(pearson_model_path) 
lasso_model_files = glob(lasso_model_path)
uoi_model_files = glob(uoi_model_path) 

print(len(pearson_files)) 
print(len(lasso_model_files)) 
print(len(uoi_model_files)) 

uoi_r2_files = glob(uoi_r2_path)
lasso_r2_files = glob(lasso_r2_path)

print(len(uoi_r2_files))
print(len(lasso_r2_files))


def unpack_model_results(result_files, result_string): 
    result_dict = {} 
    for idx, file in enumerate(result_files): 
      #  print(file) 
        with open(file, 'rb') as f:
             mat = pickle.load(f)    
        ses_id = '_'.join(file.split('/')[-1].split('_')[3:5])
        sub_id = file.split('/')[-1].split('_')[0]
        
        if sub_id not in result_dict.keys(): 
            result_dict[sub_id] = {} 
            
        if result_string == 'pearson': 
            result_dict[sub_id].update({ses_id: mat})
        else: 
            result_dict[sub_id].update({ses_id: 
                                        np.median(np.array([*mat.values()]), 
                                                  axis=0)})

        with open(op.join(organized_result_path,
                          f'{result_string}_dict.pkl'), 'wb') as f:
            pickle.dump(result_dict, f)
            
    return(result_dict) 


# write dictionaries of results into pkls

lasso_dict = unpack_model_results(lasso_model_files, 'lasso-bic') 
uoi_dict = unpack_model_results(uoi_model_files, 'uoi') 
pearson_dict = unpack_model_results(pearson_files, 'pearson') 

# write dataframes for reliability analyses (this can be a function). 

pearson_icc_df = pd.DataFrame(columns=['values', 'ses', 
                                       'sub', 'pos'] )
for sub in pearson_dict.keys():
    for ses in pearson_dict[sub].keys():
        temp_df = pd.DataFrame(data = 
                      {'values': pearson_dict[sub][ses].ravel(), 
                       'sub':  sub, 
                       'ses': ses, 
                       'pos': list(range(1,10001,1))})
        pearson_icc_df = pd.concat([pearson_icc_df, temp_df])
pearson_icc_df.to_csv(op.join(organized_result_path, 
                              'pearson_icc_df.csv'))

lasso_icc_df = pd.DataFrame(columns=['values', 'ses', 
                                     'sub', 'pos'] )
for sub in lasso_dict.keys():
    for ses in lasso_dict[sub].keys():
        temp_df = pd.DataFrame(data = 
                   {'values': lasso_dict[sub][ses].ravel(), 
                    'sub':  sub, 
                    'ses': ses, 
                    'pos': list(range(1,10001,1))})
        lasso_icc_df = pd.concat([lasso_icc_df, temp_df])
lasso_icc_df.to_csv(op.join(organized_result_path, 
                            'lasso_icc_df.csv'))

lasso_icc_df.loc[:, ['values', 'sub']].pivot(columns='sub').T

lasso_icc_df.loc[,[].pivot(columns='sub').T

# +
uoi_icc_df = pd.DataFrame(columns=['values', 'ses', 
                                   'sub', 'pos'] )
for sub in uoi_dict.keys():
    for ses in uoi_dict[sub].keys():
        temp_df = pd.DataFrame(data = {
                         'values': uoi_dict[sub][ses].ravel(), 
                         'sub':  sub, 
                         'ses': ses, 
                         'pos': list(range(1,10001,1))})
        
        uoi_icc_df = pd.concat([uoi_icc_df, temp_df])
        
uoi_icc_df.to_csv(op.join(organized_result_path, 
                          'uoi_icc_df.csv'))
# -

# make wide format for R prediction code

# +
# make "wide" data here
# -

# pull out r2 values

# +
r2_lasso = []
for i in lasso_r2_files: 
    with open(i, 'rb') as f:
         mat = pickle.load(f)  
    r2_df = pd.json_normalize(mat).filter(like='test')
    
    r2_lasso.append(np.mean(r2_df.explode(list(r2_df.columns))))
    
    
r2_pyuoi = []
for i in pyuoi_r2_files: 
    with open(i, 'rb') as f:
         mat = pickle.load(f)  
    r2_df = pd.json_normalize(mat).filter(like='test')
    
    r2_pyuoi.append(np.mean(r2_df.explode(list(r2_df.columns))))
# +
#this is weird
plt.hist(r2_lasso) 
plt.hist(r2_pyuoi) 

plt.title('Histogram of UoI and Lasso Accuracies') 
# -

#I shoudl investigate those lower ones and cut them
plt.hist(r2_mean_list)
plt.title('Average Union of Intersections Model Accuracy Per Scan') 


