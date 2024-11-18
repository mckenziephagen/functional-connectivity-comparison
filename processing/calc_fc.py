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

# +
import sys
import os

from numpy.matlib import repmat

# -

#work around until I install fc_comparison as an actual package
sys.path.append(os.path.dirname('../fc_comparison'))

# +
import time

import nilearn

from pyuoi.utils import log_likelihood_glm, AIC, BIC

import numpy as np

import matplotlib.pyplot as plt

import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pickle

from fc_comparison.models import init_model, run_model

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='101107') 
parser.add_argument('--atlas_name', default='schaefer')
parser.add_argument('--n_rois', default=100, type=int) #default for hcp; 
parser.add_argument('--n_trs', default=1200, type=int) #default for hcp;
parser.add_argument('--n_folds', default=5) 
parser.add_argument('--model', default='uoi-lasso') 
parser.add_argument('--cv', default='blocks') 

parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/hcp-functional-connectivity') 
parser.add_argument('--max_iter', default=5000) #testing - default for uoi is 1000

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
    
subject_id = args.subject_id
atlas_name = args.atlas_name
n_rois = args.n_rois
n_trs = args.n_trs
n_folds = args.n_folds
model_str = args.model
cv = args.cv
fc_data_path = args.fc_data_path

max_iter = 5000
print(args)

# +
#this is brittle - but adhering to BEP-17 will make it les brittle
ts_files = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'parcellated-timeseries',
                        f'sub-{subject_id}',
                        '*',
                        '*',
                        '*.tsv'))

results_path = op.join(fc_data_path, 
                       'derivatives',
                       'connectivity-matrices',
                       model_str,
                       f'sub-{subject_id}')
                       
os.makedirs(results_path, exist_ok=True)

assert len(ts_files) > 0 

print(f"Found {len(ts_files)} rest scans for subject {subject_id}.") 

print(f"Saving results to {results_path}.")
# -

random_state = 1 

model = init_model(model_str, max_iter)

print(model)


def split_kfold(cv): 
    if cv == 'random': 
        
        kfold = KFold(n_splits=n_folds,
                      shuffle=True,
                      random_state=random_state)
        
        splits = kfolds.split(X=time_series)
        
    if cv == 'blocks':
        group =  repmat(np.arange(1, n_folds+1), 
                        int(n_trs/n_folds), 1).T.ravel()
        
        kfold = GroupKFold(n_splits=n_folds, 
                           random_state=random_state)
        
        splits = kfold.split(X=time_series, groups=group) 
        
    return splits


def read_ts(file, ses_string, n_trs, n_rois):
    if  'run-combined' in ses_string:        
        time_series = np.loadtxt(file, delimiter=',').reshape(-1, 100) #TODO: maybe change hard coding
            #use -1 to infer if I know how many regions but not how many trs 
    
    else: 
        time_series = np.loadtxt(file, delimiter='\t').reshape(n_trs, n_rois)

        
    return(time_series)

#iterate over each scan for a subject
for file in ts_files:   
    ses_string = file.split('/')[-2]
    print(ses_string)
    
    
    if model_str in ["lasso-cv", "uoi-lasso", "enet", "lasso-bic"] :
        print(f"Calculating {model_str} FC for {ses_string}")
        
        time_series = read_ts(file, ses_string, n_trs, n_rois)
        
        splits = split_kfold(cv)

        fc_mats = {}
        rsq_dict = {}

        for fold_idx, (train_idx, test_idx) in enumerate( splits ): 

            train_ts = time_series[train_idx, :]
            test_ts = time_series[test_idx, :]
            
            scaler = StandardScaler()
            scaler.fit_transform(train_ts)
            scaler.transform(test_ts)

            start_time = time.time()
            fc_mats[fold_idx], rsq_dict[fold_idx], tst_model = run_model(train_ts, 
                                                                       test_ts, 
                                                                       n_rois, 
                                                                       model=model)

            print(time.time() - start_time, ' seconds') 

        #TODO: helper function to manage filenames 
        mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fc-{model_str}_model.pkl'
        with open(op.join(results_path,mat_file), 'wb') as f:
            pickle.dump(fc_mats, f)
        with open(op.join(results_path, 
                          f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fc-{model_str}_r2.pkl'), 
                  'wb') as f: 
            pickle.dump(rsq_dict, f)
            
    elif model_str in ['correlation', 'tangent']: 
        print(f"Calculating {model_str} FC for {ses_string}")

        corr_mat = model.fit_transform([time_series])[0]
        np.fill_diagonal(corr_mat, 0)
    
        mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fc-{model_str}.pkl'

        with open(op.join(results_path,mat_file), 'wb') as f:
            pickle.dump(corr_mat, f) 








