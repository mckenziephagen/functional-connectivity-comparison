# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
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
sys.path.append(os.path.dirname('/global/homes/m/mphagen/functional_connectivity_comparison/fc_comparison'))

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

from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pickle

from fc_comparison.models import init_model, run_model

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='112516')
parser.add_argument('--ses_id', default='ses-1') 
parser.add_argument('--combine_runs', default=True) 

parser.add_argument('--atlas_name', default='schaefer')
parser.add_argument('--n_rois', default=100, type=int) #default for hcp; 
parser.add_argument('--n_trs', default=1200, type=int) #default for hcp;
parser.add_argument('--n_folds', default=5) 
parser.add_argument('--model', default='correlation') 
parser.add_argument('--cv', default='blocks') 

parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/hcp-functional-connectivity') 
parser.add_argument('--max_iter', default=1000) 

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
    
subject_id = args.subject_id
ses_id = args.ses_id
combine_runs = args.combine_runs
atlas_name = args.atlas_name
n_rois = args.n_rois
n_trs = args.n_trs
n_folds = args.n_folds
model_str = args.model
cv = args.cv
fc_data_path = args.fc_data_path

max_iter = 1000

random_state = 1
print(args)

# +
#this is brittle - but adhering to BEP-17 will make it les brittle
ts_files = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'parcellated-timeseries',
                        f'sub-{subject_id}',
                        f'{ses_id}*',
                        '*',
                        '*.tsv'))

results_path = op.join(fc_data_path, 
                       'derivatives',
                       'connectivity-matrices',
                       f'{model_str}-{cv}',
                       f'sub-{subject_id}')
                       
os.makedirs(results_path, exist_ok=True)

assert len(ts_files) == 2 

print(f"Found {len(ts_files)} rest scans for subject {subject_id}.") 

print(f"Saving results to {results_path}.")
# -

model = init_model(model_str, max_iter, random_state)

print(model)


def split_kfold(cv, time_series): 
    if cv == 'random': 
        
        kfolds = KFold(n_splits=n_folds,
                      shuffle=True)
        
        splits = kfolds.split(X=time_series)
        
    if cv == 'blocks':
        group =  repmat(np.arange(1, n_folds+1), 
                        int(2400/n_folds), 1).T.ravel()
        
        kfold = GroupKFold(n_splits=n_folds)
        
        splits = kfold.split(X=time_series, groups=group) 
        
    if cv == 'timeseries': 
        tscv = TimeSeriesSplit()
        splits = tscv(tscv.split(X=time_series))
        
    return splits


def read_ts(file, n_trs, n_rois):
    if len(ts_files) > 1: 
        
        time_series = np.loadtxt(file, delimiter='\t').reshape(-1, n_rois)
    
    else: 
        time_series = np.loadtxt(file, delimiter='\t').reshape(n_trs, n_rois)

        
    return(time_series)

time_series = np.append(read_ts(ts_files[0], n_trs, n_rois), 
                        read_ts(ts_files[1], n_trs, n_rois)).reshape(2400,100)


# +
if model_str in ["lasso-cv", "uoi-lasso", "enet", "lasso-bic"] :
    print(f"Calculating {model_str} FC for {ses_id}")
    
    
    splits = split_kfold(cv, time_series)

    fc_mats = {}
    rsq_dict = {}

    for fold_idx, (train_idx, test_idx) in enumerate( splits ): 
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: range={train_idx[0]}, {train_idx[-1]}")
        print(f"  Test:  index={test_idx[0]}, {test_idx[-1]}")


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
        print(np.mean(rsq_dict[fold_idx]['test'])) 

    #TODO: helper function to manage filenames 
    mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_id}_fc-{model_str}_model.pkl'
   
    with open(op.join(results_path,mat_file), 'wb') as f:
        pickle.dump(fc_mats, f)
    
    with open(op.join(results_path, 
                      f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_id}_fc-{model_str}_r2.pkl'), 
              'wb') as f: 
        pickle.dump(rsq_dict, f)
        
elif model_str in ['correlation', 'tangent']: 
    print(f"Calculating {model_str} FC for {ses_id}")

    corr_mat = model.fit_transform([time_series])[0]
    np.fill_diagonal(corr_mat, 0)

    mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_id}_fc-{model_str}_desc-kfold{cv}.pkl'

    with open(op.join(results_path,mat_file), 'wb') as f:
        pickle.dump(corr_mat, f) 
# -

import matplotlib.pyplot as plt

plt.hist(np.arctanh(corr_mat.ravel()) ) 


