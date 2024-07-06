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
import time

import nilearn
from nilearn.connectome import ConnectivityMeasure

from mpi4py import MPI

import pyuoi
from pyuoi.linear_model import UoI_Lasso
from pyuoi.utils import log_likelihood_glm, AIC, BIC

import numpy as np

import matplotlib.pyplot as plt

import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


import pickle
import pandas as pd

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='205220') 
parser.add_argument('--atlas_name', default='schaefer')
parser.add_argument('--n_rois', default=100) #default for hcp
parser.add_argument('--n_trs', default=1200) #default for hcp
parser.add_argument('--n_folds', default=5) 
parser.add_argument('--model', default='lassoBIC') 
parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/hcp-functional-connectivity') 
parser.add_argument('--task', default='rest') 


#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
    
subject_id = args.subject_id
atlas_name = args.atlas_name
n_rois = int(args.n_rois)
n_trs = int(args.n_trs)
n_folds = args.n_folds
model_str = args.model
fc_data_path = args.fc_data_path
task = args.task
print(args)

# +
ts_files = glob(op.join(fc_data_path, 
                        'derivatives', 
                        f'timeseries_{atlas_name}-{n_rois}', 
                        f'sub-{subject_id}', 'ses-*', '*.csv'))


results_path = op.join(fc_data_path, 
                        'derivatives', 
                        f'connectivity-{model_str}',
                        f'sub-{subject_id}', 
                        'func')

os.makedirs(results_path, exist_ok=True)

print(f"Found {len(ts_files)} {task} scans for subject {subject_id}.") 

print(f"Saving results to {results_path}.")
# -

random_state =1 

# +
if model_str == 'uoiLasso': 
    uoi_lasso = UoI_Lasso(estimation_score="BIC")

    comm = MPI.COMM_WORLD
    
    uoi_lasso.copy_X = True
    uoi_lasso.estimation_target = None
    uoi_lasso.logger = None
    uoi_lasso.warm_start = False
    uoi_lasso.comm = comm
    uoi_lasso.random_state = 1
    uoi_lasso.n_lambdas = 100
    
    model = uoi_lasso

elif model_str == 'lassoCV': 
    lasso = LassoCV(fit_intercept = True,
                    cv = 5, 
                    n_jobs=-1, 
                    max_iter=2000)
    
    model = lasso
    
elif model_str == 'lassoBIC': 
    lasso = LassoLarsIC(criterion='bic',
                        fit_intercept = True,
                        max_iter=2000)
    
    model = lasso
    
elif model_str == 'eNetCV':
    enet = ElasticNetCV(fit_intercept = True,
                        cv = 5, 
                        n_jobs=-1, 
                        max_iter=2000)
    model = enet
    
elif model_str in ['correlation', 'tangent']: 
    model = ConnectivityMeasure(
            kind=model_str)


# -

def calc_fc(train_ts, test_ts, n_rois, model, **kwargs): 
    assert train_ts.shape[1] == n_rois == test_ts.shape[1]
    fc_mat = np.zeros((n_rois,n_rois))
    
    inner_rsq_dict = {
        'train': list(), 
        'test': list()
    }

    for target_idx in range(train_ts.shape[1]):    
        y_train = np.array(train_ts[:,target_idx])
        X_train = np.delete(train_ts, target_idx, axis=1) 

        
        y_test = np.array(test_ts[:,target_idx])
        X_test = np.delete(test_ts, target_idx, axis=1)
        
        model.fit(X=X_train, y=y_train)

        fc_mat[target_idx,:] = np.insert(model._final_estimator.coef_, target_idx, 1) 
        test_rsq, train_rsq = eval_metrics(X_train, y_train, 
                                           X_test, y_test, model)

        inner_rsq_dict['test'].append(test_rsq)
        inner_rsq_dict['train'].append(train_rsq)

    return(fc_mat, inner_rsq_dict, model)


def eval_metrics(X_train, y_train,
                 X_test, y_test, model):
    
    test_rsq = r2_score(y_test, model.predict(X_test))
    
    train_rsq = r2_score(y_train, model.predict(X_train))

    return(test_rsq, train_rsq)


# +
def write_metadata(rsq_dict, atlas_name, n_rois, file_name): 
    atlas_lookup = {'schaefer': 'Schaefer_100_Yeo_17'}

    pd.DataFrame({"index": np.arange(0,n_rois),
             "node_file": atlas_lookup[atlas_name], 
              "node_file_index": np.arange(0,n_rois), 
              "model_accuracy_test": rsq_dict[fold_idx]['test'], 
              "model_accuracy_train": rsq_dict[fold_idx]['train']}).to_csv(
        node_idx_file, sep='\t')


# -

import warnings
warnings.simplefilter("ignore")
#iterate over each scan for a subject
for file in ts_files:   
    ses_string = file.split('/')[-2]
    bids_str = f'sub-{subject_id}_{ses_string}_task-{task}_meas-{model_str}'

    if  'run-combined' in ses_string:        
        time_series = np.loadtxt(file, delimiter=',').reshape(2*n_trs, n_rois)
    
    else: 
        time_series = np.loadtxt(file, delimiter=',').reshape(n_trs, n_rois)

    if model_str in ["lassoCV", "uoiLasso", "eNetCV", "lassoBIC"] :
        print(f"Calculating {model_str} FC for {ses_string}")

        kfolds = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state)

        rel_mats = np.zeros((n_rois,n_rois, n_folds))

        rsq_dict = {} 
        for fold_idx, (train_idx, test_idx) in enumerate( 
                                        kfolds.split(X=time_series) ): 
            print(fold_idx)

            train_ts = time_series[train_idx, :]
            test_ts = time_series[test_idx, :]
            
            model = make_pipeline(StandardScaler(with_mean=False), 
                                  LassoLarsIC())

            start_time = time.time()
            
            rel_mats[:,:, fold_idx], rsq_dict[fold_idx], fit_model = calc_fc(
                train_ts, test_ts, n_rois, model=model)
            
            print(time.time() - start_time, ' seconds') 

            relmat_file = f'{bids_str}_desc-fold{fold_idx}_relmat.dense.tsv'
            node_idx_file = f'{bids_str}_desc-fold{fold_idx}_nodeindices.tsv'
        
            os.makedirs(op.join(results_path, ses_string[0:5]), exist_ok=True)
            np.savetxt(op.join(results_path, ses_string[0:5], relmat_file),
                       rel_mats[:,:, fold_idx], delimiter='\t')
                
            write_metadata(rsq_dict, atlas_name, n_rois, node_idx_file)
                        
    elif model_str in ['correlation', 'tangent']: 
        print(f"Calculating {model_str} FC for {ses_string}")
        
        relmat_file = f'{bids_str}_relmat.dense.tsv'
        node_idx_file = f'{bids_str}_nodeindices.tsv'
        
        corr_mat = model.fit_transform([time_series])[0]
        np.fill_diagonal(corr_mat, 1)
        
        np.savetxt(op.join(results_path, relmat_file),
                       corr_mat, delimiter='\t')


