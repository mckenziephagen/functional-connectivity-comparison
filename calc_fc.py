# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
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

from pyuoi.linear_model import UoI_Lasso
from pyuoi.utils import log_likelihood_glm, AIC, BIC

import numpy as np

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, RidgeCV

import pickle

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='205220') 
parser.add_argument('--atlas_name', default='shaefer')
parser.add_argument('--n_rois', default=100) #default = hcp
parser.add_argument('--n_trs', default=1200) #default = hcp
parser.add_argument('--n_folds', default=5) 
parser.add_argument('--model', default='correlation') 


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
print(args)

# +
fc_data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity'

ts_files = glob(op.join(fc_data_path, 'derivatives', f'fc_{atlas_name}-{n_rois}', f'sub-{subject_id}', '*', '*.csv'))
results_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_{atlas_name}-{n_rois}', f'sub-{subject_id}')

os.makedirs(results_path, exist_ok=True)

print(f"Found {len(ts_files)} rest scans for subject {subject_id}.") 

print(f"Saving results to {results_path}.")
# -

op.join(fc_data_path, 'derivatives', f'fc_{atlas_name}-{n_rois}', f'sub-{subject_id}', '*.csv')

random_state =1 

# +
if model_str == 'uoi_lasso': 
    uoi_lasso = UoI_Lasso(estimation_score="BIC")

    comm = MPI.COMM_WORLD
    rank = comm.rank

    uoi_lasso.copy_X = True
    uoi_lasso.estimation_target = None
    uoi_lasso.logger = None
    uoi_lasso.warm_start = False
    uoi_lasso.comm = comm
    uoi_lasso.random_state = 1
    uoi_lasso.n_lambdas = 100
    
    model = uoi_lasso

elif model_str == 'lasso': 
    lasso = LassoCV(fit_intercept = True,
                    cv = 5, 
                    n_jobs=-1, 
                    max_iter=2000)
    
    model = lasso

elif model_str in ['correlation', 'tangent']: 
    model = ConnectivityMeasure(
        kind=model_str)
    
    
# -

print(model)


# +
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

        fc_mat[target_idx,:] = np.insert(model.coef_, target_idx, 0) 
        test_rsq, train_rsq = eval_metrics(X_train, y_train, X_test, y_test, model)

        inner_rsq_dict['test'].append(test_rsq)
        inner_rsq_dict['train'].append(train_rsq)

      #  print(test_rsq)

    return(fc_mat, inner_rsq_dict)
       
        
# -

def eval_metrics(X_train, y_train, X_test, y_test, model):
    
    test_rsq = r2_score(y_test, model.predict(X_test))
    
    train_rsq = r2_score(y_train, model.predict(X_train))

    return(test_rsq, train_rsq)


#iterate over each scan for a subject
for file in ts_files[:2]: 
    time_series = np.loadtxt(file, delimiter=',').reshape(n_trs, n_rois)
    ses_string = ts_files[1].split('/')[-2]
    print(f"Calculating FC for {ses_string}")
    
    if model_str in ["lasso", "uoi_lasso"] :

        kfolds = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state)

        fc_mats = {}
        rsq_dict = {}

        for fold_idx, (train_idx, test_idx) in enumerate( kfolds.split(X=time_series) ): 
            print(fold_idx)

            train_ts = time_series[train_idx, :]
            test_ts = time_series[test_idx, :]

            start_time = time.time()
            fc_mats[fold_idx], rsq_dict[fold_idx] = calc_fc(train_ts, test_ts, n_rois, model=model)

            print(time.time() - start_time, ' seconds') 

        mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fc-{model_str}.pkl'
        with open(op.join(results_path,mat_file), 'wb') as f:
            pickle.dump(fc_mats, f) 
            
    elif model_str in ['correlation', 'tangent']: 
    
        corr_mat = correlation_measure.fit_transform([time_series])[0]
        np.fill_diagonal(corr_mat, 0)
    
        mat_file = f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fc-{model_str}.pkl'

        with open(op.join(results_path,mat_file), 'wb') as f:
            pickle.dump(corr_mat, f) 
