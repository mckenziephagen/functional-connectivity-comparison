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
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_carpet
from nilearn.connectome import ConnectivityMeasure

from mpi4py import MPI

from pyuoi.linear_model import UoI_Lasso

import numpy as np

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='102109') 
parser.add_argument('--atlas_name', default='shaefer')
parser.add_argument('--n_rois', default=100)


#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
    
subject_id = args.subject_id
atlas_name = args.atlas_name
n_rois = args.n_rois

n_folds=5

# +
fc_data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity'

ts_files = glob(op.join(fc_data_path, 'derivatives', f'fc_{atlas_name}-{n_rois}', f'sub-{subject_id}', '*.csv'))
results_path = op.join(fc_data_path, 'derivatives', f'fc-matrices_{atlas_name}-{n_rois}', f'sub-{subject_id}')

os.makedirs(results_path, exist_ok=True)

# +
uoi_lasso = UoI_Lasso()

comm = MPI.COMM_WORLD
rank = comm.rank

uoi_lasso.copy_X = True
uoi_lasso.estimation_target = None
uoi_lasso.logger = None
uoi_lasso.warm_start = False
uoi_lasso.selection_frac = 0.9
uoi_lasso.comm = comm
uoi_lasso.random_state = 1
# -

for file in ts_files[1:2]: #iterate over a subject's different runs
    
    time_series = np.loadtxt(file, delimiter=',').reshape(1200, 100)

    #fit uoi
    uoi_conn_mat = np.zeros((100,100))
    ses_string = ts_files[1].split('/')[-2]
    
    #sylvia's code has an inner and outer cv loop
    kfolds = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate( kfolds.split(X=time_series) ): 
        print(fold_idx)
        uoi_conn_mat = np.zeros((100,100))

        train_ts = time_series[train_idx, :]
        test_ts = time_series[test_idx, :]
        
        start_time = time.time()
        for target_idx in range(train_ts.shape[1]): 
            #pull this as function (down to print(rsq))
            y = np.array(train_ts[:,target_idx])
            X = np.delete(train_ts, target_idx, axis=1) 
        
            uoi_lasso.fit(X=X , y=y)
            
            uoi_conn_mat[target_idx,:] = np.insert(uoi_lasso.coef_, target_idx, 0) 
            y_pred_test = uoi_lasso.predict(np.delete(test_ts, target_idx, axis=1))
            
            rsq = r2_score(y, y_pred_test)
            print(rsq)
            
        print(time.time() - start_time, ' seconds') 
   
        uoi_conn_mat.tofile(op.join(results_path,
                               f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fold-{fold_idx}_fc-uoiLasso.csv'), 
                        sep = ',')
     
    #pull this out as function
    correlation_measure = ConnectivityMeasure(kind="correlation" )
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix.tofile(op.join(results_path,
                               f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fold-{fold_idx}_fc-fullCorrelation.csv'), 
                                sep = ',')

    #pull this out as function
    partial_correlation_measure = ConnectivityMeasure(kind="partial correlation" )
    partial_correlation_matrix = partial_correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(partial_correlation_matrix, 0)
    partial_correlation_matrix.tofile(op.join(results_path,
                               f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_{ses_string}_fold-{fold_idx}_fc-partialCorrelation.csv'), 
                                sep = ',')


