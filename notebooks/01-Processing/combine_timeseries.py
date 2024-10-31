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
#combine timeseries for runs within the same session.
# -

from glob import glob
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse

# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser("extract_timeseries.py")
parser.add_argument('--subject_id',  default='117728')

try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])

subject_id = args.subject_id
# -

atlas_name ='schaefer'
n_rois=100

# +
fc_data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity'
ts_files = glob(op.join(fc_data_path, 'derivatives', 
                        f'fc_{atlas_name}-{n_rois}_timeseries', 
                        f'sub-{subject_id}', '*', '*.csv'))

results_path = op.join(fc_data_path, 'derivatives', 
                       f'fc_{atlas_name}-{n_rois}_timeseries', 
                       f'sub-{subject_id}')

# +
ts_files.sort()

ses_1 = list()
ses_2 = list()

_ = [ses_1.append(ts) for ts in ts_files if 'ses-1' in ts]
_ = [ses_2.append(ts) for ts in ts_files if 'ses-2' in ts]
# -

if len(ses_1) > 1: 
    os.mkdir(op.join(results_path, 'ses-1_run-combined') ) 
    combined_ts_1 = np.append(np.loadtxt(ses_1[0], delimiter=',').reshape(1200,100), 
          np.loadtxt(ses_1[1], delimiter=',').reshape(1200,100), axis=0)
    
    combined_ts_1.tofile(op.join(results_path, 'ses-1_run-combined', 
                               f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_timeseries.csv'), 
                        sep = ',')

if len(ses_2) > 1: 
    os.mkdir(op.join(results_path, 'ses-2_run-combined') ) 

    combined_ts_2 = np.append(np.loadtxt(ses_2[0], delimiter=',').reshape(1200,100), 
          np.loadtxt(ses_2[1], delimiter=',').reshape(1200,100), axis=0)
    
    combined_ts_2.tofile(op.join(results_path, 'ses-2_run-combined', 
                               f'sub-{subject_id}_{atlas_name}-{n_rois}_task-Rest_timeseries.csv'), 
                        sep = ',')
