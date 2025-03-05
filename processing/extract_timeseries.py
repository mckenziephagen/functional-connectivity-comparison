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
#     display_name: fc_w_datalad
#     language: python
#     name: env
# ---

# This script takes in pre-processed niftis, and outputs a CSV with the timeseries by parcel. 

# +
import numpy as np

import argparse

import nilearn 
from nilearn import datasets 
from nilearn.maskers import NiftiLabelsMasker

from glob import glob

import sys
import os 
import os.path as op

sys.path.append(os.path.dirname('../fc_comparison'))

from fc_comparison.files import parcellate_data
# -


#to access git-annex, add env bin to $PATH
#add to jupyter kernel spec to get rid of this line
os.environ["PATH"] = "/global/homes/m/mphagen/miniconda3/envs/fc_w_datalad/bin:" + os.environ["PATH"]

# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser("extract_timeseries.py")
parser.add_argument('--subject_id',  default='101107') 
parser.add_argument('--atlas_name', default='schaefer')
parser.add_argument('--n_rois', default=100)
parser.add_argument('--resolution_mm', default=1) 
parser.add_argument('--yeo_networks', default=7)
parser.add_argument('--dataset', default='HCP') 

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])

subject_id = args.subject_id
atlas_name = args.atlas_name
n_rois = args.n_rois
resolution_mm = args.resolution_mm
yeo_networks = args.yeo_networks

print(args)
# +
def define_paths(path_prefix, dataset, results_str): 
    """
    DOCSTRING
    """
    if dataset == 'HCP': 
        dataset_path = op.join(path_prefix, 'hcp-functional-connectivity') 
   
    derivatives_path = op.join(dataset_path, 'derivatives') 
    results_dir = op.join(derivatives_path, results_str)

    path_dict = {'dataset_path': dataset_path, 
                 'derivatives_path': derivatives_path,
                 'results_dir': results_dir}
    
    
    return path_dict
        
        
def save_parcellated_ts(results_dir, subject_id, ses_string, atlas_name, n_rois): #make this a dictionary
     
    tsv_path = op.join(results_dir, 
                       ses_string, 
                       f'{atlas_name}-{n_rois}')
                       
    os.makedirs(tsv_path, exist_ok=True)
    
    file_str =  f'sub-{subject_id}_{ses_string}_task-Rest_atlas-{atlas_name}{n_rois}_timeseries.tsv'
                
    ts.tofile(op.join(tsv_path, file_str), 
              sep = '\t')
        


# +
path_dict = define_paths('/pscratch/sd/m/mphagen', 'HCP', 'parcellated-timeseries')
                       
os.makedirs(path_dict['results_dir'], exist_ok=True)

rest_scans = glob(op.join(path_dict['dataset_path'], 
                          subject_id, 
                          'MNINonLinear/Results/rfMRI*', 
                          '*clean.nii.gz'))

assert len(rest_scans) > 0 

print(f"Found {len(rest_scans)} rest scans for subject {subject_id}") 

# +
#add elif here for other atlas choice
if atlas_name == 'schaefer': 
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,
                                                  yeo_networks,
                                                  resolution_mm)
    atlas = schaefer['maps']

masker = NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')
# -

for file in rest_scans: 
    ts, ses_string = parcellate_data(file, path_dict['dataset_path'], masker)  
    
    save_parcellated_ts(path_dict['dataset_path'],
                        subject_id, 
                        ses_string, 
                        atlas_name,
                        n_rois)
    
    #double check BIDS nroi (does this need to be desc instead?

ts[:,0]

import matplotlib.pyplot as plt

plt.plot(ts[150:700,20], linewidth=2.5)
plt.axis('off')
plt.savefig('timeseries4.png', transparent=True) 


