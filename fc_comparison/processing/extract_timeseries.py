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
#     display_name: fc_w_datalad
#     language: python
#     name: env
# ---

# This script takes in pre-processed niftis, and outputs a CSV with the timeseries by parcel. 

# +
import numpy as np

import nilearn
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting
from nilearn.plotting import plot_carpet
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

import nibabel as nib

import argparse

import datalad.api as dl

from glob import glob

import sys
import os 
import os.path as op
# -


#to access git-annex, add env bin to $PATH
#add to jupyter kernel spec to get rid of this line
os.environ["PATH"] = "/global/homes/m/mphagen/miniconda3/envs/fc_w_datalad/bin:" + os.environ["PATH"]

# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser("extract_timeseries.py")
parser.add_argument('--subject_id',  default='117728') 
parser.add_argument('--atlas_name', default='schaefer')
parser.add_argument('--n_rois', default=100)
parser.add_argument('--resolution_mm', default=1) #I don't remember where I got this #
parser.add_argument('--yeo_networks', default=7)

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
fc_data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity'
results_path = op.join(fc_data_path, 'derivatives', 
                       'parcellated-timeseries', 
                       f'sub-{subject_id}')
os.makedirs(results_path, exist_ok=True)

rest_scans = glob(op.join(fc_data_path, 
                          subject_id, 'MNINonLinear/Results/rfMRI*', 
                          '*clean.nii.gz'))

print(f"Found {len(rest_scans)} rest scans for subject {subject_id}") 
# -

#create pseudo-bids naming mapping dict
#assuming that the RL was always run before LR
bids_dict = {
    'rfMRI_REST1_RL': 'ses-1_run-1',
    'rfMRI_REST2_RL': 'ses-2_run-1',
    'rfMRI_REST2_LR': 'ses-2_run-2',
    'rfMRI_REST1_LR': 'ses-1_run-2' 
}

# +
#add elif here for other atlas choice
if atlas_name == 'schaefer': 
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,
                                                  yeo_networks,
                                                  resolution_mm)
    atlas = schaefer['maps']

masker = NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')


# -

def parcellate_data(file): 
    
    try: 
        ses_string = bids_dict[file.split('/')[-2]]
    except KeyError: 
        return
        
    dl.get(file, dataset='/pscratch/sd/m/mphagen/hcp-functional-connectivity')
    
    data = nib.load(file)
    
    time_series = masker.fit_transform(data)
    
    dl.drop(file, dataset='/pscratch/sd/m/mphagen/hcp-functional-connectivity')
    
    os.makedirs(op.join(results_path, ses_string), exist_ok=True)
    time_series.tofile(op.join(results_path, 'func', ses_string, f'{atlas_name}-{n_rois}'
                               f'sub-{subject_id}_ses-{ses_string}_task-Rest_atlas-{atlas_name}{n_rois}_timeseries.tsv'), 
                        sep = '/t')
    return time_series


for file in rest_scans: 
    ts = parcellate_data(file)  


