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
#     display_name: fc_w_datalad
#     language: python
#     name: env
# ---

import datalad

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

import nest_asyncio
nest_asyncio.apply()

import datalad.api as dl

from glob import glob

import os.path as op


# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='100206') 
parser.add_argument('--atlas', default='shaefer')
parser.add_argument('--n_rois', default=100)
parser.add_argument('--resolution_mm', default=1) #I don't remember where I got this #
parser.add_argument('--yeo_networks', default=7)

args = parser.parse_args([])
subject_id = args.subject_id
atlas = args.atlas
n_rois = args.n_rois
resolution_mm = args.resolution_mm
yeo_networks = args.yeo_networks
# -
rest_scans = glob(op.join('/pscratch/sd/m/mphagen/hcp-functional-connectivity', 
                         subject_id, 'MNINonLinear/Results/rfMRI*', '*clean.nii.gz'))

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
if atlas == 'schaefer': 
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,yeo_networks,resolution_mm)
    atlas = schaefer['maps']

masker = NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')

# -

rest_scans

dl.get(rest_scans[0], dataset='/pscratch/sd/m/mphagen/hcp-functional-connectivity')

for ii in rest_scans: 
    
    data = nib.load(rest_scans[ii])
    time_series = masker.fit_transform(data)
    
    
    
    time_series.tofile('test_timeseries.csv', sep = ',')


