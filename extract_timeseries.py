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
# -
from glob import glob

import os.path as op

# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='761957')           # positional argument
args = parser.parse_args([])
subject_id = args.subject_id
# +
#glob(op.join('/pscratch/sd/m/mphagen/hcp-functional-connectivity', subject_id, 'MNINonLinear/Results/*/*nii.gz')) 

data_files = glob(op.join('/pscratch/sd/m/mphagen/temp_data/*.nii.gz')) 

# +
n_rois=100
yeo_networks=7
resolution_mm=1

schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,yeo_networks,resolution_mm)
atlas = schaefer['maps']
# -

data = nib.load(data_files[0])
masker = NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')
time_series = masker.fit_transform(data)

time_series.tofile('test_timeseries.csv', sep = ',')


