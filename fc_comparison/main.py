__all__ = ['parcellate_data']

import datalad.api as dl
import nibabel as nib

import nilearn

import os
import os.path as op

from nilearn.input_data import NiftiMasker, NiftiLabelsMasker


#create pseudo-bids naming mapping dict
#assuming that the RL was always run before LR
bids_dict = {
    'rfMRI_REST1_RL': 'ses-1_run-1',
    'rfMRI_REST2_RL': 'ses-2_run-1',
    'rfMRI_REST2_LR': 'ses-2_run-2',
    'rfMRI_REST1_LR': 'ses-1_run-2' 
}

def parcellate_data(file, dataset_path): 
    
    """
    Takes a minimally processed HCP fMRI scan and saves a parcellated 
    time series as a csv.
    
    file: /path/to/file.nii.gz
    dataset_path: for datalad dataset parameter
    masker: NiftiMasker object initalized with atlas and  
    
    """
    
    try: 
        ses_string = bids_dict[file.split('/')[-2]]
    except KeyError: 
        return
        
    dl.get(file, dataset=dataset_path)
    
    data = nib.load(file)
    
    time_series = masker.fit_transform(data)
    
    dl.drop(file, dataset=dataset_path)
    
    
    return time_series, ses_string
   
