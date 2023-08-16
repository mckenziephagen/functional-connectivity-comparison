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
import nilearn
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting
from nilearn.plotting import plot_carpet
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker

import nibabel as nib

from pyuoi import UoI_Lasso

import numpy as np

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
# -

from mpi4py import MPI
from pyuoi.mpi_utils import load_data_MPI
from pyuoi.linear_model import UoI_Lasso

time_series = np.loadtxt("test_timeseries.csv",
                 delimiter=",").reshape(1200,100)


comm = MPI.COMM_WORLD
rank = comm.rank

uoi_lasso = UoI_Lasso(comm=comm)

uoi_lasso.copy_X = True
uoi_lasso.estimation_target = None
uoi_lasso.logger = None
uoi_lasso.warm_start = False
uoi_lasso.random_state = 21

# +
#for target_idx in range(time_series.shape[1]): 

import time
start_time = time.time()

uoi_conn_mat = np.zeros((100,100))

for target_idx in range(time_series.shape[1]): 

    Y = np.array(time_series[:,target_idx])
    x = np.delete(time_series, target_idx, axis=1) 
    
    uoi_lasso.fit(x , Y)
    uoi_conn_mat[target_idx,:] = np.insert(uoi_lasso.coef_, target_idx, 0) 
    
    print(r2_score(Y, uoi_lasso.predict(x) )) 
    
print(time.time() - start_time, ' seconds') 

# +
n_rois=100
yeo_networks=7
resolution_mm=1
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,yeo_networks,resolution_mm)

plotting.plot_matrix(
    uoi_conn_mat,
    figure=(10, 8),
    labels=schaefer['labels'],
    vmax=0.8,
    vmin=-0.8,
    reorder=False,
)

# +
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind="correlation")
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
# -

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=schaefer['labels'],
    vmax=0.8,
    vmin=-0.8,
    reorder=False,
)


