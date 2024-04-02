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

# To do:
#
# - finger print/ descriminability (see Noble et a., 2021?) 
# - sparsify Pearson for modularity, in/out degree, etc - find common thresholds
# - correlation with movement (see Mahadevan et al 2021)
#     - pull text file from datalad, compile dataframe w/ qc metrics
#     
# - update github
# - compare selection accuracy
# - can we predict better? use code from LASSO project
# - debug runtime of small world networkx calculation
# - refactor code, make functions

# +
import pickle
import os
import os.path as op

from glob import glob

from nilearn import datasets, plotting

from scipy import mean

import numpy as np

import pingouin as pg

import pandas as pd

from itertools import chain

import matplotlib.pyplot as plt
  
import networkx as nx
# -

with open('results/2023-11-07_lasso_dict.pkl', 'rb') as f:
        lasso_dict = pickle.load(f)

with open('results/2023-11-07_uoi_dict.pkl', 'rb') as f:
        uoi_dict = pickle.load(f)

with open('results/2023-11-07_pearson_dict.pkl', 'rb') as f:
        pearson_dict = pickle.load(f)

len(lasso_dict)

len(uoi_dict)

len(pearson_dict)

# ### Let's look at one subject: 

# +
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, width_ratios=[1, 2])
ax1.hist(uoi_dict['sub-793465']['ses-2_run-2'].ravel(), 
         np.linspace(-.2, 1, 100), 
         alpha=.5, 
        label='UofI', color='blue')
ax1.hist(lasso_dict['sub-793465']['ses-2_run-2'].ravel(), 
         np.linspace(-.2, 1, 100),
         alpha=.5, 
        label='LASSO', color='red')
ax1.hist(np.arctanh(pearson_dict['sub-793465']['ses-2_run-2'].ravel()),
        np.linspace(-.2, 1, 100), 
        alpha=.5, 
        label='Pearson', color='grey')


ax2.hist(lasso_dict['sub-793465']['ses-2_run-2'].ravel(), 
         np.linspace(-.2, 1, 100),
         alpha=.5, 
        label='LASSO', color='red')
ax2.hist(np.arctanh(pearson_dict['sub-793465']['ses-2_run-2'].ravel()),
        np.linspace(-.2, 1, 100), 
        alpha=.25, 
        label='Pearson', color='grey')

ax2.hist(uoi_dict['sub-793465']['ses-2_run-2'].ravel(), 
         np.linspace(-.2, 1, 100), 
         alpha=.5, 
        label='UofI', color='blue')
ax2.set_ylim(0,500)
ax2.legend()
f.set_figwidth(15)
#the log scale looked weird
# -

# As expected, the lasso matrix is MUCH sparser than the Pearson matrix, and still sparse compared to Lasso. The LASSO edges between 0-.1 are likely false positives.
#
# @Kris: is there someway to validate that they're false pos, aside from pointing to simulation data? 

lasso_list = []
pearson_list = []
uoi_list = []

for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        [pearson_list.extend(i) for i in pearson_dict[outer][inner]]

for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        [lasso_list.extend(i) for i in lasso_dict[outer][inner]]

for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        [uoi_list.extend(i) for i in uoi_dict[outer][inner]]

# +
_, axes = plt.subplots(1, 3, figsize=(15, 5))

n_rois=100
yeo_networks=7
resolution_mm=1
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois,yeo_networks,resolution_mm)

plotting.plot_matrix(
    np.arctanh(pearson_dict['sub-793465']['ses-1_run-1']),
    axes=axes.flatten()[0], 
    vmax=0.8,
    vmin=-0.8,
    auto_fit=True)

plotting.plot_matrix(
    lasso_dict['sub-793465']['ses-1_run-1'],
    axes=axes.flatten()[1], 
    vmax=0.4,
    vmin=-0.4)

plotting.plot_matrix(
    uoi_dict['sub-793465']['ses-1_run-1'],
    axes=axes.flatten()[2], 
    vmax=0.4,
    vmin=-0.4)


# -

plotting.plot_matrix(
    lasso_dict['sub-204521']['ses-1_run-1']- uoi_dict['sub-204521']['ses-1_run-1'],
    vmax=0.4,
    vmin=-0.4)

labels = [schaefer['labels'][i] if (i % 3) == 0 else " " for i in range(len(schaefer['labels']))]

# +
_, axes = plt.subplots(1, 3, figsize=(15, 15))

plotting.plot_matrix(
    np.arctanh(pearson_dict['sub-680250']['ses-1_run-1']),
    axes=axes.flatten()[0], 
    vmax=0.8,
    vmin=-0.8,
    labels = labels,
    auto_fit=True)

plotting.plot_matrix(
    lasso_dict['sub-680250']['ses-1_run-1'],
    axes=axes.flatten()[1], 
    vmax=0.4,
    vmin=-0.4)

plotting.plot_matrix(
    uoi_dict['sub-680250']['ses-1_run-1'],
    axes=axes.flatten()[2], 
    vmax=0.4,
    vmin=-0.4)

# -

plotting.plot_matrix(
     pearson_dict['sub-680250']['ses-1_run-1'],
    vmax=0.8,
    vmin=-0.8,
    labels = labels,
    auto_fit=True)

# This looks like noise left after subtracting out the lasso adjcaceny matrix - maybe the false pos? 

plotting.plot_matrix(
    uoi_dict['sub-680250']['ses-1_run-1'] -lasso_dict['sub-680250']['ses-1_run-1'],
    vmax=0.4,
    vmin=-0.4)

# +
# #standardizations? 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(lasso_dict['sub-680250']['ses-1_run-1'].ravel(), 
                uoi_dict['sub-680250']['ses-1_run-1'].ravel(), alpha=.05)
axes[0].axline((0, 0), slope=1)
axes[0].set_ylim(-.4, 1)
axes[0].set_xlim(-.4, 1)
axes[0].set_ylabel('UoI Edge Weights')
axes[0].set_xlabel('Lasso Edge Weights')
axes[0].scatter(lasso_dict['sub-204521']['ses-1_run-1'].ravel(), 
                uoi_dict['sub-204521']['ses-1_run-1'].ravel(), alpha=.05)

axes[1].scatter(np.arctanh(pearson_dict['sub-680250']['ses-1_run-1'].ravel()),
                uoi_dict['sub-680250']['ses-1_run-1'].ravel(), alpha=.25)

axes[1].axline((0, 0), slope=1)
axes[1].set_ylim(-.4, 1)
axes[1].set_xlim(-.4, 1.25)
axes[1].set_ylabel('UoI Edge Weights')
axes[1].set_xlabel('Pearson Edge Weights')

axes[2].scatter(np.arctanh(pearson_dict['sub-680250']['ses-1_run-1'].ravel()), 
                lasso_dict['sub-680250']['ses-1_run-1'].ravel(), alpha=.25)
axes[2].axline((0, 0), slope=1)
axes[2].set_ylim(-.4, 1)
axes[2].set_xlim(-.4, 1.25)
axes[2].set_ylabel('Lasso Edge Weights')
axes[2].set_xlabel('Pearson Edge Weights')

fig.suptitle('Comparison of Edge Weights by Method for Example Subject')
# -



# ### Networks: 

# +
lasso_edges_list = []
lasso_ratio = []

for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        lasso_edges_list.append(sum(sum(lasso_dict[outer][inner] != 0) )/ 10000 ) 
        lasso_ratio.append(sum(sum(lasso_dict[outer][inner] < 0)) / sum(sum(lasso_dict[outer][inner] > 0 )))
# -

uoi_edges_list = []
uoi_ratio = []
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        uoi_edges_list.append(sum(sum(uoi_dict[outer][inner] != 0) ) / 10000 ) 
        
        uoi_ratio.append(sum(sum(uoi_dict[outer][inner] < 0)) / sum(sum(uoi_dict[outer][inner] > 0 )))

pearson_edges_list = []
pearson_ratio = []
for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        pearson_edges_list.append(sum(sum(pearson_dict[outer][inner] != 0) ) / 10000 ) 
        
        pearson_ratio.append(sum(sum(pearson_dict[outer][inner] < 0)) / sum(sum(pearson_dict[outer][inner] > 0 )))

# +
plt.hist(lasso_ratio, color='red', alpha=.75, label = 'LASSO') 
plt.hist(uoi_ratio, alpha=.75, label = 'UoI')
plt.hist(pearson_ratio, alpha=.75, label = 'Pearson')

plt.vlines(np.mean(lasso_ratio), ymin=0, ymax=80, color='black')
plt.vlines(np.mean(uoi_ratio), ymin=0, ymax=80, color='black')
plt.vlines(np.mean(pearson_ratio), ymin=0, ymax=80, color='black')

plt.legend()
plt.title('Ratio of Negative Edges to Positive Edges')
# -

plt.hist(lasso_edges_list, color='red', alpha=.75, label = 'LASSO') 
plt.hist(uoi_edges_list, label = 'UoI')
#this will be more informative with more subjects
plt.legend()
txt="Selection Ratio"
plt.title(txt)

plt.scatter(lasso_edges_list, uoi_edges_list) 
plt.axline((0, 0), slope=1)
plt.ylim(0,1)
plt.xlim(0,1)
#less variability across subjects

mean_out_lasso = list()
mod_lasso = list()
for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        graph = nx.DiGraph(lasso_dict[outer][inner])
        mean_out_lasso.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_lasso.append(nx.community.modularity(graph, c))

mean_out_uoi = list()
mod_uoi = list()
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        graph = nx.DiGraph(uoi_dict[outer][inner])
        mean_out_uoi.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_uoi.append(nx.community.modularity(graph, c))

mean_out_uoi = list()
mod_uoi = list()
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        graph = nx.DiGraph(uoi_dict[outer][inner])
        mean_out_uoi.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_uoi.append(nx.community.modularity(graph, c))

mean_out_pearson = list()
mod_pearson = list()
for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        graph = nx.DiGraph(pearson_dict[outer][inner])
        mean_out_pearson.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_pearson.append(nx.community.modularity(graph, c))

# +
plt.hist(mean_out_lasso, color='red', label='LASSO')
plt.hist(mean_out_uoi, label = 'UoI')

plt.legend()
plt.title('Average Out Degree Across Nodes') 
plt.ylabel('Count')
plt.xlabel('Out Degree')

# +
plt.hist(mod_lasso, color='red', label='LASSO')
plt.hist(mod_uoi, label='UoI')

plt.legend()
plt.title('Modularity Index') 

# +
#threshold pearson and do all of these gain
