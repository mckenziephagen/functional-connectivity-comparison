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
#     display_name: FC
#     language: python
#     name: fc
# ---

# Visualize and compare sparsities for different models. 

import matplotlib

import scipy

# +
import pickle
import os
import os.path as op

from glob import glob

from nilearn import datasets, plotting

import scipy
import numpy as np

import pingouin as pg

import pandas as pd

from itertools import chain

import matplotlib.pyplot as plt
  
import networkx as nx
# -

date_string='2023-11-07'


with open(f'../../results/{date_string}_lasso_dict.pkl', 'rb') as f:
        lasso_dict = pickle.load(f)

with open(f'../../results/{date_string}_uoi_dict.pkl', 'rb') as f:
        uoi_dict = pickle.load(f)

with open('../../results/2023-11-07_pearson_dict.pkl', 'rb') as f:
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
plt.savefig('zeros.png')
# -

# As expected, the lasso matrix is MUCH sparser than the Pearson matrix, and still sparse compared to Lasso. The LASSO edges between 0-.1 are likely false positives according to simulations in one of Kris's papers.

# +
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, width_ratios=[1, 3], layout='compressed')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax1.hist(uoi_median_list, 
         np.linspace(-.2, 1, 100), 
         alpha=.5, 
        label='UofI', color='#00313C')

ax1.hist(lasso_median_list, 
         np.linspace(-.2, 1, 100),
         alpha=.5, 
        label='LASSO', color='#007681')
ax1.hist(pearson_median_list,
        np.linspace(-.2, 1, 100), 
        alpha=.5, 
        label='Pearson', color='#B1B3B3')

ax1.axes.set_ylabel('Number of Edges', fontsize=14)

ax2.hist(lasso_median_list, 
         np.linspace(-.2, 1, 100),
         alpha=.3, 
        label='LASSO', color='#007681')

ax2.hist(uoi_median_list, 
         np.linspace(-.2, 1, 100), 
         alpha=.7, 
        label='UofI', color='#00313C')
# ax2.hist(pearson_median_list,
#         np.linspace(-.2, 1, 100), 
#         alpha=.4, 
#         label='Pearson', color='#B1B3B3')

ax2.xaxis.set_tick_params(labelsize=14)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.set_ylim(0,400)
ax2.set_xlim(-.2,.5)
ax2.legend(prop={'size': 16})
f.set_figwidth(10)
plt.xlabel('Edge Weight', size=14) 
ax2.axes.set_ylabel('Number of Edges', size=14) 


#plt.suptitle('Median Connectivity Values Across Individuals', size=20)
plt.savefig('plots/sparsity_average_minus_pearson.png') 

# +
cmap1 = plt.cm.RdPu
cmap1.set_under('b',1)

cmap2 = plt.cm.Purples
cmap2.set_under('b',1)

h1 = plt.hist2d(pd.Series(lasso_mat.ravel()).dropna(),
                pd.Series(uoi_mat.ravel()).dropna(),
                bins=(100, 50), norm=matplotlib.colors.LogNorm(), 
                cmap = cmap1)

# h2 = plt.hist2d((pearson_dict['sub-793465']['ses-2_run-2'].ravel()), 
#            uoi_dict['sub-793465']['ses-2_run-2'].ravel(), 
#           bins=(50, 50), norm=matplotlib.colors.LogNorm(), 
#               cmap = cmap2)
plt.colorbar(h1[3])


# +
pearson_mat = np.zeros((100,100, 0))

for participant in pearson_dict.keys():
    for ses in pearson_dict[participant].keys():
        temp_mat = pearson_dict[participant][ses]
        np.fill_diagonal(temp_mat, np.nan)
        pearson_mat = np.dstack((pearson_mat, temp_mat)  ) 

# +
lasso_mat = np.zeros((100,100, 0))

for participant in lasso_dict.keys():
    for ses in lasso_dict[participant].keys():
        temp_mat = lasso_dict[participant][ses]
        np.fill_diagonal(temp_mat, np.nan)
        lasso_mat = np.dstack((lasso_mat, temp_mat)  ) 

# +
uoi_mat = np.zeros((100,100, 0))

for participant in uoi_dict.keys():
    for ses in uoi_dict[participant].keys():
        temp_mat = uoi_dict[participant][ses]
        np.fill_diagonal(temp_mat,np.nan)
        uoi_mat = np.dstack((uoi_mat, temp_mat)  ) 
# -
uoi_median_list = np.median(uoi_mat, axis=2).ravel()
lasso_median_list = np.median(lasso_mat, axis=2).ravel()
pearson_median_list = np.median(np.arctanh(pearson_mat), axis=2).ravel()


# +
ax = plotting.plot_matrix(lasso_median_list.reshape(100,100) , 
    vmax=.5,
    vmin=-0.5, 
    reorder=False) 

#ax.set_xticklabels([])
# plotting.plot_matrix(lasso_median_list.reshape(100,100) , 
#     vmax=.5,
#     vmin=-0.5, 
#     reorder=False)
plt.savefig('matrix.png') 
# -

plotting.plot_matrix((pearson_median_list.reshape(100,100)) , 
    vmax=1,
    vmin=-0.8, 
    reorder=False,
    labels=labels)

labels = [(' ').join(str(schaefer['labels'][i]).split('_')[-3:-1])  if (i % 3) == 0 else " " for i in range(len(schaefer['labels']))]

check_list = [] 

for i in range(len(labels)): 
 #   print(labels[i]) 
    if labels[i] in check_list: 
        labels[i] = ''
    check_list.append(labels[i]) 

# +
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

plotting.plot_matrix(
    lasso_median_list.reshape(100,100),
    axes=ax1, 
    vmax=0.4,
    vmin=-0.4,
    labels=labels, 
    colorbar=False)

plotting.plot_matrix(
    uoi_median_list.reshape(100,100),
    axes=ax2, 
    vmax=0.4,
    vmin=-0.4,
    colorbar=True)



ax1.set_xticklabels([]) 
ax1.set_title('LASSO') 
ax2.set_title('Union of Intersections') 
ax3.set_title('Pearson') 
f.show()
plt.savefig('model_medians.png')
# -


# This looks like noise left after subtracting out the lasso adjcaceny matrix - maybe the false pos? 

np.max(pearson_median_list)

# +
plotting.plot_matrix(
    pearson_median_list.reshape(100,100), 
    vmax=.8,
    vmin=-.8,
    label=labels,
    auto_fit=False)

plt.title('Pearson') 
plt.savefig('Pearson_median.png')
# -

plotting.plot_matrix(
    uoi_median_list.reshape(100,100),
    vmax=0.4,
    vmin=-0.4,
    colorbar=False)



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

plt.scatter(pearson_edges_list, uoi_edges_list)

# +
plt.hist(lasso_ratio, color='red', alpha=.75, label = 'LASSO') 
plt.hist(uoi_ratio, alpha=.75, label = 'UoI')
plt.hist(pearson_ratio, alpha=.75, label = 'Pearson')

plt.vlines(np.mean(lasso_ratio), ymin=0, ymax=80, color='black')
plt.vlines(np.mean(uoi_ratio), ymin=0, ymax=80, color='black')
plt.vlines(np.mean(pearson_ratio), ymin=0, ymax=80, color='black')

plt.legend()
plt.title('Ratio of Negative Edges to Positive Edges')
plt.show()
# -

import scipy

scipy.stats.ttest_ind(lasso_edges_list, uoi_edges_list, equal_var=False).confidence_interval() 

print(np.mean(uoi_edges_list))
print(np.std(uoi_edges_list))

print(np.mean(lasso_edges_list)) 
print(np.std(lasso_edges_list)) 

len(lasso_edges_list) 

len(uoi_edges_list) 

# +
bins = np.linspace(0, .7, 50)

plt.hist(uoi_edges_list, bins, alpha=.75, label = 'UoI', color=matplotlib.colors.to_rgba('#00313C')) 
plt.hist(lasso_edges_list, bins, alpha=.75, label = 'LASSO', color=matplotlib.colors.to_rgba('#007681')) 

# plt.vlines(np.mean(lasso_edges_list), ymin=0, ymax=100, color='black')
# plt.vlines(np.mean(uoi_edges_list), ymin=0, ymax=100, color='black')
plt.gca().set_aspect(1/250)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Number of Participants', size=14) 
plt.xlabel('Selection Ratio', size=14) 
plt.savefig('selection_ratio.png', bbox_inches="tight") 

# +
bins = np.linspace(0, .7, 50)

plt.hist(uoi_edges_list, bins, alpha=.75, label = 'UoI', color=matplotlib.colors.to_rgba('#00313C')) 
plt.hist(lasso_edges_list, bins, alpha=.75, label = 'LASSO', color=matplotlib.colors.to_rgba('#007681')) 
plt.hist(pearson_edges_list, alpha=.75, label = 'pearson') 

# plt.vlines(np.mean(lasso_edges_list), ymin=0, ymax=100, color='black')
# plt.vlines(np.mean(uoi_edges_list), ymin=0, ymax=100, color='black')
plt.gca().set_aspect(1/250)
plt.legend()
plt.savefig('selection_ratio.png', bbox_inches="tight") 
# -

plt.hist2d(pearson_edges_list, uoi_edges_list, bins=(100,100) ) 


np.mean(uoi_edges_list)

np.mean(lasso_edges_list)

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
