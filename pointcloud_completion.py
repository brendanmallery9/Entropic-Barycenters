import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
from scipy import stats, linalg
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fsolve, least_squares, minimize, curve_fit
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
import cvxpy as cp
from sympy import symbols, Eq, solve
from os.path import join
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import h5py
import tensorflow as tf
current_dir = os.getcwd()
# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(current_dir))
from EOT_tools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed = 1

#Data processing

#Each block converts a list of pointclouds, indexed by ''label_list'', to ''labeled_data'' objects. 
#The pointcloud corruptions are (in order): dropout_local_1, dropout_local_2, dropout_global_4, dropout_jitter_4, and add_global_4
#The processed ''clean'' pointclodus are stored in ''batch_list''. The corrupted pointclouds are stored in ''corrupted_batch_list''.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_obj_dict={0:'plane',2:'bed', 17:'guitar', 22:'television',37:'vases'}
label_list=[0,2,17,22,37]

with h5py.File('classification_example/pointcloud-c/clean.h5', 'r') as f:
    clean_data_file = f['data'] 
    clean_label_file=f['label']
    clean_dataset = clean_data_file[:] 
    all_labels=clean_label_file[:]
reduced_data_indices=[]
reduced_data_labels=[]
for j in label_list:
    A=find_indices(all_labels,j)
    reduced_data_indices.append(A)
    reduced_data_labels.append(np.dot(np.ones(len(A)),j))
reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(clean_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
batch_list=np.transpose(reduced_labeled_data)


class labeled_data():
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

def map_labels(labels,dictionary):
    new_labels=[]
    for i in labels:
        new_labels.append(dictionary[int(i[0])])
    return np.array(new_labels)

with h5py.File('classification_example/pointcloud-c/dropout_local_1.h5', 'r') as f:
    corrupted_data_file = f['data'] 
    corrupted_label_file=f['label']
    corrupted_dataset = corrupted_data_file[:] 
    all_labels=corrupted_label_file[:]
reduced_data_indices=[]
reduced_data_labels=[]
for j in label_list:
    A=find_indices(all_labels,j)
    reduced_data_indices.append(A)
    reduced_data_labels.append(np.dot(np.ones(len(A)),j))

reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(corrupted_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
dropout_local_1_batches=np.transpose(reduced_labeled_data)

with h5py.File('classification_example/pointcloud-c/dropout_local_4.h5', 'r') as f:
    corrupted_data_file = f['data'] 
    corrupted_label_file=f['label']
    corrupted_dataset = corrupted_data_file[:] 
    all_labels=corrupted_label_file[:]
reduced_data_indices=[]
reduced_data_labels=[]
for j in label_list:
    A=find_indices(all_labels,j)
    reduced_data_indices.append(A)
    reduced_data_labels.append(np.dot(np.ones(len(A)),j))

reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(corrupted_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
dropout_local_4_batches=np.transpose(reduced_labeled_data)

with h5py.File('classification_example/pointcloud-c/dropout_global_4.h5', 'r') as f:
    corrupted_data_file = f['data'] 
    corrupted_label_file=f['label']
    corrupted_dataset = corrupted_data_file[:] 
    all_labels=corrupted_label_file[:]
reduced_data_indices=[]
reduced_data_labels=[]
for j in label_list:
    A=find_indices(all_labels,j)
    reduced_data_indices.append(A)
    reduced_data_labels.append(np.dot(np.ones(len(A)),j))
reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(corrupted_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
dropout_global_4_batches=np.transpose(reduced_labeled_data)


#Point cloud completion tool
#Inputs:
#source: corrupted data, stored as ''measure'' object 
#references: list of ''measure'' objects
#method: 'Entropy' or 'Sinkhorn'
#dim: dimension (3)
#support_size: support size of completed pointcloud (integer)

def completion(source,references,regularization,method,dim,support_size):
    if method=='Entropy':
        analysis_method=ent_analysis
        synthesis_method=entropic_synthesis
    elif method=='Sinkhorn':
        analysis_method=sink_analysis
        synthesis_method=sinkhorn_synthesis
    source_points=source.points
    reference_points=[]
    for i in references:
        reference_points.append(i.points)
    coefficients=analysis_method(reference_points,source_points,regularization)
    iterations=60
    stepsize=0.08
    uniform_points = np.random.rand(support_size, dim)
    uniform_masses=np.dot(np.ones(support_size),1/support_size)
    base_measure=measure(uniform_points,uniform_masses)
    completed_pointcloud=synthesis_method(regularization,references,base_measure,coefficients,stepsize,iterations)
    return completed_pointcloud

planes=np.transpose(batch_list)[0]
dropout_local_4_planes=np.transpose(dropout_local_4_batches)[0]
dropout_global_4_planes=np.transpose(dropout_global_4_batches)[0]
plane_measures=[]
dropout_local_4_plane_measures=[]
dropout_global_4_plane_measures=[]
for i in np.arange(len(planes)):
    plane_no=np.shape(planes[i].data)[0]
    dlocal_no=np.shape(dropout_local_4_planes[i].data)[0]
    dglobal_no=np.shape(dropout_global_4_planes[i].data)[0]
    plane_masses=np.dot(np.ones(plane_no),1/plane_no)
    plane_measures.append(measure(planes[i].data,plane_masses))
    dlocal_masses=np.dot(np.ones(dlocal_no),1/dlocal_no)
    dropout_local_4_plane_measures.append(measure(dropout_local_4_planes[i].data,dlocal_masses))
    dglobal_masses=np.dot(np.ones(dglobal_no),1/dglobal_no)
    dropout_global_4_plane_measures.append(measure(dropout_global_4_planes[i].data,dglobal_masses))

references=plane_measures[0:5]

def compute_and_plot_completion(uncorrupted_data_array,corrupted_data_array,index,method,reference_measures,regularization,dim,support_size):
    recon=completion(corrupted_data_array[index],reference_measures,regularization,'{}'.format(method),dim,support_size)
    A=recon.points
    A_mass=np.dot(np.ones(len(A)),1/len(A))
    original = uncorrupted_data_array[index].points
    original_mass=np.dot(np.ones(len(original)),1/len(original))
    M=euclidean_distances(A,original)
    OT_cost=ot.emd2(A_mass,original_mass,M)
    x_min, x_max = A[:, 0].min(), A[:, 0].max()
    y_min, y_max = A[:, 1].min(), A[:, 1].max()
    z_min, z_max = A[:, 2].min(), A[:, 2].max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=5, c='blue', alpha=0.8)
    ax.set_title(f'Reconstructed Pointcloud with Local Dropout \n ({method}, $\\epsilon={regularization}$, $OT_2$-cost={OT_cost:.4f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # Show the plot
    stoptime=time.time()
    plt.savefig(f"completion_{method}_reg{regularization}_index{stoptime}.png")
    plt.close(fig)

compute_and_plot_completion(plane_measures,dropout_global_4_plane_measures,12,'Entropy',references,0.003,3,1024)
compute_and_plot_completion(plane_measures,dropout_global_4_plane_measures,12,'Sinkhorn',references,0.003,3,1024)
compute_and_plot_completion(plane_measures,dropout_local_4_plane_measures,12,'Entropy',references,0.003,3,1024)
compute_and_plot_completion(plane_measures,dropout_local_4_plane_measures,12,'Sinkhorn',references,0.003,3,1024)

