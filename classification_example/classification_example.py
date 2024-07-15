import os
import sys
import numpy as np
import random
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
from pointnet_implementation import *
from kscore.estimators import *
from kscore.kernels import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EOT_tools import *


random.seed = 1

#Data processing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_obj_dict={0:'plane',2:'bed', 17:'guitar', 22:'television',37:'vases'}
label_list=[0,2,17,22,37]

ordered_labels={0:0,2:1,17:2,22:3,37:4}
invert_ordered={0:0,1:2,2:17,3:22,4:37}


pointnet = PointNet()
pointnet.to(device);
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0002)

class entry:
    def __init__(self,labels,data,coefficients):
        self.labels=labels
        self.data=data
        self.coefficients=coefficients


def get_labels(obj):
    return obj.labels

def split(input_list, size1, size2):
    if size1 + size2 != len(input_list):
        raise ValueError("Sizes do not match the length of the input list")
    
    shuffled_list = input_list.copy()
    permutation = list(range(len(shuffled_list)))
    random.shuffle(permutation)
    shuffled_list = [shuffled_list[i] for i in permutation]
    
    list1 = shuffled_list[:size1]
    list2 = shuffled_list[size1:]
    
    return list1, list2, permutation

def apply_permutation(input_list, permutation):
    if len(input_list) != len(permutation):
        raise ValueError("Input list and permutation must be of the same length")
    
    return [input_list[i] for i in permutation]


with h5py.File('pointcloud-c/clean.h5', 'r') as f:
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



pointnet = PointNet()
pointnet.to(device);
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0002)


def train(model,batchlist, val_loader=None,  epochs=4):
    for epoch in range(epochs): 
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(batchlist, 0):
            shuffled_batch=np.random.permutation(batchlist[i])
            point_list=[]
            label_list=[]
            for pointcloud in shuffled_batch:
                point_list.append(pointcloud.data)
                label_list.append(pointcloud.labels)
            point_list=np.array(point_list)
            point_list=np.transpose(point_list,(0,2,1))
            label_list=np.array(label_list)
            label_list=map_labels(label_list,ordered_labels)
            inputs, labels = torch.from_numpy(point_list).to(device).float(), torch.from_numpy(label_list).to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs)
            loss = pointnetloss(outputs, labels.squeeze(), m3x3, m64x64)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:   
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(batchlist), running_loss / 10))
                running_loss = 0.0
        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)
            
device = torch.device("cpu")

with h5py.File('pointcloud-c/dropout_local_1.h5', 'r') as f:
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

with h5py.File('pointcloud-c/dropout_local_2.h5', 'r') as f:
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
dropout_local_2_batches=np.transpose(reduced_labeled_data)

with h5py.File('pointcloud-c/dropout_global_4.h5', 'r') as f:
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
dropout_global_batches=np.transpose(reduced_labeled_data)

with h5py.File('pointcloud-c/jitter_4.h5', 'r') as f:
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
np.shape(reduced_data_indices)
reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(corrupted_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
jitter_batches=np.transpose(reduced_labeled_data)

with h5py.File('pointcloud-c/add_global_4.h5', 'r') as f:
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
np.shape(reduced_data_indices)
reduced_labeled_data=[]
for i in reduced_data_indices:
    A=[]
    for j in i:
        A.append(labeled_data(corrupted_dataset[j],all_labels[j]))
    reduced_labeled_data.append(np.array(A))
reduced_labeled_data=np.array(reduced_labeled_data)
add_global_batches=np.transpose(reduced_labeled_data)

with h5py.File('pointcloud-c/add_local_4.h5', 'r') as f:
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
add_local_batches=np.transpose(reduced_labeled_data)
corrupted_batchlist=np.array([dropout_local_1_batches,dropout_local_2_batches,dropout_global_batches,jitter_batches,add_local_batches])
corrupted_batchlist = corrupted_batchlist.transpose((0, 2, 1))

def compute_accuracy(pred_labels, true_labels):
    assert len(pred_labels) == len(true_labels), "Length of predicted labels and true labels must be the same."

    correct = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
    total = len(pred_labels)
    accuracy = (correct / total) * 100
    return accuracy

def score_results(list_of_entr,atoms):
    maxes=[]
    coeff_list=[]
    entry_label_list=[]
    dictionary_labels=[]
    for entr in list_of_entr:
        coeff_list.append(entr.coefficients)
        entry_label_list.append(entr.labels)
    for i in atoms:
        dictionary_labels.append(i.labels)
    dictionary_labels=np.array(dictionary_labels).flatten()
    dictionary_labels=list(set(dictionary_labels))
    dictionary_labels.sort()
    
    for coeff in coeff_list:
        maximum=np.argmax(coeff)
        maxes.append(dictionary_labels[maximum])
    maxes=np.array(maxes)
    entry_label_list=np.array(entry_label_list).flatten()
    score_vec=(maxes != entry_label_list).astype(int)
    return score_vec

def reset_weights(model):
    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.normal_(m.weight, mean=0, std=0.1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    model.apply(weights_init)

######
#Classification experiment using Sinkhorn, entropy-regularized and unregularized functionals


def benchmark_trial(data,corrupted_batches,m,regularization,inner_trials,functional):
    data=np.transpose(data)
    train_data=[]
    test_data=[]
    for i in data:
        i_train,i_test,permutation=split(i,80,20)
        train_data.append(i_train)
        test_data.append(i_test)
    train_data=train_data
    test_data=test_data
    train_data_points=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        train_data_points.append(type_bin)
    train_data_labels=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        train_data_labels.append(type_bin)
    test_data_points=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        test_data_points.append(type_bin)
    test_data_labels=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        test_data_labels.append(type_bin)
    corrupted_data=corrupted_batches
    corrupted_test_data=[]
    for corr_batch in corrupted_data:
        corr_test_set=[]
        for cloud_type in corr_batch:
            test_batch=[]
            shuffled=apply_permutation(cloud_type,permutation)
            i_test=shuffled[80:]
            corr_test_set.append(i_test)
        corrupted_test_data.append(corr_test_set)
    corrupted_test_data.append(test_data)
    corrupted_test_data_copy=corrupted_test_data.copy()
    corrupted_test_data_copy=np.array(corrupted_test_data_copy).transpose(0,2,1)
    corrupted_test_data=np.reshape(corrupted_test_data,(6,100))
  #Wass learning
    big_wass_learning_vec=[]
    for trial in np.arange(inner_trials):
        test_entry_list=[]
        for corr_batch in corrupted_test_data:
            corr_test_entry_list=[]
            for pointcloud in corr_batch:
                points=pointcloud.data
                label=pointcloud.labels
                coefs=0
                new_entry=entry(label,points,coefs)
                corr_test_entry_list.append(new_entry)
            test_entry_list.append(corr_test_entry_list)
        references=[]
        for i in train_data:
            sampled_ref=random.sample(list(i),k=m)
            references.append(sampled_ref)
        reference_list=list(np.array(references).flatten())
        sorted_references = sorted(reference_list, key=get_labels)
        small_wass_learning_vec=[]
        counter=0
        for i in corrupted_test_data:
            counter=counter+1
            #Sinkhorn functional
            if functional=='Sinkhorn':
                learned_entries=sink_dictionary_learn(i,sorted_references,regularization,m)

            #Unregularized functional/Tangential Wasserstein projection
            elif functional=='Unregularized':
                learned_entries=tangent_dictionary_learn(i,sorted_references,m)  

            #Entropy-regularized functional
            elif functional=='Entropy':
                learned_entries=dictionary_learn(i,sorted_references,regularization,m)

            wass_learning_results=1-np.mean(score_results(list(learned_entries),sorted_references))
            small_wass_learning_vec.append(wass_learning_results)
        big_wass_learning_vec.append(small_wass_learning_vec)
        print(trial)
    #NN
    reset_weights(pointnet)
    train(pointnet,np.transpose(train_data))
    pointnet.eval()
    all_preds = []
    all_labels = []
    nn_result_vec=[]
    with torch.no_grad():
        for corr_batch in corrupted_test_data_copy:
            for i, data in enumerate(corr_batch):
                #print(i,data)
                print('Batch [%4d / %4d]' % (i+1, len(corr_batch)))
                shuffled_batch=np.random.permutation(data)
                point_list=[]
                label_list=[]
                for pointcloud in shuffled_batch:
                    point_list.append(pointcloud.data)
                    label_list.append(pointcloud.labels)
                label_list=map_labels(label_list,ordered_labels)
                inputs, labels = torch.from_numpy(np.array(point_list)).to(device).float(), torch.from_numpy(np.array(label_list)).to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.numpy())
                all_labels += list(labels.numpy())
            nn_results=compute_accuracy(all_preds,all_labels)
            nn_result_vec.append(nn_results)
    return big_wass_learning_vec,nn_result_vec,train_data_points,test_data_points,train_data_labels,test_data_labels

def benchmark_experiment(inner_trials,outer_trials,functional):
    wass_learning_vec=[]
    nn_learning_vec=[]
    for i in np.arange(outer_trials):
        A,B,train_points,test_points,train_labels,test_labels=benchmark_trial(batch_list,corrupted_batchlist,3,0.089,inner_trials,functional)
        wass_learning_vec.append(A)
        nn_learning_vec.append(B)
        print(i)
    return wass_learning_vec,nn_learning_vec




###########
#Classification experiment using the doubly regularized functional
#Score estimation relies on supplemental code to ``Nonparametric Score Estimation'' by Yuhao Zhou, Jiaxin Shi, Jun Zhu. https://github.com/miskcoo/kscore

def doubly_reg_benchmark_trial(data,corrupted_batches,m,inner_regularization,outer_regularization,inner_trials):
    data=np.transpose(data)
    train_data=[]
    test_data=[]
    for i in data:
        i_train,i_test,permutation=split(i,80,20)
        train_data.append(i_train)
        test_data.append(i_test)
    train_data=train_data
    test_data=test_data
    train_data_points=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        train_data_points.append(type_bin)
    train_data_labels=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        train_data_labels.append(type_bin)
    test_data_points=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        test_data_points.append(type_bin)
    test_data_labels=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        test_data_labels.append(type_bin)
    corrupted_data=corrupted_batches
    corrupted_test_data=[]
    for corr_batch in corrupted_data:
        corr_test_set=[]
        for cloud_type in corr_batch:
            test_batch=[]
            shuffled=apply_permutation(cloud_type,permutation)
            i_test=shuffled[80:]
            corr_test_set.append(i_test)
        corrupted_test_data.append(corr_test_set)
    corrupted_test_data.append(test_data)
    corrupted_test_data_copy=corrupted_test_data.copy()
    corrupted_test_data_copy=np.array(corrupted_test_data_copy).transpose(0,2,1)
    corrupted_test_data=np.reshape(corrupted_test_data,(6,100))
  #Wass learning
    coefficient_vec=[]
    big_wass_learning_vec=[]
    for trial in np.arange(inner_trials):
        test_entry_list=[]
        for corr_batch in corrupted_test_data:
            corr_test_entry_list=[]
            for pointcloud in corr_batch:
                points=pointcloud.data
                label=pointcloud.labels
                coefs=0
                new_entry=entry(label,points,coefs)
                corr_test_entry_list.append(new_entry)
            test_entry_list.append(corr_test_entry_list)
        references=[]
        for i in train_data:
            sampled_ref=random.sample(list(i),k=m)
            references.append(sampled_ref)
        reference_list=list(np.array(references).flatten())
        sorted_references = sorted(reference_list, key=get_labels)
        small_wass_learning_vec=[]
        for i in corrupted_test_data:
            noise_learned_entries=doubly_reg_noise_dictionary_learn(i,sorted_references,inner_regularization,outer_regularization,m)
            coefficient_vec.append(noise_learned_entries)
            noise_learning_results=1-np.mean(score_results(list(noise_learned_entries),sorted_references))
            print(noise_learning_results)
            small_wass_learning_vec.append(noise_learning_results)
        big_wass_learning_vec.append(small_wass_learning_vec)
        print(trial)
    #NN
    reset_weights(pointnet)
    train(pointnet,np.transpose(train_data))
    pointnet.eval()
    all_preds = []
    all_labels = []
    nn_result_vec=[]
    with torch.no_grad():
        for corr_batch in corrupted_test_data_copy:
            for i, data in enumerate(corr_batch):
                #print(i,data)
                print('Batch [%4d / %4d]' % (i+1, len(corr_batch)))
                shuffled_batch=np.random.permutation(data)
                point_list=[]
                label_list=[]
                for pointcloud in shuffled_batch:
                    point_list.append(pointcloud.data)
                    label_list.append(pointcloud.labels)
                label_list=map_labels(label_list,ordered_labels)
                inputs, labels = torch.from_numpy(np.array(point_list)).to(device).float(), torch.from_numpy(np.array(label_list)).to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.numpy())
                all_labels += list(labels.numpy())
            nn_results=compute_accuracy(all_preds,all_labels)
            nn_result_vec.append(nn_results)
    return big_wass_learning_vec,nn_result_vec,train_data_points,test_data_points,train_data_labels,test_data_labels,coefficient_vec

def doubly_benchmark_experiment(inner_trials,outer_trials):
    wass_learning_vec=[]
    nn_learning_vec=[]
    for i in np.arange(outer_trials):
        A,B,train_points,test_points,train_labels,test_labels,coefficient_vec=doubly_reg_benchmark_trial(batch_list,corrupted_batchlist,3,.009,0.01,inner_trials)
        wass_learning_vec.append(A)
        nn_learning_vec.append(B)
        print(i)
    return wass_learning_vec,nn_learning_vec




#Script

sinkhorn_vec,nn_vec=benchmark_experiment(3,10,'Sinkhorn')
np.save('sinkhorn_learn_vec',sinkhorn_vec)
np.save('nn_learn_vec',nn_vec)

entropy_vec,nn_vec=benchmark_experiment(3,10,'Entropy')
np.save('entropy_learn_vec',entropy_vec)

unreg_vec,nn_vec=benchmark_experiment(3,10,'Unregularized')
np.save('unreg_learn_vec',unreg_vec)

dreg_vec,nn_vec=doubly_benchmark_experiment(3,10)
np.save('dreg_learn_vec',dreg_vec)


def means_and_confidence_intervals(vector_array,confidence_level):
    inner_means=[]
    for i in vector_array:
        inner_means.append(np.mean(i,axis=0))
    inner_means=np.array(inner_means)
    errors=[]
    outer_means=[]
    all_data=[inner_means[:,0],inner_means[:,1],inner_means[:,2],inner_means[:,3],inner_means[:,4],inner_means[:,5]]
    for i in all_data:
        t_statistic, p_value = stats.ttest_1samp(i, np.mean(i))
        sample_mean = np.mean(i)
        outer_means.append(sample_mean)
        sample_sem = stats.sem(i)
        degrees_of_freedom = len(i) - 1
        confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom, sample_mean, sample_sem)
        errors.append((confidence_interval[1]-confidence_interval[0])/2)
    return outer_means,errors


alpha=0.95
sinkhorn_means,sinkhorn_errors=means_and_confidence_intervals(sinkhorn_vec,alpha)
entropy_means,entropy_errors=means_and_confidence_intervals(entropy_vec,alpha)
unreg_means,unreg_errors=means_and_confidence_intervals(unreg_vec,alpha)
dreg_means,dreg_errors=means_and_confidence_intervals(dreg_vec,alpha)
nn_means,nn_errors=means_and_confidence_intervals(nn_vec,alpha)


