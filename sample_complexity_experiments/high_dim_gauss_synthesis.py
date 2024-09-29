import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import h5py
import pandas as pd
from scipy import stats,linalg
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import ot
import random
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
from scipy.optimize import minimize
import cvxpy as cp 
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
from os.path  import join
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import plotly.express as px
import h5py
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import os
import time
#////////

class measure:
    def __init__(self,points,masses):
        self.points=points
        self.masses=masses


def empirical_measure(input_measure,samples):
    points=input_measure.points
    masses=input_measure.masses
    uniform_mass=np.dot(np.ones(samples),1/samples)
    empirical_support=[]
    labels = np.random.choice(len(masses), size=samples, p=masses)
    for i in labels:
        empirical_support.append(points[i])
    output=measure(empirical_support,uniform_mass)
    return output


def sample_with_replacement(lst, sample_size):
    sample = [random.choice(lst) for _ in range(sample_size)]
    return sample

def sample_array(array,samples):
    index_list=np.arange(len(array))
    sampled_idx=np.random.choice(index_list,size=samples)
    sample_list=[]
    for i in sampled_idx:
        sample_list.append(array[i])
    return sample_list

def onedim_extended_map(potential,source,target,epsilon):
    source_points=source.points
    target_points=target.points
    log_potential=np.transpose(epsilon*np.log(potential))
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    log_potential_matrix=np.tile(log_potential,(len(source_points),1))
    log_coupling=1/epsilon*(log_potential_matrix-cost_matrix)
    gamma=np.exp(log_coupling)
    unnormalized_map=np.dot(gamma,target_points)
    ones=np.ones((len(target_points),1))
    normalization=gamma@ones
    J=np.divide(unnormalized_map,normalization)
    return J


def random_rotation_matrix(n):
    random_matrix = np.random.rand(n, n)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = np.sign(d)
    q *= ph
    return q

def random_weights(size):
    vec=np.random.rand(size)
    total=sum(vec)
    normalized=np.dot(vec,1/total)
    return normalized

def random_vec(size,low,high):
    vec=np.random.uniform(low,high,size)
    mat=random_rotation_matrix(size)
    rot_vec=mat@vec
    return rot_vec

def oneD_barystdv_threerefsolver(weights, stdvs,regularization):
    eps_prime=max((regularization/2)**0.5,0)
    def func_to_solve(x):
        summand1 = weights[0] * (eps_prime**4 + 4 * stdvs[0]**2 * x**2)**0.5
        summand2 = weights[1] * (eps_prime**4 + 4 * stdvs[1]**2 * x**2)**0.5
        summand3 = weights[2] * (eps_prime**4 + 4 * stdvs[2]**2 * x**2)**0.5
        return summand1 + summand2 + summand3 - (eps_prime**2 + 2 * x**2)
    roots = least_squares(func_to_solve,x0=2)
    return roots.x


def oneD_barystdv_solver_tworef(weights, stdvs,regularization):
    eps_prime=max((regularization/2)**0.5,0)
    def func_to_solve(x):
        summand1 = weights[0] * (eps_prime**4 + 4 * stdvs[0]**2 * x**2)**0.5
        summand2 = weights[1] * (eps_prime**4 + 4 * stdvs[1]**2 * x**2)**0.5
        return summand1 + summand2 - (eps_prime**2 + 2 * x**2)
    roots = least_squares(func_to_solve,x0=2)
    return roots.x

def randcov(low,high,dim,alpha):
    A = np.random.uniform(low,high,(dim, dim))
    B = A @ A.T + alpha*np.eye(dim)
    rotation_matrix, _ = np.linalg.qr(np.random.randn(dim, dim))
    composed_matrix = rotation_matrix @ B @ rotation_matrix.T
    return composed_matrix

def get_potential(source,target,epsilon):
    source_points=source.points
    target_points=target.points
    source_masses=source.masses
    target_masses=target.masses
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    J=ot.sinkhorn(source_masses,target_masses,cost_matrix,epsilon,log='True',stopThr=1e-8,numItermax=4000)
    u=J[1].get('u')
    v=J[1].get('v')
    return u,v

def highdim_extended_map(potential,source,target,epsilon,dim):
    source_points=source.points
    target_points=target.points
    log_potential=np.transpose(epsilon*np.log(potential))
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    log_potential_matrix=np.tile(log_potential,(len(source_points),1))
    log_coupling=1/epsilon*(log_potential_matrix-cost_matrix)
    gamma=np.exp(log_coupling)
    unnormalized_map=np.dot(gamma,target_points)
    ones=np.ones((len(target_points),dim))
    normalization=gamma@ones
    J=np.divide(unnormalized_map,normalization)
    return J

def barycenter_functional(source_points,reference_points,weights,epsilon):
    cost_vec=[]
    source_masses=np.dot(1/len(source_points),np.ones(len(source_points)))
    for i in reference_points:
        M=np.power(euclidean_distances(source_points,i),2)
        target_masses=np.dot(1/len(i),np.ones(len(i)))
        cost=ot.sinkhorn2(source_masses,target_masses,M,epsilon)
        cost_vec.append(cost)
    score=np.dot(np.array(weights),np.array(cost_vec))
    return score



#////////////

def highdim_synthesis_experiment_three_refs(regularization,trials,instances,dim,initial_samps):
    dir_name='dim=15_highdim_synthesis_{}'.format(time.time())
    os.mkdir(dir_name)
    #synthesize
    weight_vec=random_weights(3)
    mean_1=random_vec(dim,0.2,0.4)
    mean_2=random_vec(dim,0.4,0.6)
    mean_3=random_vec(dim,0.6,0.8)
    A1=randcov(.2,.5,dim,0.7)
    A2=randcov(.2,.5,dim,0.7)
    A3=randcov(.2,.5,dim,0.7)
    Gauss_1=stats.multivariate_normal(mean_1,A1)
    Gauss_2=stats.multivariate_normal(mean_2,A2)
    Gauss_3=stats.multivariate_normal(mean_3,A3)
    Gauss_1_points=Gauss_1.rvs(initial_samps)
    Gauss_2_points=Gauss_2.rvs(initial_samps)
    Gauss_3_points=Gauss_3.rvs(initial_samps)    
    uniform_points = np.random.rand(initial_samps, dim)
    uniform_masses=np.dot(np.ones(initial_samps),1/initial_samps)
    uniform_measure=measure(uniform_points,uniform_masses)
    measure_location_vec=[np.array(Gauss_1_points),np.array(Gauss_2_points),np.array(Gauss_3_points)]
    measure_masses=[np.array(uniform_masses),np.array(uniform_masses),np.array(uniform_masses)]
    true_barycenter=ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measure_location_vec,measures_weights=measure_masses,X_init=np.array(uniform_measure.points),b=np.array(uniform_measure.masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
    os.mkdir('{}/params'.format(dir_name))
    np.save('{}/params/true_barycenter'.format(dir_name),np.array(true_barycenter))
    np.save('{}/params/mean_vec'.format(dir_name),np.array([mean_1,mean_2,mean_3]))
    np.save('{}/params/cov_vec'.format(dir_name),np.array([A1,A2,A3]))
    np.save('{}/params/ref_points'.format(dir_name),np.array(measure_location_vec))
    true_value=barycenter_functional(true_barycenter,measure_location_vec,weight_vec,regularization)
    np.save('{}/true_value'.format(dir_name),true_value)
    #clear variables
    true_barycenter=[]
    Gauss_1_points=[]
    Gauss_2_points=[]
    Gauss_3_points=[]
    measure_location_vec=[]
    uniform_points = []
    uniform_measure=[]
    for samps in trials:
        os.mkdir('{}/{}'.format(dir_name,samps))
        for j in np.arange(instances):
            Gauss_1_points=Gauss_1.rvs(samps)
            Gauss_2_points=Gauss_2.rvs(samps)
            Gauss_3_points=Gauss_3.rvs(samps)
            j_uniform_points = np.random.rand(samps, dim)
            j_masses=np.dot(np.ones(samps),1/samps)
            j_reference_points=[np.array(Gauss_1_points),np.array(Gauss_2_points),np.array(Gauss_3_points)]
            j_reference_masses=[np.array(j_masses),np.array(j_masses),np.array(j_masses)]
            j_output=ot.bregman.free_support_sinkhorn_barycenter(j_reference_points,j_reference_masses,np.array(j_uniform_points),b=np.array(j_masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
            true_Gauss_1_points=Gauss_1.rvs(initial_samps)
            true_Gauss_2_points=Gauss_2.rvs(initial_samps)
            true_Gauss_3_points=Gauss_3.rvs(initial_samps)   
            measure_location_vec=[np.array(true_Gauss_1_points),np.array(true_Gauss_2_points),np.array(true_Gauss_3_points)]
            j_output_value=barycenter_functional(j_output,measure_location_vec,weight_vec,regularization)
            np.save('{}/{}/{}'.format(dir_name,samps,j),j_output_value)
    return true_value



#Parameters used for experiments in paper:
regularization=1
sample_range=[10,20,40,80,160,320,640,1280,2560,5120,10240]
initial_samps=20000
#sample_range=[10,20]
#initial_samps=100
highdim_synthesis_experiment_three_refs(regularization,sample_range,100,15,initial_samps)

#script to plot results
