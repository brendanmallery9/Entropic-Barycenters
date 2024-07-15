import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats,linalg
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import ot
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
from scipy.optimize import minimize
import cvxpy as cp 
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import random
from os.path  import join

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EOT_tools import *

regularization=1
initial_samps=100
sample_range=[10,20]


def uniform_sample_box(dimension, diameter, num_samples,center):
    samples = np.random.rand(num_samples, dimension)-0.5
    samples *= diameter
    samples=samples+center
    return samples

def unif_highdim_synthesis_iteration(regularization,diameter,dim,samps,base_measure,means,weight_vec,initial_measure):
    Unif_1=uniform_sample_box(dim,diameter,samps,means[0])
    Unif_2=uniform_sample_box(dim,diameter,samps,means[1])
    Unif_3=uniform_sample_box(dim,diameter,samps,means[2])

    measure_location_vec=[np.array(Unif_1),np.array(Unif_2),np.array(Unif_3)]
    uniform_mass_reference=np.dot(np.ones(samps),1/samps)
    measure_masses=[np.array(uniform_mass_reference),np.array(uniform_mass_reference),np.array(uniform_mass_reference)]
    initial_mass=np.array(initial_measure.masses)
    output=ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measure_location_vec,measures_weights=measure_masses,X_init=np.array(initial_measure.points),b=np.array(initial_measure.masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
    cost_matrix=1/2*np.power(euclidean_distances(output,base_measure),2)
    print(np.shape(cost_matrix))
    print(np.shape(uniform_mass_reference),np.shape(initial_measure.masses))
    ot_cost=ot.emd2(initial_mass,initial_mass,cost_matrix)**0.5
    instance_parameters={'ot_cost':ot_cost, 'output':output, 'base_measure':base_measure,'means':means,'weights':weight_vec}
    return instance_parameters

def unif_highdim_synthesis(sample_range,mean_range,diameter,iterations,regularization,dim):
    for i in np.arange(iterations):
            weight_vec=random_weights(3)
            mean_1=random_vec(dim,0.2,0.4)
            mean_2=random_vec(dim,0.4,0.6)
            mean_3=random_vec(dim,0.6,0.8)
            means=[mean_1,mean_2,mean_3]
            Unif_1=uniform_sample_box(dim,diameter,10000,means[0])
            Unif_2=uniform_sample_box(dim,diameter,10000,means[1])
            Unif_3=uniform_sample_box(dim,diameter,10000,means[2])
            uniform_points = np.random.rand(10000, dim)
            uniform_masses=np.dot(np.ones(10000),1/10000)
            measure_masses=[np.array(uniform_masses),np.array(uniform_masses),np.array(uniform_masses)]
            initial_measure=measure(uniform_points,uniform_masses)
            measure_location_vec=[np.array(Unif_1),np.array(Unif_2),np.array(Unif_3)]
            base_measure=ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measure_location_vec,measures_weights=measure_masses,X_init=np.array(initial_measure.points),b=np.array(initial_measure.masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
            for samps in sample_range:
                instance_output=unif_highdim_synthesis_iteration(regularization,diameter,dim,samps,base_measure,means,weight_vec,initial_measure)
                np.save('uniform_synthesis{}'.format(samps),instance_output)
                print(i,samps)

 
def highdim_analysis_experiment_three_refs(regularization,diameter,trials,instances,dim,initial_samps):
    losses=[]
    #synthesize
    weight_vec=random_weights(3)
    mean_1=random_vec(dim,0.2,0.4)
    mean_2=random_vec(dim,0.4,0.6)
    mean_3=random_vec(dim,0.6,0.8)    
    means=[mean_1,mean_2,mean_3]
    Unif_1=uniform_sample_box(dim,diameter,initial_samps,means[0])
    Unif_2=uniform_sample_box(dim,diameter,initial_samps,means[1])
    Unif_3=uniform_sample_box(dim,diameter,initial_samps,means[2])
    uniform_points = np.random.rand(initial_samps, dim)
    uniform_masses=np.dot(np.ones(initial_samps),1/initial_samps)
    uniform_measure=measure(uniform_points,uniform_masses)
    measure_location_vec=[np.array(Unif_1),np.array(Unif_2),np.array(Unif_3)]
    measure_masses=[np.array(uniform_masses),np.array(uniform_masses),np.array(uniform_masses)]
    output=ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measure_location_vec,measures_weights=measure_masses,X_init=np.array(uniform_measure.points),b=np.array(uniform_measure.masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
    for samps in trials:
        trial_loss=[]
        for j in np.arange(instances):
            Unif_1_instance=uniform_sample_box(dim,diameter,samps,means[0])
            Unif_2_instance=uniform_sample_box(dim,diameter,samps,means[1])
            Unif_3_instance=uniform_sample_box(dim,diameter,samps,means[2])
            instance_masses=np.dot(np.ones(samps),1/samps)
            Unif_1_meas=measure(Unif_1_instance,instance_masses)
            Unif_2_meas=measure(Unif_2_instance,instance_masses)
            Unif_3_meas=measure(Unif_3_instance,instance_masses)
            reference=[Unif_1_meas,Unif_2_meas,Unif_3_meas]
            output_1=sample_array(output,samps)
            output_2=sample_array(output,samps)
            output_measure_1=measure(output_1,instance_masses)
            output_measure_2=measure(output_2,instance_masses)
            potential_vec=[]
            for k in reference:
                g=get_potential(output_measure_1,k,regularization)
                potential_vec.append(g[1])
            entmap_list=[]
            for k in np.arange(len(reference)):
                extended=highdim_extended_map(potential_vec[k],output_measure_2,reference[k],regularization,dim)
                entmap_list.append(extended)
            l2norms=[]
            for p in np.arange(len(entmap_list)):
                for q in np.arange(len(entmap_list)):
                    T_p=entmap_list[p]-output_measure_2.points
                    T_q=entmap_list[q]-output_measure_2.points
                    dotvec=[]
                    for k in np.arange(len(T_p)):
                        innerprod=np.dot(T_p[k],T_q[k])
                        dotvec.append(innerprod)
                    l2diff=np.dot(dotvec,instance_masses)
                    l2norms.append(l2diff)
            l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
            x=cp.Variable(len(entmap_list))
            objective=cp.Minimize(cp.quad_form(x,l2matrix))
            constraints=[x>=0,cp.sum(x)==1]
            problem=cp.Problem(objective,constraints)
            problem.solve()
            optimal_x=x.value
            loss=np.linalg.norm(weight_vec-optimal_x)**2
            trial_loss.append(loss)
            if j%10==0:
                print(samps,j)
            params=[initial_samps,reference,potential_vec,[output_1,output_2],optimal_x]
        losses.append(trial_loss)  
    instance_parameters={'means':[mean_1,mean_2,mean_3],'weights':weight_vec,'output':output}
    return losses

losses=highdim_analysis_experiment_three_refs(regularization,1,sample_range,100,5,initial_samps)

#Parameters used for experiments in paper:
#initial_samps=20000
#losses=highdim_analysis_experiment_three_refs(regularization,1,[10,20,40,80,160,320,640,1280,2560,5120,10240],100,5,initial_samps)


#script to plot results
logtrials=np.log10(sample_range)
avg_loss=[]
for i in losses:
    avg_loss.append(np.mean(i))
log_loss=np.log10(avg_loss)
coefficients = np.polyfit(logtrials, log_loss, 1)
line_of_fit=[]
for i in logtrials:
    line_of_fit.append(coefficients[0]*i+coefficients[1])
plt.figure(figsize=(7, 5))
plt.scatter(logtrials,log_loss, label='(Log_10) L2 loss')
plt.plot(logtrials,line_of_fit,c='red')
textbox_content = "Slope={}, Intercept={}".format(round(coefficients[0],4),round(coefficients[1],3))
plt.text(.5,.76,textbox_content, bbox=dict(facecolor='red', alpha=0.5),transform=plt.gca().transAxes,fontsize=11)
plt.xlabel(r'$\log_{10}$ Sample #',fontsize=14)
plt.ylabel(r'$\log_{10} \|\lambda^*-\hat{\lambda}^n\|^2$')
fig_size = plt.gcf().get_size_inches()
plt.savefig('box_experiment.png')
plt.show()




