import numpy as np
from hod import *
from tpcf_obs import *
import h5py

# observational data to fit
wp_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wp_logrp0.5_50_20bins_pimax80_z0.2_0.4.txt')

wp_Parejko_2009 = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/wp_DR9_Parejko_2009.dat')

#load haloes from simulation
haloes_sim = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L768_M200c_pos_R200_c200_logMhalomin_11.2.hdf5','r')
haloes = haloes_sim['data']

theta_Manera2015_LOWZ = np.array([13.2,14.32,13.24,0.62,0.93])
G1 = HOD_mock(theta_Manera2015_LOWZ,haloes,Lbox=768,weights_haloes=None)

wp_sim = wp_from_box(G1,n_threads=16,Lbox = 768,Nsigma = 15)

haloes_sim = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L1536_M200c_pos_R200_c200_logMhalomin_11.2.hdf5','r')
halo_catalogue = haloes_sim['data']

w1 = weight_function(haloes['M200'],HMF_incomp=HMF_sample,HMF_comp=HMF_ref)

G1 = HOD_mock(theta_Manera2015_LOWZ,halo_catalogue,Lbox=1536,weights_haloes=None)

wp_sim_LR = wp_from_box(G1,n_threads=16,Lbox = 1536,Nsigma = 15)

f,ax = plt.subplots(1,1,figsize=(7,6))
ax.errorbar(wp_data[:,0],wp_data[:,1]/wp_data[:,0],yerr=wp_data[:,2]/wp_data[:,0],fmt='ko')
ax.plot(wp_data[:,0],wp_sim*1.5,'r-')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()