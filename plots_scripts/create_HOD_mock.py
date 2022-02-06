import numpy as np
import sys
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
import h5py

# observational data to fit
wp_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wp_logrp0.5_50_20bins_pimax80_z0.2_0.4.txt')
n_data, n_err = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_numberDensity_z0.2_0.4.txt')
rp = wp_data[:,0]
wp_rp = wp_data[:,1]/wp_data[:,0]
wp_err = wp_data[:,2]/wp_data[:,0]

#theta_max = np.array([13.102, 14.0771384 , 13.11738439,  0.15,  1.011])#GR simulations
#theta_max = np.array([13.16076923, 13.982307692, 13.580      ,  0.11153846,  0.99153846])#F5


#======= L768 =======#
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L768_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5','r')

G0 = HOD_mock_subhaloes(theta_max,haloes_table,Lbox=768,weights_haloes=None)
n0 = len(G0)/768**3
wp_sim_L768 = wp_from_box(G0,n_threads=16,Lbox = 768,Nsigma = 20,return_rpavg=True)

#====== L1536 =========#
HMF_ref = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_GR_z0.3_L768_dlogM0.15.dat')
HMF_L1536 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_GR_z0.3_L1536_dlogM0.15.dat')

haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L1536_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5','r')
w1 = weight_function(haloes_table['MainHaloes']['M200c'],HMF_incomp=HMF_L1536,HMF_comp=HMF_ref)
G1 = HOD_mock_subhaloes(theta_max,haloes_table,Lbox=1536,weights_haloes=w1)
n0 = len(G1)/1536**3
wp_sim_L1536 = wp_from_box(G1,n_threads=16,Lbox = 1536,Nsigma = 20,return_rpavg=True)


#fit line (power law wp/rp)
def line(x,A,B):
    return A*x + B 

from scipy.optimize import curve_fit
P0 = curve_fit(line,np.log10(wp_data[:,0]),np.log10(wp_data[:,1]/wp_data[:,0]),p0=[3,-2])

rrange = np.arange(np.log10(wp_data[0,0]),np.log10(wp_data[-1,0]),0.01)
pl = line(rrange,P0[0][0],P0[0][1])
pl_bins = 10**(line(np.log10(wp_data[:,0]),P0[0][0],P0[0][1]))


f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].errorbar(wp_data[:,0],wp_rp,yerr=3*wp_err,fmt='ko',label='LOWZ $0.2<z<0.4$')
ax[0].plot(wp_sim0[:,0],wp_sim0[:,1],'r-',label='HOD mock box $z=0.3$')
ax[0].plot(rp,wp_1,'r--',label='HOD mock lightcone')
ax[1].errorbar(wp_data[:,0],np.zeros_like(wp_rp),yerr=3*wp_err/wp_rp,fmt='ko')
ax[1].hlines(0,0.5,50,color='grey',linewidth=2.)
ax[1].plot(wp_sim0[:,0],(wp_sim0[:,1]- wp_rp)/wp_rp,'r-')
ax[1].plot(rp,(wp_1- wp_rp)/wp_rp,'r--')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$w_p(r_p)/r_p$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax[0].legend(loc=1,prop={'size':14})
ax[1].set_ylim(-0.5,0.5)
ax[0].set_xlim(0.5,50)
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.show()

