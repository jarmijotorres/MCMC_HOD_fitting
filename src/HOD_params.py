import numpy as np
from scipy.special import erf
from hod_GR import HOD_mock
from binning_data import *

M='GR'
B='5'
infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'
data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))#parent halo data

Lbox=1024.0                                     # box size in Mpc/h
Mhalo_min=4.0E11  

theta = np.array([13.22766781, 14.1677514,  13.35623038,  0.72601731,  1.1742457 ])
logMmin, log_M1, log_M0, sigma_logM, alpha = theta

#define HOD
Mrange = np.logspace(11.,16,num=200)

def HOD_analytic(Mi,theta):
    logMmin, log_M1, log_M0, sigma_logM, alpha = theta
    N_cen = 0.5*(1+erf((np.log10(Mi)- logMmin)/(sigma_logM)))
    N_sat = N_cen*((Mi-10**log_M0)/(10**log_M1))**alpha
    N_sat[np.isnan(N_sat)] = 0.0
    return N_cen + N_sat

G1 = HOD_mock(theta,data_r)
G2 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_GR_LOWZ_z0.3_Box5_test_1halo.dat')#idl simil

ids_cen_py = np.where(G1[:,-1]==1)[0]
ids_cen_idl = np.where(G2[:,-1]==1)[0]

N_gal_halo_py = np.zeros_like(ids_cen_py)
for i in range(len(N_gal_halo_py)):
    if i != len(N_gal_halo_py)-1:
        N_gal_halo_py[i] = len(G1[ids_cen_py[i]:ids_cen_py[i+1],-1])
    else:
        N_gal_halo_py[i] = len(G1[ids_cen_py[i]:,-1])

N_gal_halo_idl = np.zeros_like(ids_cen_idl)
for i in range(len(N_gal_halo_idl)):
    if i != len(N_gal_halo_idl)-1:
        N_gal_halo_idl[i] = len(G2[ids_cen_idl[i]:ids_cen_idl[i+1],-1])
    else:
        N_gal_halo_idl[i] = len(G2[ids_cen_idl[i]:,-1])
        

M_cens = np.log10(G1[:,3][G1[:,4]==1])
Ns,_=binning_function(M_cens,N_gal_halo_py,logMmin,M_cens.max(),Nb=10,percentile=34)

M_cens2 = np.log10(G2[:,3][G2[:,4]==1])
Ns2,_=binning_function(M_cens,N_gal_halo_py,logMmin,M_cens.max(),Nb=10,percentile=34)


##### use halo parent catalogue ############