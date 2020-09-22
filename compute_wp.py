import numpy as np
import multiprocessing
import time,h5py
from glob import glob
from hod_GR import HOD_mock
from chi2 import chi_wp

t= time.time()
M = 'GR'#sys.argv[1]
B = 1#sys.argv[2]

infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'

infile_data_wp = '/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/wp_'+M+'_box'+str(B)+'_test.txt'

wp_obs = np.loadtxt(infile_data_wp,usecols=(1,2,))

data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))

def compute_n(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/cats/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1_data = HOD_mock(theta,data_r)
        n_sim = len(G1_data) / (1024.**3.)
        wp_true = wp_from_box(G1,Lbox = 1024.,Nsigma = 25)
        
    wp_model = np.array([wp_true, wp_true*0.01]).T
    chi2wp = chi_wp(wp_model,wp_obs)
        
    return np.array([n_sim,chi2wp])




if __name__ == '__main__': 
    chain_walkers = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/HOD_mcmcpost_chains_GR_box1_1000iter_10walkers_chi2_0.5An_0.5Awp.npy')
    samples_1D = chain_walkers.reshape((chain_walkers.shape[0]*chain_walkers.shape[1],chain_walkers.shape[2]))
    pool = multiprocessing.Pool(processes=16)
    Fn_wp = np.zeros((len(sample_chains),2))
    Fn_wp[:] = pool.map(compute_n_wp,sample_1D)
    pool.close() 
    pool.join()    
    np.save('n_chiswp_gal_test.npy',Fn_wp)
    #print('done')
    #print(F[:10])
    
    