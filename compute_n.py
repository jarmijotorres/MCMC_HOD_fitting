import numpy as np
import multiprocessing
import time,h5py
from glob import glob
from hod_GR import HOD_mock

t= time.time()
M = 'GR'#sys.argv[1]
B = 1#sys.argv[2]

infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'

data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))

def compute_n(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1_data = HOD_mock(theta,data_r)
        n_sim = len(G1_data) / (1024.**3.)
        wp_true = wp_from_box(G1_data[:,(0,1,2)],Lbox = 1024.,Nsigma = 25)
        
    #wp_model = np.array([wp_true, wp_true*0.01]).T
    #chi2wp = chi_wp(wp_model,wp_obs)
        
    return n_sim

if __name__ == '__main__': 
    sample_chains = np.load('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/samples_s3.npy')
    pool = multiprocessing.Pool(processes=16)
    F = np.zeros(len(sample_chains))
    F[:] = pool.map(compute_n,sample_chains)
    pool.close() 
    pool.join()    
    np.save('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/n_gal.npy',F)