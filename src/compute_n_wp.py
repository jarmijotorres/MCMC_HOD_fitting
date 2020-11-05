import numpy as np
import multiprocessing
import time,h5py, yaml,sys
from glob import glob
from hod_GR import HOD_mock
from chi2 import chi_wp
from tpcf_obs import wp_from_box

yaml_name = sys.argv[1]
with open(yaml_name) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    input_dict = yaml.load(file)

t= time.time()
M = input_dict['Model']
B = input_dict['Box']
A_n = input_dict['A_n']
A_wp = input_dict['A_wp']
nwalkers = input_dict['N_walkers']
burnin_it = input_dict['burnin_iterations']
prod_it = input_dict['production_iterations']
burnin_file = input_dict['burnin_file']
chain_file = input_dict['chain_file']
logProb_file = input_dict['logProb_file']
halo_file = input_dict['halo_file']
observable_wp = input_dict['observable_wp'] 
observable_n = input_dict['observable_n']

wp_obs = np.loadtxt(observable_wp)
data_r = np.loadtxt(halo_file,usecols=(2,33,8,9,10,5,6))

def compute_n_wp(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1_data = HOD_mock(theta,data_r)
        n_sim = len(G1_data) / (1024.**3.)
        wp_true = wp_from_box(G1_data[:,(0,1,2)],n_threads=1,Lbox = 1024.,Nsigma = 25)
        
    wp_model = np.array([wp_true, wp_true*0.01]).T
    chi2wp = chi_wp(wp_model,wp_obs)
        
    return np.array([n_sim,chi2wp])

chain_walkers = np.load(chain_file)
samples_1D = chain_walkers.reshape((chain_walkers.shape[0]*chain_walkers.shape[1],chain_walkers.shape[2]))

pool = multiprocessing.Pool(processes=28)
Fn_wp = np.zeros((len(samples_1D),2))
Fn_wp[:] = pool.map(compute_n_wp,samples_1D)
pool.close() 
pool.join()
Fnwp_rs = Fn_wp.reshape((chain_walkers.shape[0],chain_walkers.shape[1],2))
np.save(chain_file.split(sep='chains')[0]+'ns/HOD_mcmc_ngal_wp'+chain_file.split(sep='chains')[-1],Fn_wp)
#print('done')
#print(F[:10])
