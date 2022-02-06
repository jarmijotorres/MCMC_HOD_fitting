import yaml
import sys

#this script generates a yaml file that serves as an input for the mcmc routines. The file should contain parameters as the number of iterations and, names of the outputs to generate, and weights for the chi square function. 
#Also, it needs the parameters of the parent halo to generate the HOD mock. For this simulations there is 3 different models of gravity (standar General relativity GR; f(R) modified gravity F5, F6) and there are 5 boxes realizations.

yaml_name = sys.argv[1]#'/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/params/param_ini.yaml'

M='GR'
Lbox = 768
halo_file = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z0.3_L'+str(Lbox)+'_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'

A_n = 0.5
A_wp = 0.5
#number of iterations 
nwalkers = 112
burnin_it = 800
prod_it = 1600

#burnin_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMCpost_burnin_'+str(burnin_it)+'it_'+str(nwalkers)+'wlk_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.24_0.46_err_3sigma_sim_subhaloes.npy'
#burnin_logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMClklhd_burnin_'+str(burnin_it)+'it_'+str(nwalkers)+'wlk_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.24_0.36_err_3sigma_sim_subhaloes.npy'
chain_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.24_0.36_err_sigma_sim_fullcov_subhaloes.npy'
logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.24_0.36_err_sigma_fullcov_sim_subhaloes.npy'

wp_obs = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_wp_logrp0.5_50_13logrpbins_pimax100_z0.24_0.36.txt'
n_obs = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/number_density/JK100_ngal_z0.24_0.36.txt'
cov_matrix = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_Cov_logrp0.5_50_13bins.txt'#targets+covariance

#create dictionary
input_dict = {
    'Model':M,
    'Lbox': Lbox,
    'halo_file':halo_file,
    'observable_wp': wp_obs,
    'observable_n': n_obs,
    'cov_matrix': cov_matrix,
    'A_n': A_n,
    'A_wp': A_wp,
    #'survey_mask': '/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/randoms_LOWZ_North_NS1024.mask',
    #'randoms_clustering': '/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/random0_LOWZ_North_z0.2_0.4.hdf5',
    #'RR_pairs': '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/RR_est_random0_xi3Dps_rp0.5_50_20bins_pi_0_80_80bins.npy',
    'N_walkers':nwalkers,
    'burnin_iterations': burnin_it,
    'production_iterations': prod_it,
    #'burnin_file':burnin_file,
    #'burnin_logProb_file':burnin_logProb_file,
    'chain_file':chain_file,
    'logProb_file':logProb_file
}


with open(yaml_name, 'w') as file:
    save_dict = yaml.dump(input_dict, file)
    
