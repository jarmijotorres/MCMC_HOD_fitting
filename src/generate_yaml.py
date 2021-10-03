import yaml
import sys

#this script generates a yaml file that serves as an input for the mcmc routines. The file should contain parameters as the number of iterations and, names of the outputs to generate, and weights for the chi square function. 
#Also, it needs the parameters of the parent halo to generate the HOD mock. For this simulations there is 3 different models of gravity (standar General relativity GR; f(R) modified gravity F5, F6) and there are 5 boxes realizations.

yaml_name = sys.argv[1]#'/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/params/param_ini.yaml'

M='GR'
Lbox = 1536
halo_file = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z0.3_L'+str(Lbox)+'_ID_M200c_pos_R200_c200scatter_logMhalomin_11.2.hdf5'

A_n = 0.5
A_wp = 0.5
#number of iterations 
nwalkers = 56
burnin_it = 200
prod_it = 400

burnin_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMCpost_burnin_'+str(burnin_it)+'it_'+str(nwalkers)+'wlk_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.2_0.4_err_3sigma_sim_subhaloes.npy'
burnin_logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMClklhd_burnin_'+str(burnin_it)+'it_'+str(nwalkers)+'wlk_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.2_0.4_err_3sigma_sim_subhaloes.npy'
chain_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.2_0.4_err_3sigma_sim_subhaloes.npy'
logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_LOWZ_z0.2_0.4_err_3sigma_sim_subhaloes.npy'

#create dictionary
input_dict = {
    'Model':M,
    'Lbox': Lbox,
    'halo_file':halo_file,
    'observable_wp': '/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wp_logrp0.5_50_20bins_pimax80_z0.2_0.4.txt',
    'observable_n': '/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_numberDensity_z0.2_0.4.txt',
    'A_n': A_n,
    'A_wp': A_wp,
    'N_walkers':nwalkers,
    'burnin_iterations': burnin_it,
    'production_iterations': prod_it,
    'burnin_file':burnin_file,
    'burnin_logProb_file':burnin_logProb_file,
    'chain_file':chain_file,
    'logProb_file':logProb_file
}


with open(yaml_name, 'w') as file:
    save_dict = yaml.dump(input_dict, file)
    
