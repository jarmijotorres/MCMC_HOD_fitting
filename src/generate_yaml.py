import yaml
import sys

#this script generates a yaml file that serves as an input for the mcmc routines. The file should contain parameters as the number of iterations and, names of the outputs to generate, and weights for the chi square function. 
#Also, it needs the parameters of the parent halo to generate the HOD mock. For this simulations there is 3 different models of gravity (standar General relativity GR; f(R) modified gravity F5, F6) and there are 5 boxes realizations.

yaml_name = sys.argv[1]#'/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/params/param_ini.yaml'

target=sys.argv[2]
M=sys.argv[3]#'GR'
Lbox = 768
A_n = float(sys.argv[4])
A_wp = float(sys.argv[5])
#number of iterations 
nwalkers = 28
#burnin_it = 1000
prod_it = 500
N_batches = int(sys.argv[6])

if target == 'CMASS_North':
    redshift=0.5
    zmin=0.474
    zmax=0.528

if target == 'LOWZ_North':
    redshift=0.3
    zmin=0.24
    zmax=0.36    

halo_file = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z'+str(redshift)+'_L768_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5'
    
out_dir = '/cosma7/data/dp004/dc-armi2/mcmc_runs/'+target+'_'+M+'_An'+str(A_n)+'_Awp'+str(A_wp)+'/'

if N_batches <= 2:
    use_prev_state = False
else:
    use_prev_state = True

if use_prev_state:
    p0 = out_dir +  'chains/MCMCpost_chains_HOD_'+M+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_'+target+'_z'+str(zmin)+'_'+str(zmax)+'_err_1sigma_fullcov_sim_subhaloes_batch_%d.npy'%(N_batches - 3)
else:
    p0 = None
    
chain_file = out_dir +  'chains/MCMCpost_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_'+target+'_z'+str(zmin)+'_'+str(zmax)+'_err_1sigma_fullcov_sim_subhaloes'
logProb_file = out_dir + 'likelihoods/MCMClklhd_chains_HOD_'+str(M)+'_L'+str(Lbox)+'_'+str(prod_it)+'it_'+str(nwalkers)+'walkers_'+str(A_n)+'An_'+str(A_wp)+'Awp_target_'+target+'_z'+str(zmin)+'_'+str(zmax)+'_err_1sigma_fullcov_sim_subhaloes'

wp_obs = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_wp_galaxy_'+target+'_z'+str(zmin)+'_'+str(zmax)+'_RA_DEC_Z_weight_JK100.hdf5_logrp0.5_50_10bins_pimax80.txt'
n_obs = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/number_density/JK100_'+target+'_ngal_z'+str(zmin)+'_'+str(zmax)+'.txt'
cov_matrix = '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_Cov_'+target+'_z'+str(zmin)+'_'+str(zmax)+'_logrp0.5_50_13bins.txt'#targets+covariance

#create dictionary
input_dict = {
    'Model':M,
    'Lbox': Lbox,
    'halo_file': halo_file,
    'target': target,
    'z_snap': redshift,
    'observable_wp': wp_obs,
    'observable_n': n_obs,
    'cov_matrix': cov_matrix,
    'A_n': A_n,
    'A_wp': A_wp,
    #'survey_mask': '/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/randoms_LOWZ_North_NS1024.mask',
    #'randoms_clustering': '/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/random0_LOWZ_North_z0.2_0.4.hdf5',
    #'RR_pairs': '/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/RR_est_random0_xi3Dps_rp0.5_50_20bins_pi_0_80_80bins.npy',
    'N_walkers': nwalkers,
    'N_batches': N_batches,    #'burnin_iterations': burnin_it,
    'production_iterations': prod_it,
    #'burnin_file':burnin_file,
    #'burnin_logProb_file':burnin_logProb_file,
    'chain_file': chain_file,
    'logProb_file': logProb_file,
    'prev_state': p0
}

with open(yaml_name, 'w') as file:
    savedict = yaml.dump(input_dict, file)
    