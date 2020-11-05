import yaml

#this script generates a yaml file that serves as an input for the mcmc routines. The file should contain parameters as the number of iterations and, names of the outputs to generate, and weights for the chi square function. 
#Also, it needs the parameters of the parent halo to generate the HOD mock. For this simulations there is 3 different models of gravity (standar General relativity GR; f(R) modified gravity F5, F6) and there are 5 boxes realizations.

yaml_name = '/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/param_ini.yaml'

M='GR'
B = 1
halo_file = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'

A_n = 0.1 
A_wp = 0.9
#number of iterations
nwalkers = 20
burnin_it = 1000
prod_it = 3000

burnin_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/burnin_'+str(nwalkers)+'_theta1_wlk_'+str(burnin_it)+'_'+str(A_n)+'An_'+str(A_wp)+'Awp.npy'
burnin_logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/burnin_'+str(nwalkers)+'_theta1_logProb_wlk_'+str(burnin_it)+'_'+str(A_n)+'An_'+str(A_wp)+'Awp.npy'
chain_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/HOD_mcmcpost_chains_theta1_'+str(M)+'_box'+str(B)+'_'+str(burnin_it)+'burniniter_'+str(prod_it)+'proditer_'+str(nwalkers)+'walkers_chi2_'+str(A_n)+'An_'+str(A_wp)+'Awp.npy'
logProb_file = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/HOD_mcmcpost_logprob_theta1_'+str(M)+'_box'+str(B)+'_'+str(burnin_it)+'burniniter_'+str(prod_it)+'iter_'+str(nwalkers)+'walkers_chi2_'+str(A_n)+'An_'+str(A_wp)+'Awp.npy'


#create dictionary
input_dict = {
    'Model':M,
    'Box': B,
    'halo_file':halo_file,
    'observable_wp': '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/test/wp_13.15000_14.20000_13.30000_0.60000_1.00000.txt',
    'observable_n': '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/test/ngal_13.15000_14.20000_13.30000_0.60000_1.00000.txt',
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
    