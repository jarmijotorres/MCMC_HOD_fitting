import sys,os
import numpy as np
    
#=============== load data to compute the chi^2 ===================#
def chis(y_sim,y_obs,A_n,A_wp):

    n_obs = y_obs[0][0]
    n_obs_std = y_obs[0][1]
    
    wp_obs = y_obs[1][:,0]
    wp_obs_std = y_obs[1][:,1]
    N_bins_data = len(wp_obs)
    
    n_sim = y_sim[0][0]
    n_sim_std = y_sim[0][1]
    
    wp_sim = y_sim[1][:,0]
    wp_sim_std = y_sim[1][:,1]
    
    wp_err_tot = np.sqrt(wp_sim_std**2+wp_obs_std**2)#sum errors in quadrature
    n_err_tot = np.sqrt(n_obs_std + n_sim_std**2.)

    chis_bins = ((wp_sim - wp_obs)/wp_err_tot)**2.

    chi_square_wp = np.sum(chis_bins)
        
    chi_square_n = ((n_sim - n_obs)/n_err_tot)**2
        
    return A_wp*chi_square_wp + A_n*chi_square_n


def chi_wp(y_sim,y_obs):
    
    wp_obs = y_obs[:,0]
    wp_obs_std = y_obs[:,1]
    N_bins_data = len(wp_obs)
    
    wp_sim = y_sim[:,0]
    wp_sim_std = y_sim[:,1]
    
    wp_err_tot = np.sqrt(wp_sim_std**2+wp_obs_std**2)#sum errors in quadrature

    chis_bins = ((wp_sim - wp_obs)/wp_err_tot)**2.

    chi_square_wp = np.sum(chis_bins)
        
    return chi_square_wp


