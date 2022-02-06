import sys,os
import numpy as np
    
#=============== load data to compute the chi^2 ===================#
def chis(y_sim,y_obs,cov_matrix,A_n,A_wp):

    n_obs = y_obs[0]
    
    wp_obs = y_obs[1][:,0]
    wp_obs_std = y_obs[1][:,1]
    N_bins_data = len(wp_obs)
    
    n_sim = y_sim[0]
    
    wp_sim = y_sim[1][:,0]
    wp_sim_std = y_sim[1][:,1]
    
    chi_square_wp = chis_wp_cov(wp_sim,wp_obs,cov_matrix)
        
    chi_square_n = chis_n(n_sim,n_obs)
        
    return A_wp*chi_square_wp + A_n*chi_square_n


def chis_wp(wp_mod,wp_data):

    wp_obs = wp_data[:,0]
    wp_obs_std = wp_data[:,1]
    N_bins_data = len(wp_data)
    
    wp_sim = wp_mod[:,0]
    wp_sim_std = wp_mod[:,1]
    
    wp_err_tot = np.sqrt(wp_sim_std**2+wp_obs_std**2)#sum errors in quadrature

    chis_bins = ((wp_sim - wp_obs)/wp_err_tot)**2.

    chi_square_wp = np.sum(chis_bins)
        
    return chi_square_wp

def chis_n(n_mod,n_data):

    n_obs = n_data[0]
    n_obs_std = n_data[1]
    
    n_sim = n_mod[0]
    n_sim_std = n_mod[1]
    
    n_err_tot = np.sqrt(n_obs_std**2. + n_sim_std**2.)

    chi_square_n = ((n_sim - n_obs)/n_err_tot)**2
        
    return chi_square_n

def chis_wp_cov(wp_sim,wp_obs,cov):
    
    y_ = wp_obs - wp_sim
    
    cov_inv = np.linalg.inv(cov)

    chi_square_wp = np.dot(np.dot(y_,cov_inv),y_)
        
    return chi_square_wp