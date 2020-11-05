import numpy as np
from glob import glob
from halotools.mock_observables import wp

def wp_from_box(G1,n_threads,Lbox = 1024.,Nsigma = 25):
    ## replace by real error estimation  #number of bins on rp log binned
    sini = 0.1
    sfini = 80.0
    sigma = np.logspace(np.log10(sini),np.log10(sfini),Nsigma+1)
    pimax = 100 
    pi = np.linspace(0,pimax,pimax+1)
    dpi = np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    wp_obs = wp(G1,sigma,pi_max=pimax,period=[Lbox,Lbox,Lbox],num_threads=n_threads,randoms=None)
    wp_true = wp_obs / rp
    
    return wp_true