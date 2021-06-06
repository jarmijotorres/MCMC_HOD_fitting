import numpy as np
from glob import glob
#from halotools.mock_observables import wp
from Corrfunc.theory import wp

def wp_from_box(G1,n_threads,Lbox,Nsigma):
    ## replace by real error estimation  #number of bins on rp log binned
    sini = 0.5
    sfini = 50.0
    sigma = np.logspace(np.log10(sini),np.log10(sfini),Nsigma+1)
    pimax = 80
    pi = np.linspace(0,pimax,pimax+1)
    dpi = np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    wp_obs = wp(boxsize=Lbox,binfile=sigma,X=G1[:,0],Y=G1[:,1],Z=G1[:,2],pimax=pimax,nthreads=n_threads,weights=G1[:,4],output_rpavg=True, weight_type='pair_product',xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,max_cells_per_dim=100)
    wp_true = wp_obs['wp'] / wp_obs['rpavg']
    
    return wp_true

