import numpy as np
import healpy as hp
from glob import glob
#from halotools.mock_observables import wp
from Corrfunc.theory import wp
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import h5py,sys
from astropy.cosmology import Planck15 as cosmo
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value

def wp_from_box(G1,n_threads,Lbox,Nsigma,z_snap,return_rpavg=False):
    Om0 = cosmo.Om0
    Ol0 = 1 - cosmo.Om0
    #z_snap = 0.3

    Hz = 100.0*np.sqrt(Om0*(1.0+z_snap)**3 + Ol0)
    Hz_i = (1+z_snap)/Hz
    
    ## replace by real error estimation  #number of bins on rp log binned
    sini = 0.5
    sfini = 50.0
    sigma = np.logspace(np.log10(sini),np.log10(sfini),Nsigma+1)
    pimax = 100
    pi = np.linspace(0,pimax,pimax+1)
    dpi = np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    
    #move z-coordinate to redshift space
    
    vz = G1[:,5]
    z_obs = G1[:,2] + vz*Hz_i
    z_obs[z_obs < 0] += Lbox
    z_obs[z_obs > Lbox] -= Lbox
    
    wp_obs = wp(boxsize=Lbox,binfile=sigma,X=G1[:,0],Y=G1[:,1],Z=z_obs,pimax=pimax,nthreads=n_threads,weights=None,output_rpavg=True, weight_type=None,xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,max_cells_per_dim=100)
    wp_true = wp_obs['wp'] / wp_obs['rpavg']
    if return_rpavg:
        return np.array([wp_obs['rpavg'],wp_true]).T
    else:
        return wp_true

def MCF_from_tpcf(xi2d_m,xi2d_um):
    xi1 = xi2d_m['full_result/xi']
    xi2 = xi2d_um['full_result/xi']
    rp = xi2d_m['full_result/Axis_1_bin_centre'][:]
    drp = xi2d_m['full_result/Axis_1_bin_width'][:]
    pi = xi2d_m['full_result/Axis_2_bin_centre'][:]
    dpi = xi2d_m['full_result/Axis_2_bin_width'][:]
    dlogpi = np.diff(np.log(dpi))
    wp_full_w = np.sum(xi1[:]*pi,axis=1)*dlogpi[0]
    wp_full_uw = np.sum(xi2[:]*pi,axis=1)*dlogpi[0]
    
    MCF = (1 + (wp_full_w/rp))/(1+(wp_full_uw/rp))
    return MCF#np.array([rp,MCF]).T


def MCF_JK_from_tpcf(xi2d_m,xi2d_um,NJK):
    rp = xi2d_m['full_result/Axis_1_bin_centre'][:]
    drp = xi2d_m['full_result/Axis_1_bin_width'][:]
    pi = xi2d_m['full_result/Axis_2_bin_centre'][:]
    dpi = xi2d_m['full_result/Axis_2_bin_width'][:]
    dlogpi = np.diff(np.log(dpi))
    MCFs_JK = []
    for i in range(NJK):
        xi1 = xi2d_m['jk_reg%d/xi'%i]
        xi2 = xi2d_um['jk_reg%d/xi'%i]
        wp_full_w = np.sum(xi1[:]*pi,axis=1)*dlogpi[0]
        wp_full_uw = np.sum(xi2[:]*pi,axis=1)*dlogpi[0]
        MCF_JK = (1 + (wp_full_w/rp))/(1+(wp_full_uw/rp))
        MCFs_JK.append(MCF_JK)
    MCF_mean = np.mean(MCFs_JK,axis=0)
    Nrp = len(rp)
    C_ = np.zeros((Nrp,Nrp))
    for i in range(len(C_)):
        for j in range(len(C_)):
            for k in range(NJK):
                C_[i][j] += (NJK-1)/float(NJK)*(MCFs_JK[k][i] - MCF_mean[i])*(MCFs_JK[k][j] - MCF_mean[j])
    MCF_err = np.sqrt(C_.diagonal())
    
    S = np.array([rp,MCF_mean,MCF_err]).T
    return S,C_