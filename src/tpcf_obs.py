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

def wp_from_box(G1,n_threads,Lbox,Nsigma,return_rpavg=False):
    ## replace by real error estimation  #number of bins on rp log binned
    sini = 0.5
    sfini = 50.0
    sigma = np.logspace(np.log10(sini),np.log10(sfini),Nsigma+1)
    pimax = 100
    pi = np.linspace(0,pimax,pimax+1)
    dpi = np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    wp_obs = wp(boxsize=Lbox,binfile=sigma,X=G1[:,0],Y=G1[:,1],Z=G1[:,2],pimax=pimax,nthreads=n_threads,weights=G1[:,4],output_rpavg=True, weight_type='pair_product',xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,max_cells_per_dim=100)
    wp_true = wp_obs['wp'] / wp_obs['rpavg']
    if return_rpavg:
        return np.array([wp_obs['rpavg'],wp_true]).T
    else:
        return wp_true

def box_to_lightcone(G1,zf):
    G1_BB = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                G1_BB.append(G1[:,(0,1,2,3,5)]+np.array([i,j,k,0,0])*768.0)
    G1_BB = np.concatenate(G1_BB,axis=0)

    p0 = np.concatenate((np.mean(G1[:,(0,1,2)],axis=0),[0.0,0.0]))
    observer_frame = G1_BB - p0
    del G1_BB

    nn_tree = cKDTree(observer_frame[:,(0,1,2)])
    origin = np.array([0.0,0.0,0.0])
    #ri = dcomov(zi)
    rf = dcomov(zf)
    #lc_z02 = nn_tree.query_ball_point(x=origin,r=ri,)
    lc_z = nn_tree.query_ball_point(x=origin,r=rf,)

    #LC_M = observer_frame[lc_z02]
    LC_L = observer_frame[lc_z]

    c = SkyCoord(x=LC_L[:,0], y=LC_L[:,1], z=LC_L[:,2], unit='Mpc', representation_type='cartesian')
    c.representation_type = 'spherical'

    #phi = c.ra.value[(c.distance.value > 100)&(c.distance.value < 150)]
    #theta = c.dec.value[(c.distance.value > 100)&(c.distance.value < 150)]

    RA = c.ra.value#to(u.radian).value#[(c.dec.value > 0)&(c.dec.value < 5)]
    DEC = c.dec.value#
    r = c.distance.value#[(c.dec.value > 0)&(c.dec.value < 5)]
    LC_data = np.array([RA,DEC,r]).T

    return LC_data

def masking_LC(LC,mask_IDs,zi = 0.2,zf = 0.4):

    NSIDE=1024
    ranmask = np.zeros(12*NSIDE**2)
    ranmask[mask_IDs.astype(int)] = 1.
    
    LC_chunk = LC[(LC[:,2]>=dcomov(zi))&(LC[:,2]<=dcomov(zf))]
    
    alpha = np.radians(LC_chunk[:,0])
    delta = np.radians(LC_chunk[:,1]) + np.pi/2.

    is_p = hp.ang2pix(NSIDE, theta=delta, phi=alpha, nest=True)
    
    mask_points = ranmask[is_p]==1.0
    masked_LC = LC_chunk[mask_points]
    return masked_LC

def wp_from_LC(C0,randoms,RR_pairs,n_threads,Nsigma):
    R0 = randoms['data']
    sini = 0.5
    sfini = 50.0
    sigma = np.logspace(np.log10(sini),np.log10(sfini),Nsigma+1)
    pimax = 80
    Npi = 80
    pi = np.linspace(0,pimax,pimax+1)
    dpi = np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    
    D1D2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=n_threads,pimax=pimax,binfile=sigma,RA1=C0[:,0],DEC1=C0[:,1],CZ1=C0[:,2],weights1=np.ones_like(C0[:,2]),is_comoving_dist=True,weight_type='pair_product')
    
    D1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=n_threads,pimax=80,binfile=sigma,RA1=C0[:,0],DEC1=C0[:,1],CZ1=C0[:,2],weights1=np.ones_like(C0[:,0]),RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['comoving_distance'],weights2=R0['weight'],is_comoving_dist=True,weight_type='pair_product')
    
    NRC = len(R0)/len(C0)
    D1D2 = D1D2_est['npairs']
    D1R2 = D1R2_est['npairs'] / NRC
    R1R2 = RR_pairs['npairs'] / (NRC*NRC)
    xi2D = (D1D2 - D1R2 - D1R2 + R1R2)/R1R2
    xi_pi_sigma = np.zeros((Npi,Nsigma))
    for j in range(Nsigma):
        for i in range(Npi):
            xi_pi_sigma[i,j] = xi2D[i+j*Npi]
            
    
    wp_LC = 2*np.sum(xi_pi_sigma,axis=0)
    wp_rp = wp_LC/rp
    
    return wp_rp