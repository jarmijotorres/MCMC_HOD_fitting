import numpy as np
import healpy as hp
import sys,h5py
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from scipy.spatial import Voronoi,ConvexHull,cKDTree
from astropy.coordinates import SkyCoord
from scipy.interpolate import UnivariateSpline
from multiprocessing import Pool

from mpi4py import MPI
comm =MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Om0 = cosmo.Om0
Ol0 = 1 - cosmo.Om0
alpha0 = 108.95322520021244#np.min(G0['RA'])
#_snap = 0.3

def voronoi_volume(XY_box):
    v = Voronoi(XY_box[:,(0,1)])
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

def dcomov(z):
    '''
    include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value
zrange = np.arange(1e-3,1.0,1e-3)
dc_z = dcomov(zrange)
z_interp = UnivariateSpline(dc_z,zrange,k=5)

def dc2z(z,dc):
    return dcomov(z) - dc
dc2z_vec = np.vectorize(dc2z)

def deg_to_Mpc_overh(z):
    '''
    use arcmin to comoving at redshift z divide by 0.25 
    to obtain kpc per deg and divide by 1000 to get results in Mpc 
    (include little h to get Mpc/h units)
    '''
    return cosmo.h*60*(cosmo.kpc_comoving_per_arcmin(z).value)/1e3

def RADECZ_2_XY(G1_chunk,z_mean,alpha0):
    z = dcomov(G1_chunk[:,2])
    dc_mean = dcomov(z_mean)
    alpha = (G1_chunk[:,0])
    delta = np.radians(G1_chunk[:,1])
    y = np.sin(delta)*dc_mean
    x = (alpha - alpha0)*deg_to_Mpc_overh(z_mean)
    XY_chunk = np.array([x,y]).T
    return XY_chunk

def survey_tessellation(G0,survey_dp):
    Ns=16
    slice_edges = np.linspace(0.24,0.36,Ns+1)
    vols_all = []
    G1_all = []
    for i in range(Ns):
        zi = slice_edges[i]
        zf = slice_edges[i+1]
        ddc = dcomov(zf) - dcomov(zi)
        z_mean = 0.5*(zi+zf)
        dz = zf-zi

        G0_slice = G0[(G0['Z']>=zi)&(G0['Z']<=zf)]

        wrap_slice = survey_dp[(survey_dp[:,2]>=zi)&(survey_dp[:,2]<=zf)]

        fill_slice = np.array([G0_slice['RA'],G0_slice['DEC'],G0_slice['Z']]).T

        XY_fill = RADECZ_2_XY(fill_slice,z_mean,alpha0)
        XY_wrap = RADECZ_2_XY(wrap_slice,z_mean,alpha0)

        XY_chunk = np.concatenate([XY_fill[:,(0,1)],XY_wrap[:,(0,1)]],axis=0)

        V2D = voronoi_volume(XY_chunk)
        vol2D_chunk = V2D[:G0_slice.shape[0]]

        vols_all.append(vol2D_chunk)
        G1_all.append(G0_slice)

    vols = np.concatenate(vols_all)
    G1 = np.concatenate(G1_all,axis=0)
    
    return np.array([G1['RA'],G1['DEC'],G1['Z'],G1['weight'],vols]).T

def chunk_tessellation(XY_chunk):

    V2D = voronoi_volume(XY_chunk)
    
    return V2D

def survey_tessellation_MT(Ns,z_l,z_h,G0,survey_dp):
    slice_edges = np.linspace(z_l,z_h,Ns+1)
    XY_chunks = []
    Ngals_per_chunk = []
    G0_chunks = []
    for i in range(Ns):
        zi = slice_edges[i]
        zf = slice_edges[i+1]
        z_mean = 0.5*(zi+zf)

        G0_slice = G0[(G0['Z']>=zi)&(G0['Z']<=zf)]
        Ngals_per_chunk.append(G0_slice.shape[0])
        G0_chunks.append(G0_slice)
        wrap_slice = survey_dp[(survey_dp[:,2]>=zi)&(survey_dp[:,2]<=zf)]
        fill_slice = np.array([G0_slice['RA'],G0_slice['DEC'],G0_slice['Z']]).T

        XY_fill = RADECZ_2_XY(fill_slice,z_mean,alpha0)
        XY_wrap = RADECZ_2_XY(wrap_slice,z_mean,alpha0)

        XY_chunk = np.concatenate([XY_fill[:,(0,1)],XY_wrap[:,(0,1)]],axis=0)
        XY_chunks.append(XY_chunk)
        
    p = Pool(processes=Ns)
    A = p.map(func=chunk_tessellation,iterable=XY_chunks)
    V0_chunks = []
    for i,chunk in enumerate(A):
        V0_chunks.append(chunk[:Ngals_per_chunk[i]])
    V0_all = np.concatenate(V0_chunks)
    G0_all = np.concatenate(G0_chunks)
    return np.array(list(zip(G0_all['RA'],G0_all['DEC'],G0_all['Z'],G0_all['weight'],G0_all['JK_ID'],V0_all)),dtype=[('RA',float),('DEC',float),('Z',float),('weight',float),('JK_ID',int),('V2D_%d_zslices'%Ns,float)])

def box_to_lightcone(G1,mask_IDs,zi,zf,is_pec_vel=True):
    G1_BB = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                G1_BB.append(G1[:,(0,1,2,3,4,5)]+np.array([i,j,k,0,0,0])*768.0)
    G1_BB = np.concatenate(G1_BB,axis=0)

    p0 = np.concatenate((np.mean(G1[:,(0,1,2)],axis=0),[0.0,0.0,0.0]))
    observer_frame = G1_BB - p0
    del G1_BB

    nn_tree = cKDTree(observer_frame[:,(0,1,2)])
    origin = np.array([0.0,0.0,0.0])
    ri = dcomov(zi)
    rf = dcomov(zf)
    #lc_z02 = nn_tree.query_ball_point(x=origin,r=ri,)
    lc_z = nn_tree.query_ball_point(x=origin,r=rf,)

    #LC_M = observer_frame[lc_z02]
    LC_L = observer_frame[lc_z]

    c = SkyCoord(x=LC_L[:,0], y=LC_L[:,1], z=LC_L[:,2], unit='Mpc', representation_type='cartesian')
    
    #add peculiar velocities to mocks!
    r_vec = np.array([c.x,c.y,c.z])
    r_vec_normed = (r_vec / np.linalg.norm(r_vec,axis=0)).T
    v_vec = np.zeros_like(r_vec_normed)
    
    if is_pec_vel:
        v_vec = np.array([LC_L[:,3],LC_L[:,4],LC_L[:,5]]).T

    v_proj_los = np.einsum('ij,ij->i',r_vec_normed,v_vec)#
    c.representation_type = 'spherical'

    #phi = c.ra.value[(c.distance.value > 100)&(c.distance.value < 150)]
    #theta = c.dec.value[(c.distance.value > 100)&(c.distance.value < 150)
    
    RA = c.ra.value#to(u.radian).value#[(c.dec.value > 0)&(c.dec.value < 5)]
    DEC = c.dec.value#
    r = c.distance.value#[(c.dec.value > 0)&(c.dec.value < 5)]
    LC = np.array([RA,DEC,r,LC_L[:,3],LC_L[:,4],LC_L[:,5]]).T

    NSIDE=1024
    ranmask = np.zeros(12*NSIDE**2)
    ranmask[mask_IDs.astype(int)] = 1.
    
    LC_chunk = LC[(LC[:,2]>=dcomov(zi))&(LC[:,2]<=dcomov(zf))]
    v_proj_chunk = v_proj_los[(LC[:,2]>=dcomov(zi))&(LC[:,2]<=dcomov(zf))]
    
    alpha = np.radians(LC_chunk[:,0])
    delta = np.radians(LC_chunk[:,1]) + np.pi/2.

    is_p = hp.ang2pix(NSIDE, theta=delta, phi=alpha, nest=True)
    
    mask_points = ranmask[is_p]==1.0
    masked_LC = LC_chunk[mask_points]
    v_proj_masked = v_proj_chunk[mask_points]
    
    z_i = z_interp(masked_LC[:,2])
    
    Hz = 100.0*np.sqrt(Om0*(1.0+z_i)**3 + Ol0)
    Hz_i = (1+z_i)/Hz
    r_obs = masked_LC[:,2] + v_proj_masked*Hz_i
    
    z_LC = z_interp(r_obs)
    z_in = (z_LC >=zi)&((z_LC <=zf))
    
    LC_S = np.array(list(zip(masked_LC[:,0][z_in],masked_LC[:,1][z_in],z_LC[z_in],np.full_like(z_LC[z_in],1.))),dtype=[('RA',float),('DEC',float),('Z',float),('weight',float)])
    return LC_S

def create_parfile_twopcf(M,z_l,z_h,redshift,theta):
    A = """data_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/V2D_Galaxy_{0}_z{1}_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_RA_DEC_Zobs_weight_mark1.hdf5
data_file_type = hdf5      # ascii/hdf5
random_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/R_LC_GR_L768_RA_Dec_z_Mh_IDcen_weight_LOWZ_z0.24_0.36_HOD_13.09683_13.97830_13.26748_0.04329_0.94784_MCMCfit.hdf5
random_file_type = hdf5    # ascii/hdf5

coord_system = equatorial            # equatorial/cartesian

ra_x_dataset_name = RA        # hdf5 dataset names
dec_y_dataset_name = DEC      # ra/dec/z for equatorial
z_z_dataset_name = Z          # x/y/z for cartesian
weight_dataset_name = mark1    # Name for weight dataset if needed
#jk_dataset_name = jk_region

use_weights = 0    # Boolean 0/1, assumes column 4 if reading ascii file
n_threads = 64       # Set to zero for automatic thread detection

#n_jk_regions = 0

omega_m = 0.2865
h = 0.6774
z_min = {7:.2f}
z_max = {8:.2f}

plot_monopole = 0     # Boolean 0/1
monopole_filename = none
monopole_output_type = hdf5
monopole_log_base = 1.3 # Set to 1 for linear, any float above 1.1 valid
monopole_min = 0.0
monopole_max = 100.0
monopole_n_bins = 30

plot_sigma_pi = 1        # Boolean 0/1
sigma_pi_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/twopcf_ouputs/TWOPCF_xi2D_unmarked_{0}_{1}_LC_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_logrp_0.5_50_10logrpbins_logpi_0.5_80_50pibins_z{7:.2f}_{8:.2f}.hdf5 #note
sigma_pi_output_type = hdf5
sigma_log_base = 1.58489319    # Set to 1 for linear, any float above 1.1 valid
sigma_min = 0.5
sigma_max = 50.0
sigma_n_bins = 10
pi_log_base = 1.10683377            # Set to 1 for linear, any float above 1.1 valid
pi_min = 0.5
pi_max = 80.0
pi_n_bins = 50

plot_s_mu = 0        # Boolean 0/1
s_mu_filename = s_mu.hdf5
s_mu_output_type = hdf5
s_log_base = 1.3      # Set to 1 for linear, any float above 1.1 valid
s_min = 0.0
s_max = 100.0
s_n_bins = 40
mu_n_bins = 50
    """.format(M,redshift,theta[0],theta[1],theta[2],theta[3],theta[4],z_l,z_h)
    
    B = """data_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/V2D_Galaxy_{0}_z{1}_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_RA_DEC_Zobs_weight_mark1.hdf5
data_file_type = hdf5      # ascii/hdf5
random_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/R_LC_GR_L768_RA_Dec_z_Mh_IDcen_weight_LOWZ_z0.24_0.36_HOD_13.09683_13.97830_13.26748_0.04329_0.94784_MCMCfit.hdf5
random_file_type = hdf5    # ascii/hdf5

coord_system = equatorial            # equatorial/cartesian

ra_x_dataset_name = RA        # hdf5 dataset names
dec_y_dataset_name = DEC      # ra/dec/z for equatorial
z_z_dataset_name = Z          # x/y/z for cartesian
weight_dataset_name = mark1    # Name for weight dataset if needed
#jk_dataset_name = jk_region

use_weights = 1    # Boolean 0/1, assumes column 4 if reading ascii file
n_threads = 64       # Set to zero for automatic thread detection

#n_jk_regions = 0

omega_m = 0.2865
h = 0.6774
z_min = {7:.2f}
z_max = {8:.2f}

plot_monopole = 0     # Boolean 0/1
monopole_filename = none
monopole_output_type = hdf5
monopole_log_base = 1.3 # Set to 1 for linear, any float above 1.1 valid
monopole_min = 0.0
monopole_max = 100.0
monopole_n_bins = 30

plot_sigma_pi = 1        # Boolean 0/1
sigma_pi_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/twopcf_ouputs/TWOPCF_xi2D_marked_{0}_{1}_LC_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_logrp_0.5_50_10logrpbins_logpi_0.5_80_50pibins_z{7:.2f}_{8:.2f}.hdf5 #note
sigma_pi_output_type = hdf5
sigma_log_base = 1.58489319    # Set to 1 for linear, any float above 1.1 valid
sigma_min = 0.5
sigma_max = 50.0
sigma_n_bins = 10
pi_log_base = 1.10683377            # Set to 1 for linear, any float above 1.1 valid
pi_min = 0.5
pi_max = 80.0
pi_n_bins = 50

plot_s_mu = 0        # Boolean 0/1
s_mu_filename = s_mu.hdf5
s_mu_output_type = hdf5
s_log_base = 1.3      # Set to 1 for linear, any float above 1.1 valid
s_min = 0.0
s_max = 100.0
s_n_bins = 40
mu_n_bins = 50
    """.format(M,redshift,theta[0],theta[1],theta[2],theta[3],theta[4],z_l,z_h)
    
    f1 = open('/cosma7/data/dp004/dc-armi2/HOD_mocks/twopcf/twopcf_unmarked_HOD_LC_z'+str(redshift)+'.ini', 'w')
    f1.seek(0)
    f1.write(A)
    f1.truncate()
    f1.close()
    f2 = open('/cosma7/data/dp004/dc-armi2/HOD_mocks/twopcf/twopcf_marked_HOD_LC_z'+str(redshift)+'.ini', 'w')
    f2.seek(0)
    f2.write(B)
    f2.truncate()
    f2.close()
                    
