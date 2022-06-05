import numpy as np
import time,h5py,sys
from glob import glob
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Mod='GR'
Lbox = 768
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+Mod+'_z0.3_L%d_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'%Lbox,'r')

haloes = np.array(haloes_table['MainHaloes'])
subhaloes = np.array(haloes_table['SubHaloes'])

chain_file = sys.argv[1]
chain_name = chain_file.split('/')[-1].split('.npy')[0]
Nsigma = 13

def compute_n_wp(theta):
    G1 = HOD_mock_subhaloes(theta,haloes=haloes,subhaloes=subhaloes,Lbox=Lbox,weights_haloes=None)
    n_sim = len(G1) / (Lbox**3.)
    wp_sim = wp_from_box(G1,n_threads=28,Lbox = Lbox,Nsigma = Nsigma)

    return n_sim,wp_sim

chain_walkers = np.load(chain_file)
samples_1D = chain_walkers[rank]
ns = np.zeros(len(samples_1D))
wps = np.zeros((len(samples_1D),Nsigma))
for i in range(len(samples_1D)):
    nmod,wpmod = compute_n_wp(samples_1D[i])
    ns[i] = nmod
    wps[i] = wpmod

comm.barrier()
n_all = comm.gather(ns,root=0)
wp_all = comm.gather(wps,root=0)

out_dir = '/cosma7/data/dp004/dc-armi2/mcmc_runs/GR_An0.5_Awp0.5/'

if rank == 0:
    n_all = np.array(n_all)
    wp_all = np.array(wp_all)
    np.save(out_dir+'ngal/'+chain_name+'test_ngal.npy',n_all)
    np.save(out_dir+'wp/'+chain_name+'test_wp.npy',wp_all)