import sys,os,time,h5py,yaml,emcee
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from mpi4py import MPI
import numpy as np
from hod import *
from tpcf_obs import *
from chi2 import *
from datacube import *
from schwimmbad import MPIPool
from pandas import DataFrame,read_pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

yaml_name = sys.argv[1]
#Load yaml dictionary
with open(yaml_name) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    input_dict = yaml.load(file,Loader=yaml.FullLoader)

t= time.time()  
M = input_dict['Model']
Lbox = input_dict['Lbox']
A_n = input_dict['A_n']
A_wp = input_dict['A_wp']
nwalkers = input_dict['N_walkers']
burnin_it = input_dict['burnin_iterations']
prod_it = input_dict['production_iterations']
burnin_file = input_dict['burnin_file']
burnin_logProb_file = input_dict['burnin_logProb_file']
chain_file = input_dict['chain_file']
logProb_file = input_dict['logProb_file']
halo_file = input_dict['halo_file']
observable_wp = input_dict['observable_wp'] 
observable_n = input_dict['observable_n']
#if rank == 0:
#    print('this is it %d \n'%int(it))


#================ halo parent catalogue ==================#
haloes_sim = h5py.File(halo_file,'r')
haloes = haloes_sim['data']
sh_table = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/subhaloes/Haloes_MG-Gadget_GR_z0.3_L1536_ID_subhalotable_CM_M200c_12.0.npy',allow_pickle=True)
#HMF_1 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_ELEPHANT_GR_z0.0_L1024_dlogM0.15.dat')
#HMF_2 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_GR_z0.3_L768_dlogM0.15.dat')
#w1 = weight_function(haloes['M200'],HMF_1,HMF_2)
wp_cols = np.loadtxt(observable_wp,usecols=(0,1,2))
rp = wp_cols[:,0]
wp_data = wp_cols[:,1]
wp_err = 3*wp_cols[:,2]#np.sqrt((wp_cols[:,2]**2 + (0.10*wp_cols[:,1])**2))
wp_obs = np.array([rp,wp_data/rp,wp_err/rp]).T
n_data = np.loadtxt(observable_n)
#n_data[1] = n_data[0]*0.15
# additional error has been added, just for modelling
Y_data = (n_data,wp_obs[:,(1,2)])#discard bins at the begining and at the end


#name_datacube = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/datacube/datacube_run1.pkl'
#Mock_datacube = Datacube(name_datacube)

def model(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1 = HOD_mock_subhaloes(theta,haloes,sh_table,Lbox=Lbox,weights_haloes=None)
        n_sim = np.sum(G1[:,4]) / (Lbox**3.)
        wp_true = wp_from_box(G1,n_threads=16,Lbox = Lbox,Nsigma = 20)
            #
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta),theta)
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),np.array([n_sim]))
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),wp_true)
        
    n_model = np.array([n_sim,n_sim*0.1])#try 10-15% error for models number density
    wp_model = np.array([wp_true, wp_true*0.1]).T
    
    return (n_model,wp_model)
    
def lnlike(theta,ydata=Y_data): #chi square test
    ymodel = model(theta)
    Schis2 = chis(ymodel,ydata,A_n=A_n,A_wp=A_wp)
    LnLike = -0.5*(Schis2)
    return LnLike


#============== define prior ranges ================#
#log_Mhalo_min = np.log10(Mhalo_min)
#mass_def_range = np.array([[12.5,13.5],
#                        [13.8,14.6],
#                        [12.5,13.5],
#                        [0.3,0.7],
#                        [0.8,1.3]])
mass_def_range = np.array([[12.8,13.5],
                        [13.5,14.5],
                        [13.1,14.0],
                        [0.1,0.5],
                        [0.8,1.0]])

ndim=5
#Initial state
p0 = np.zeros((nwalkers,ndim))
for i in range(nwalkers):
    p0[i,0] = np.random.uniform(mass_def_range[0,0],mass_def_range[0,1])
    p0[i,1] = np.random.uniform(mass_def_range[1,0],mass_def_range[1,1])
    p0[i,2] = np.random.uniform(mass_def_range[2,0],mass_def_range[2,1])
    p0[i,3] = np.random.uniform(mass_def_range[3,0],mass_def_range[3,1])
    p0[i,4] = np.random.uniform(mass_def_range[4,0],mass_def_range[4,1])

PR = np.array([[12.5,14.5],
             [13.5,14.5],
             [13.0,14.5],
             [0.0,0.9],
             [0.7,1.5]])

    
def lnprior(theta,prior_range=PR):
    b=np.zeros_like(theta,dtype=bool)
    for p,tp in enumerate(theta):
        if tp > prior_range[p][0] and tp < prior_range[p][1]:
            b[p] = 1
    if theta[2] > theta[1]:
        b[2] = 0
    if theta[0] > theta[1]:
        b[0] = 0
    elif theta[0] > theta[2]:
        b[0] = 0
        
    if b.all():
        return 0
    else:
        return -np.inf
    
def lnprob(theta,ydata=Y_data):
    ymodel = model(theta)
    lp = lnprior(theta) 
    if lp != 0:
        return -np.inf
    return lp + lnlike(theta,ydata)
    
def main(p0,nwalkers,niter,ndim,lnprob,data):
    #print('Running mcmc...\n')
    st = np.array([0.02,0.02,0.02,0.02,0.02])/3#array with stepsize for the parameter space
    cov_vec =  st**2
    Cov = np.identity(ndim)*cov_vec
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.GaussianMove(cov=Cov), backend=None,pool=pool, args=np.array([Y_data]))

        ##print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=niter[0],progress=True)
        if rank==0:
            np.save(burnin_file,sampler.chain)
            np.save(burnin_logProb_file,sampler.get_log_prob())
        sampler.reset()

        #print("Running production...")
        pos, prob, state = sampler.run_mcmc(initial_state = p0, nsteps =niter[1],progress=True)
    return sampler, pos, prob, state

#=================== Run main function ==========================#
niter = [burnin_it,prod_it]
sampler, p0, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)

if rank == 0:
    np.save(chain_file,sampler.chain)
    np.save(logProb_file,sampler.get_log_prob())
    e_t = time.time() - t
    #Mock_datacube.table.to_pickle(name_datacube)
    #print('data saved on {}'.format(name_datacube))
    print('MCMC done in %.3lf seconds'%e_t)
    print("\n end of program.\n ")
