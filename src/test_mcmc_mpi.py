import sys,os,time,h5py,yaml,emcee
from mpi4py import MPI
import numpy as np
from hod_GR import *
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
    input_dict = yaml.load(file)

t= time.time()  
M = input_dict['Model']
B = input_dict['Box']
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


#============================= halo parent catalogue ========================================#
#
data_r = np.loadtxt(halo_file,usecols=(2,33,8,9,10,5,6))

#============================== observable function to fit ==================================#

theta_test = np.array([13.15,14.2,13.3,0.6,1.0])
wp_data = np.loadtxt(observable_wp,usecols=(0,1,))
n_data = np.loadtxt(observable_n,usecols=(0,1,))

Y_data = (n_data,wp_data)

name_datacube = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/datacube/datacube_test_1.pkl'
Mock_datacube = Datacube(name_datacube)

def model(theta):
    if rank==1:
        bid = Mock_datacube.table['name_id'] == '%.5lf_%.5lf_%.5lf_%.5lf_%.5lf'%tuple(theta)
        bid_b = len(Mock_datacube.table[bid]) == 1
    elif rank!=1: 
        bid_b = None
    bid_b = comm.bcast(bid_b, root=1) # broadcast a boolean wether theta is already on the datacube
    if bid_b: #if so
        if rank==1:
            n_sim = np.squeeze(Mock_datacube.table['ngal'][bid]) #
            wp_true = np.squeeze(Mock_datacube.table['wp'][bid])
        elif rank!=1:
            n_sim = None
            wp_true = None
        n_sim = comm.bcast(n_sim,root=0) 
        wp_true = comm.bcast(wp_true,root=0)
    else:
        G1_data = HOD_mock(theta,data_r)
        G1 = G1_data[:,(0,1,2)]
        n_sim = len(G1) / (1024.**3.)
        wp_true = wp_from_box(G1,n_threads=8,Lbox = 1024.,Nsigma = 25)
    
        theta_job = {'name_id':'%.5lf_%.5lf_%.5lf_%.5lf_%.5lf'%tuple(theta),'params':theta,'ngal':n_sim,'wp':wp_true}
        print('I rank {} have this object {}, sending to root'.format(rank,theta))
        theta_job = comm.gather(theta_job,root=1)
        if rank == 1:
            print('I root (rank {}) gathered all objects {}'.format(rank,theta_job))
            for t in theta_job:
                Mock_datacube.add_row_dc(t)
        
    n_model = np.array([n_sim,n_sim*0.0])#fix error at 1% 
    wp_model = np.array([wp_true, wp_true*0.0]).T
    
    return None#(n_model,wp_model)
    
def lnlike(theta,ydata=Y_data): #chi square test
    ymodel = model(theta)
    Schis2 = chis(ymodel,ydata,A_n=A_n,A_wp=A_wp)
    LnLike = -0.5*(Schis2)
    return LnLike

def lnprior(theta):
    prior_range = np.array([[log_Mhalo_min,14.5],
                        [theta[0],15.5],
                        [theta[0],14.5],
                        [0.2,1.0],
                        [0.3,1.5]])
    b=np.zeros_like(theta,dtype=bool)
    for p,tp in enumerate(theta):
        if tp > prior_range[p][0] and tp < prior_range[p][1]:
            b[p] = 1
    
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
    

log_Mhalo_min = np.log10(4e11)
prior_range = np.array([[log_Mhalo_min,14.5],
                        [12.5,15.5],
                        [log_Mhalo_min,14.5],
                        [0.2,1.0],
                        [0.3,1.5]])

ndim = len(prior_range)

p0 = [ np.random.uniform(low=p[0],high=p[1],size=nwalkers) for p in prior_range]
p0 = np.array(p0).T

#filename = "/cosma7/data/dp004/dc-armi2/mcmc_runs/backends/mcmc_backend_stretch_lgprob_chain_"+str(nwalkers)+"wlk_"+str(burnin_it)+"burnin_"+str(prod_it)+"production_"+M+"_Box"+str(B)+".h5"
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
    
def main(p0,nwalkers,niter,ndim,lnprob,data):
    #print('Running mcmc...\n')
    cov_vec =  np.array([0.01,0.01,0.01,2e-3,2e-3])
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

niter = [burnin_it,prod_it]

sampler, p0, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)

if rank == 0:
    np.save(chain_file,sampler.chain)
    np.save(logProb_file,sampler.get_log_prob())
    e_t = time.time() - t
    #Mock_datacube.table.to_pickle(name_datacube)
    print('data saved on {}'.format(name_datacube))
    print('MCMC done in %.3lf seconds'%e_t)
    print("\n end of program.\n ")
