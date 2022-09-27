import sys,os,h5py,yaml,emcee,time
from pandas import DataFrame,read_pickle
import numpy as np
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *
from mpi4py import MPI
from schwimmbad import MPIPool

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

yaml_name = sys.argv[1]
#Load yaml dictionary
with open(yaml_name) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    input_dict = yaml.load(file,Loader=yaml.FullLoader)

#t= time.time()  
M = input_dict['Model']
Lbox = input_dict['Lbox']
A_n = input_dict['A_n']
A_wp = input_dict['A_wp']
nwalkers = input_dict['N_walkers']
N_batches = input_dict['N_batches']#burnin_it = input_dict['burnin_iterations']
prod_it = input_dict['production_iterations']
#burnin_file = input_dict['burnin_file']
#burnin_logProb_file = input_dict['burnin_logProb_file']
chain_file = input_dict['chain_file']
logProb_file = input_dict['logProb_file']
halo_file = input_dict['halo_file']
observable_wp = input_dict['observable_wp'] 
observable_n = input_dict['observable_n']
cov_name = input_dict['cov_matrix']
prev_state = input_dict['prev_state']
z_snap = input_dict['z_snap']
target = input_dict['target']

#================ halo parent catalogue ==================#
haloes_table = h5py.File(halo_file,'r')
haloes = np.array(haloes_table['MainHaloes'])
subhaloes = np.array(haloes_table['SubHaloes'])

wp_cols = np.loadtxt(observable_wp,usecols=(0,1,2))
cov_matrix = np.loadtxt(cov_name)
#cov_matrix *= 4
rp = wp_cols[:,0]
wp_data = wp_cols[:,1]
wp_err = wp_cols[:,2]#3*wp_cols[:,2]
wp_obs = np.array([rp,wp_data/rp,wp_err/rp]).T
n_data = np.loadtxt(observable_n)
n_obs = n_data
Y_data = (n_obs,wp_obs[:,(1,2)])

def model(theta):
    G1 = HOD_mock_subhaloes(theta,haloes=haloes,subhaloes=subhaloes,Lbox=Lbox,weights_haloes=None)
    n_sim = G1.shape[0] / (Lbox**3.)
    wp_true = wp_from_box(G1,n_threads=16,Lbox = Lbox,z_snap=z_snap,Nsigma = 13)
        
    n_model = np.array([n_sim,n_sim*0.03])
    wp_model = np.array([wp_true, wp_true*0.05]).T
    
    return (n_model,wp_model)
    
def lnlike(theta,ydata=Y_data): #chi square test
    #
    ymodel = model(theta)
    Schis2 = chis(y_sim=ymodel,y_obs=ydata,cov_matrix=cov_matrix,A_n=A_n,A_wp=A_wp)
    LnLike = -0.5*(Schis2)

    return LnLike


PR = np.array([[12.7,14.0],
             [12.7,14.8],
             [12.7,14.0],
             [0.0,0.6],
             [0.7,1.6]])

def lnprior(theta,prior_range=PR):
    b=np.zeros_like(theta,dtype=bool)
    for p,tp in enumerate(theta):
        if tp > prior_range[p][0] and tp < prior_range[p][1]:
            b[p] = 1
#    if theta[2] > theta[1]:
#        b[2] = 0
    if theta[0] > theta[1]:
        b[0] = 0
    elif theta[0] > theta[2]:
        b[0] = 0
    elif theta[1] < theta[2] + np.log10(5):
        b[0] = 0
        
    if b.all():
        return 0
    else:
        return -np.inf
    
def lnprob(theta,ydata=Y_data):
#    ymodel = model(theta)
    lp = lnprior(theta)
    if lp != 0:
        return -np.inf
    #name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.npy'%tuple(theta)
    #if len(glob(name_par)) == 1:
    #    lnlk = np.load(name_par)
    #else:
    lnlk = lnlike(theta,ydata)
        #np.save(name_par,lnlk)
    lpf = lp + lnlk
    return lpf
    
def main(cycle,p0,nwalkers,niter,ndim,lnprob,Y_data):
    batch_cycle = cycle%10
    #print('Running mcmc...\n')
    #if batch_cycle < 2:
    #    step = np.array([0.03,0.03,0.03,0.03,0.03])/3.
    #elif (batch_cycle < 5) and (batch_cycle >= 2):
    #    step = np.array([0.02,0.02,0.02,0.02,0.02])/3.
    #else:
    step = np.array([0.01,0.01,0.01,0.01,0.01])/3.
        
    cov_vec =  step**2
    Cov = np.identity(ndim)*cov_vec
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        if cycle > 0:
            cycle +=1
            
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.GaussianMove(cov=Cov), backend=None,pool=pool, args=np.array([Y_data]))
        pos = p0
        for i in range(cycle,N_batches):
            pos, prob, state = sampler.run_mcmc(initial_state = pos, nsteps =niter,progress=True)
            np.save(chain_file+'_batch_%d'%i+'.npy',sampler.chain)
            np.save(logProb_file+'_batch_%d'%i+'.npy',sampler.get_log_prob())
            #print("Acceptance rate of cycle {0}: {1} \n".format(i,sampler.acceptance_fraction))
            #
            sampler.reset()
            
        #np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/test/auto_corr_time_'+M+'_cycle_%d'%i+'.npy',sampler.get_autocorr_time()) 
    return sampler, pos, prob, state

#=================== Run main function ==========================#
mass_def_range = np.array([[12.8,13.2],
                        [13.8,14.2],
                        [13.0,13.4],
                        [0.0,0.2],
                        [0.9,1.2]])

ndim=5
#Initial state
if prev_state is None:
    p0 = np.zeros((nwalkers,ndim))
    for i in range(nwalkers):
        p0[i,0] = np.random.uniform(mass_def_range[0,0],mass_def_range[0,1])
        p0[i,2] = np.random.uniform(p0[i,0],mass_def_range[2,1])
        p0[i,1] = np.random.uniform(p0[i,2]+np.log10(5),mass_def_range[1,1])
        p0[i,3] = np.random.uniform(mass_def_range[3,0],mass_def_range[3,1])
        p0[i,4] = np.random.uniform(mass_def_range[4,0],mass_def_range[4,1])
    current_cycle = 0

else:
    current_chain = np.load(prev_state)
    p0 = current_chain[:,-1,:]
    name_chain = prev_state.split('/')
    where_cy = name_chain[-1].split('_')
    current_cycle = int(where_cy[-1].split('.')[0])

if rank == 0:
    t = time.time()
    
sampler, p0, prob, state = main(cycle = current_cycle,p0 = p0,nwalkers=nwalkers,niter = prod_it,ndim = ndim,lnprob = lnprob,Y_data = Y_data)


if rank == 0:
    #np.save(chain_file,sampler.chain)
    #np.save(logProb_file,sampler.get_log_prob())
    e_t = time.time() - t
    #Mock_datacube.table.to_pickle(name_datacube)
    #print('data saved on {}'.format(name_datacube))
    print('MCMC done in %.3lf seconds'%e_t)
    print("\n end of program.\n ")
