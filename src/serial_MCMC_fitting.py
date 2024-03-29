import sys,os,time,h5py,yaml,emcee
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

yaml_name = '/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/params/param_ini.yaml'#sys.argv[1]
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
Mhalo_min= 1e12#imposed arbitrarily. Consult your mass resol.
Mlimit = 10**13.92604
data_r = np.loadtxt(halo_file,usecols=(2,33,8,9,10,5,6))#columns from rockstar file
halo_weights_name = '/cosma7/data/dp004/dc-armi2/haloes/weights_haloes_12.01_15.37Mrange_0.12_dlog10M_28_bins_GR_box1.dat'
if len(glob(halo_weights_name))  == 1:
    weights_haloes = np.loadtxt(halo_weights_name)
    haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]
else:
    haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]
    #haloes = haloes[np.argsort(haloes[:,0])[::-1]]
    halo_mass = haloes[:,0]
    weights_haloes = np.ones_like(halo_mass)
    weights_haloes[halo_mass<Mlimit] = weight_function(halo_mass[halo_mass<Mlimit])
#============================== observable function to fit ==================================#

theta_test = np.array([13.15,14.2,13.3,0.6,1.0])
wp_data = np.loadtxt(observable_wp,usecols=(0,1,))
n_data = np.loadtxt(observable_n,usecols=(0,1,))
n_data[1] = n_data[0]*0.05 #try with 5% number density error

Y_data = (n_data,wp_data)

#name_datacube = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/datacube/datacube_run1.pkl'
#Mock_datacube = Datacube(name_datacube)

def model(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1 = HOD_mock(theta,haloes)
        n_sim = np.sum(G1[:,4]) / (1024.**3.)
        wp_true = wp_from_box(G1,n_threads=16,Lbox = 1024.,Nsigma = 25)
        #
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta),theta)
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),np.array([n_sim]))
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),wp_true)
        
    n_model = np.array([n_sim,n_sim*0.05])#fix error at 1% 
    wp_model = np.array([wp_true, wp_true*0.05]).T
    
    return (n_model,wp_model)
    
def lnlike(theta,ydata=Y_data): #chi square test
    ymodel = model(theta)
    Schis2 = chis(ymodel,ydata,A_n=A_n,A_wp=A_wp)
    LnLike = -0.5*(Schis2)
    return LnLike


#============== define prior ranges ================#
log_Mhalo_min = np.log10(Mhalo_min)
mass_def_range = np.array([[12.8,13.8],
                        [13.8,14.6],
                        [12.8,13.9],
                        [0.3,0.7],
                        [0.8,1.3]])
ndim = len(mass_def_range)

#Initial state
p0 = np.zeros((nwalkers,ndim))
for i in range(nwalkers):
    p0[i,0] = np.random.uniform(mass_def_range[0,0],mass_def_range[0,1])
    p0[i,2] = np.random.uniform(p0[i,0],mass_def_range[2,1])
    p0[i,1] = np.random.uniform(p0[i,2],mass_def_range[1,1])
    p0[i,3] = np.random.uniform(mass_def_range[3,0],mass_def_range[3,1])
    p0[i,4] = np.random.uniform(mass_def_range[4,0],mass_def_range[4,1])

PR = np.array([[12.5,14],
             [13,15],
             [12.5,14.],
             [0.1,0.8],
             [0.7,1.4]])
    
def lnprior(theta,prior_range=PR):
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
    
#p0 = [ np.random.uniform(low=p[0],high=p[1],size=nwalkers) for p in prior_range]
#p0 = np.array(p0).T

#filename = "/cosma7/data/dp004/dc-armi2/mcmc_runs/backends/mcmc_backend_stretch_lgprob_chain_"+str(nwalkers)+"wlk_"+str(burnin_it)+"burnin_"+str(prod_it)+"production_"+M+"_Box"+str(B)+".h5"
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
    
def main(p0,nwalkers,niter,ndim,lnprob,data):
    #print('Running mcmc...\n')
    st = np.array([0.1,0.1,0.1,0.05,0.05])/4.0#array with stepsize for the parameter space
    cov_vec =  st**2
    Cov = np.identity(ndim)*cov_vec
    #with MPIPool() as pool:
    #    if not pool.is_master():
    #        pool.wait()
    #        sys.exit(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.GaussianMove(cov=Cov),
                                    backend=None,pool=None, args=np.array([Y_data]))

        ##print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=niter[0],progress=True)
    #if rank==0:
    #np.save(burnin_file,sampler.chain)
    #np.save(burnin_logProb_file,sampler.get_log_prob())
    sampler.reset()

        #print("Running production...")
    pos, prob, state = sampler.run_mcmc(initial_state = p0, nsteps =niter[1],progress=True,store=True)
    return sampler, pos, prob, state


#=================== Run main function ==========================#
niter = [burnin_it,prod_it]
sampler, p0, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)

#if rank == 0:
#np.save(chain_file,sampler.chain)
#np.save(logProb_file,sampler.get_log_prob())
e_t = time.time() - t
#Mock_datacube.table.to_pickle(name_datacube)
#print('data saved on {}'.format(name_datacube))
print('MCMC done in %.3lf seconds'%e_t)
print("\n end of program.\n ")
