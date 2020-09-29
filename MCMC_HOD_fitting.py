import sys,os,time,h5py
from mpi4py import MPI
import numpy as np
#from Corrfunc.theory import wp
#from halotools.mock_observables import wp
import emcee
from hod_GR import *
from tpcf_obs import *
from chi2 import *
from schwimmbad import MPIPool

t= time.time()  
M = sys.argv[1]
B = sys.argv[2]

A_n = 0.5
A_wp = 0.5
st = 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#============================= halo parent catalogue ========================================#
infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'

data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))

#============================== observable function to fit ==================================#
#must include errors
#infile_data='/cosma7/data/dp004/dc-armi2/JK25_wp_logrp0.1_80_25bins_pimax100_z0.2_0.4.txt'

theta_test = np.array([13.15,14.2,13.3,0.6,1.0])

infile_data_wp = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/test/wp_13.15000_14.20000_13.30000_0.60000_1.00000.txt'
infile_data_n = '/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/test/ngal_13.15000_14.20000_13.30000_0.60000_1.00000.txt'

wp_data = np.loadtxt(infile_data_wp)
n_data = np.loadtxt(infile_data_n)

Y_data = (np.array([n_data,n_data*0.01]),np.array([wp_data,wp_data*0.01]).T)


def model(theta):
    name_par = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta)
    if len(glob(name_par)) == 1:
        n_sim = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
        wp_true = np.loadtxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta))
    else:
        G1_data = HOD_mock(theta,data_r)
        G1 = G1_data[:,(0,1,2)]
        n_sim = len(G1) / (1024.**3.)
        wp_true = wp_from_box(G1,n_threads=16,Lbox = 1024.,Nsigma = 25)
        #
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/params/HOD_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.par'%tuple(theta),theta)
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/ns/ngal_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),np.array([n_sim]))
        np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.txt'%tuple(theta),wp_true)
        
    n_model = np.array([n_sim,n_sim*0.01])#fix error at 1% 
    wp_model = np.array([wp_true, wp_true*0.01]).T
    
    return (n_model,wp_model)
    
    
def lnlike(theta,ydata=Y_data): #chi square test
    ymodel = model(theta)
    Schis2 = chis(ymodel,ydata,A_n=A_n,A_wp=A_wp)
    LnLike = -0.5*(Schis2)
    return LnLike

def lnprior(theta):
    prior_range = np.array([[11.0,15.0],
                    [12.0,16.0],
                    [11.0,15.0],
                    [0.1,1.2],
                    [0.2,1.6]])
    #theta_delta_priors = [2.0,2.0,2.0,1.0,1.0]
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
    

prior_range = np.array([[12.0,14.3],
                        [13.0,15.0],
                        [12.0,14.3],
                        [0.3,0.9],
                        [0.5,1.3]])
nwalkers = 10

#prior_range = np.array([theta_test - 3.2*(theta_test*0.05),theta_test + 3.2*(theta_test*0.05)]).T
#prior_range = np.array([[11.0,15.0],
#                [12.0,16.0],
#                [11.0,15.0],
#                [0.2,1.0],
#                [0.4,1.6]])

ndim = len(prior_range)
#p = theta_test

p0 = [ np.random.uniform(low=p[0],high=p[1],size=nwalkers) for p in prior_range]
#p0 = [ np.random.normal(i,i*0.05,size=10) for i in p]
p0 = np.array(p0).T
#save walkers initial state
np.savetxt('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/walkers_p0/p0_'+str(nwalkers)+'walkers_set'+str(st)+'.ini',p0)

burnin_it = 0
prod_it = 1000
#stretch_p = 2.0

#filename = "/cosma7/data/dp004/dc-armi2/mcmc_runs/backends/mcmc_backend_stretch_lgprob_chain_"+str(nwalkers)+"wlk_"+str(burnin_it)+"burnin_"+str(prod_it)+"production_"+M+"_Box"+str(B)+".h5"
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
    
def main(p0,nwalkers,niter,ndim,lnprob,data):
    #print('Running mcmc...\n')
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.StretchMove(a=2.0), backend=None,pool=pool, args=np.array([Y_data]))

        ##print("Running burn-in...")
        #p0, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=niter[0],progress=True)
        #if rank==0:
            #np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/burnin_'+str(nwalkers)+'wlk_'+str(burnin_it)+'_set'+str(st)+'_'+str(A_n)+'An_'+str(A_wp)+'Awp_set'+str(st)+'.npy',sampler.chain)
        #sampler.reset()

        #print("Running production...")
        pos, prob, state = sampler.run_mcmc(initial_state = p0, nsteps =niter[1],progress=True)

    return sampler, pos, prob, state


niter = [burnin_it,prod_it]

sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)
#
samples = sampler.flatchain
t0 = samples[np.argmax(sampler.flatlnprobability)]
log_probs = sampler.get_log_prob()
if rank == 0:
    np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/HOD_mcmcpost_chains_'+str(M)+'_box'+str(B)+'_'+str(prod_it)+'iter_'+str(nwalkers)+'walkers_chi2_'+str(A_n)+'An_'+str(A_wp)+'Awp_set'+str(st)+'.npy',sampler.chain)
    np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/HOD_mcmcpost_logprob_'+str(M)+'_box'+str(B)+'_'+str(prod_it)+'iter_'+str(nwalkers)+'walkers_chi2_'+str(A_n)+'An_'+str(A_wp)+'Awp_set'+str(st)+'.npy',log_probs)
    
    #np.save('/cosma7/data/dp004/dc-armi2/HOD_mcmc_newmove_box'+str(B)+'_chains_'+str(niter)+'iter_'+str(nwalkers)+'walkers.npy',sampler.chain)
    e_t = time.time() - t
    print('%.3lf seconds'%e_t)
    print('file save in:\n')
    print('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/HOD_mcmcpost_chains_'+str(M)+'_box'+str(B)+'_'+str(prod_it)+'iter_'+str(nwalkers)+'walkers_chi2_'+str(A_n)+'An_'+str(A_wp)+'Awp_set'+str(st)+'.npy')
    print('initial step in:\n')
    print(p0)
    print('best parameters in '+str(M)+' box'+str(B)+': \n')
    print(t0)

#print('file saved in: ')
#print('/cosma7/data/dp004/dc-armi2/HOD_fitting_...')


    print("\n end of program.\n ")