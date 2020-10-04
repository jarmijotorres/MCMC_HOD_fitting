import sys,os,time,h5py
import numpy as np
import emcee
from hod_GR import *
from tpcf_obs import *
from chi2 import *

t= time.time()
M = 'GR'#sys.argv[1]
B = 1#sys.argv[2]

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
    Schis2 = chis(ymodel,ydata,A_n=0.5,A_wp=0.5)
    LnLike = -0.5*(Schis2)
    return LnLike

def lnprior(theta):
    theta_priors = [13.,14.,13.,0.5,1.0]
    theta_delta_priors = [1.0,1.0,1.0,0.5,1.0]
    b=np.zeros_like(theta,dtype=bool)
    for tp,p in enumerate(theta):
        if p > theta_priors[tp]-theta_delta_priors[tp] and p < theta_priors[tp]+theta_delta_priors[tp]:
            b[tp] = 1
    
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
    
nwalkers = 6
prior_range = np.array([[12.5,13.5],
                        [13.5,14.5],
                        [12.5,13.5],
                        [0.3,0.9],
                        [0.5,1.5]])
ndim = len(prior_range) 
#p0 = [ np.array(initial)+ np.random.normal(0,0.1,size=ndim) for i in range(nwalkers)]
p0 = [ np.random.uniform(low=p[0],high=p[1],size=nwalkers) for p in prior_range]
p0 = np.array(p0).T
burnin_it = 0
prod_it = 3
stretch_p = 2.0

def main(p0,nwalkers,niter,ndim,lnprob,data):
    cov_vec =  np.array([0.01,0.01,0.01,1e-3,1e-3])
    Cov = np.identity(ndim)*cov_vec
    #print('Running mcmc...\n')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.GaussianMove(cov=Cov), backend=None,pool=None, args=np.array([Y_data]))

    #print("Running burn-in...")
    #p0, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=niter[0],progress=True)
    #if rank==0:
    #    np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burnin_'+str(stretch_p)+'strecth_'+str(nwalkers)+'wlk_'+str(burnin_it)+'_set2.npy',sampler.chain)
    #sampler.reset()

    #print("Running production...")
    pos, prob, state = sampler.run_mcmc(initial_state = p0, nsteps =niter[1],progress=True)

    return sampler, pos, prob, state

niter = [burnin_it,prod_it]

L_sampler = []
for i in range(5):
    sampler, p0, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)
    L_sampler.append(sampler.chain)
L_sampler = np.array(L_sampler)