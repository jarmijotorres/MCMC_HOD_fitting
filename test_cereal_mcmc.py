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
infile_data_wp = '/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/wp_'+M+'_box'+str(B)+'_test.txt'
infile_data_n = '/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/n_'+M+'_box'+str(B)+'_test.txt'

wp_data = np.loadtxt(infile_data_wp,usecols=(1,2,))
n_data = np.loadtxt(infile_data_n)

Y_data = (n_data,wp_data)

def model(theta):
    name_cat = '/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/cats/HOD_%.4lf_%.4lf_%.4lf_%.4lf_%.4lf.h5'%tuple(theta)
    if len(glob(name_cat)) == 1:
        h5_cat = h5py.File(name_cat,'r')
        G1 = h5_cat['mock_catalogue'][:,(0,1,2)]
        wp_true = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.4lf_%.4lf_%.4lf_%.4lf_%.4lf.npy'%tuple(theta))
    else:
        G1_data = HOD_mock(theta,data_r)
        h5_cat = h5py.File(name_cat, 'w')
        h5_cat.create_dataset('mock_catalogue',data=G1_data,compression='gzip')
        h5_cat.close()
        G1 = G1_data[:,(0,1,2)]        
        wp_true = wp_from_box(G1,Lbox = 1024.,Nsigma = 25)
        np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/temps/wps/wp_%.4lf_%.4lf_%.4lf_%.4lf_%.4lf.npy'%tuple(theta),wp_true)
        
    n_sim = len(G1) / (1024.**3.)
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
    
nwalkers = 10
prior_range = np.array([[12.5,13.5],
                        [13.5,14.5],
                        [12.5,13.5],
                        [0.3,0.9],
                        [0.5,1.5]])
ndim = len(prior_range) 
#p0 = [ np.array(initial)+ np.random.normal(0,0.1,size=ndim) for i in range(nwalkers)]
p0 = [ np.random.uniform(low=p[0],high=p[1],size=10) for p in prior_range]
p0 = np.array(p0).T
burnin_it = 0
prod_it = 10
stretch_p = 2.0

def main(p0,nwalkers,niter,ndim,lnprob,data):
    #print('Running mcmc...\n')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,moves=emcee.moves.StretchMove(a=stretch_p), backend=None,pool=None, args=np.array([Y_data]))

    #print("Running burn-in...")
    #p0, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=niter[0],progress=True)
    #if rank==0:
    #    np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burnin_'+str(stretch_p)+'strecth_'+str(nwalkers)+'wlk_'+str(burnin_it)+'_set2.npy',sampler.chain)
    #sampler.reset()

    #print("Running production...")
    pos, prob, state = sampler.run_mcmc(initial_state = p0, nsteps =niter[1],progress=True)

    return sampler, pos, prob, state

niter = [burnin_it,prod_it]

sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,Y_data)