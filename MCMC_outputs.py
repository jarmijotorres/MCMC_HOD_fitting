import numpy as np
import h5py
from glob import glob
from hod_GR import *

B='5'
#============================= halo parent catalogue ============================#
infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/GR/box'+str(B)+'/31/Rockstar_M200c_GR_B'+str(B)+'_B1024_NP1024_S31.dat'
data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))
#============================= observation =======================================#
infile_data='/cosma7/data/dp004/dc-armi2/JK25_wp_logrp0.1_80_25bins_pimax100_z0.2_0.4.txt'
Y_data = np.loadtxt(infile_data)

labs=[r"$\log\ M_{min}$", r"$\log\ M_1$", r"$\log\ M_0$", r"$\sigma$",r"$\alpha$"]
filename = glob('/cosma7/data/dp004/dc-armi2/mcmc_*_Box'+B+'.h5')[0]

bck = h5py.File(filename,'r')
chain = bck['mcmc/chain']
likelihood = bck['mcmc/log_prob']

samples = np.array(chain).reshape((chain.shape[0]*chain.shape[1],chain.shape[2]))
flat_logprob = np.array(likelihood).flatten()

dens_kw = {'plot_density':False}
fig = corner.corner(
    samples,
    labels=labs,
                        range=None,
                        bins=20,
                       quantiles=None,
                        #truth_color='cmap',
                        plot_contours=False,
                       show_titles=True, title_kwargs={"fontsize": 16},figsize=(7,6),**dens_kw
)
plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/September2020/mcmcbest_posterior_box1.png',bbox_inches='tight')
#plt.tight_layout()
#plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/August2020/mcmcbest_posterior.png',bbox_inches='tight')

f,ax = plt.subplots(1,1,figsize=(7,6))
ax.hist(flat_logprob,range=(-30,-10),bins=20)
ax.set_xlabel(r'log prob.')
#ax.set_yscale(r'')
#ax.set_yscale('log')
plt.tight_layout()
#plt.savefig(,bbox_inches='tight')
plt.show()


for i in range(5):
    for j in range(i):
        p1 = np.array(chain).T[i]
        p2 = np.array(chain).T[j]
        f,ax = plt.subplots(1,1,figsize=(7,6))
        for k in range(len(p1)):
            ax.plot(p1[k][0],p2[k][0],'ko')
            ax.plot(p1[k][-1],p2[k][-1],'ks')
            ax.plot(p1[k],p2[k],'.-')
        ax.set_xlabel(labs[i])
        ax.set_ylabel(labs[j])
        plt.tight_layout()
        plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/September2020/param_steps_p%d_p%d_box_%s'%(i,j,B),bbox_inches='tight')
        plt.show()
        
t0 = samples[np.argmax(flat_logprob)]
#t0_per_box = []
t0_per_box.append(t0)

G1 = HOD_mock(t0,data_r)
Lbox = 1024.
n_sim = len(G1) / (1024.**3.)
Nsigma = len(Y_data) #number of bins on rp log binned
si = 0.1
sf = 80.0
sigma = np.logspace(np.log10(si),np.log10(sf),Nsigma+1)
pimax = 100 
pi = np.linspace(0,pimax,pimax+1)
dpi = np.diff(pi)[0]
s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
rp = 10**s_l
wp_obs = wp(G1,sigma,pi_max=pimax,num_threads=4,period=[Lbox,Lbox,Lbox])

wp_true_sim = wp_obs / rp

f,ax = plt.subplots(1,1,figsize=(7,6))
ax.plot(Y_data[:,0],Y_data[:,1]/Y_data[:,0],'ko')
ax.plot(Y_data[:,0],wp_true_sim,'b-')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

wp_boxes = []
wp_boxes.append(wp_true_sim)