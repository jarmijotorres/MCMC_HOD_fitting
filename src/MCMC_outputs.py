import numpy as np
import h5py
from glob import glob
from hod_GR import *

from cycler import cycler 
#
mpl.rcParams['axes.prop_cycle'] = cycler(color=['silver', 'gold', 'darkorange','chartreuse','deeppink','crimson','purple','royalblue','forestgreen','olive'])#wiphala colours

B='5'
#============================= halo parent catalogue ============================#
infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/GR/box'+str(B)+'/31/Rockstar_M200c_GR_B'+str(B)+'_B1024_NP1024_S31.dat'
data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))
#============================= observation =======================================#
infile_data='/cosma7/data/dp004/dc-armi2/JK25_wp_logrp0.1_80_25bins_pimax100_z0.2_0.4.txt'
Y_data = np.loadtxt(infile_data)

labs=[r"$\log\ M_{min}$", r"$\log\ M_1$", r"$\log\ M_0$", r"$\sigma$",r"$\alpha$"]
filename = glob('/cosma7/data/dp004/dc-armi2/mcmc_*_Box'+B+'.h5')[0]
mpl.rcParams['axes.prop_cycle'] = cycler(color=['silver', 'gold', 'darkorange','red','purple','royalblue','forestgreen'])#wiphala colours

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
        #plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/September2020/param_steps_p%d_p%d_box_%s'%(i,j,B),bbox_inches='tight')
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


samples = chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2])
burnin_samples = burnin.reshape(burnin.shape[0]*burnin.shape[1],burnin.shape[2])

dens_kw = {'plot_density':False,'plot_datapoints':True,'data_kwargs':{'color':'k','marker':'o','ms':3,'alpha':1.0}}

dens_kw2 = {'plot_density':False,'plot_datapoints':True,'data_kwargs':{'color':'r','marker':'o','ms':3,'alpha':1.0}}

fig = corner.corner(
    burnin_samples,
    labels=labs,
    bins=40,
    #color='k',
    quantiles=None,
    range = prior_range,
    plot_contours=False,
    show_titles=True, 
    title_kwargs={"fontsize": 16},
    figsize=(7,6),
    hist_kwargs={'color':'r'},
    smooth=None,
    **dens_kw2)

fig2 = corner.corner(
    samples,
    labels=labs,
    bins=40,
    fig=fig,
    #color='k',
    quantiles=None,
    range = prior_range,
    plot_contours=False,
    show_titles=True, 
    title_kwargs={"fontsize": 16},
    figsize=(7,6),
    hist_kwargs={'color':'k'},
    smooth=None,
    **dens_kw)

axes = np.array(fig.axes).reshape((ndim, ndim))
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.plot(theta_test[xi], theta_test[yi], "cs")

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/mcmc_post_0.5An_0.5Awp_0.056nerror_JKwperror_400burnin_2000prod.png',bbox_inches='tight')



f,ax = plt.subplots(1,1,figsize=(7,6))
mc_step = np.arange(0,ns_chiwp.shape[1])
b_step = np.arange(0,ns_chiwp_b.shape[1])
for i in range(ns_chiwp.shape[0]):
    ax.plot(b_step,(ns_chiwp_b[i][:,0] - n_data[0])/n_data[0],c=co[i],linestyle='-')
    ax.plot(mc_step+b_step[-1],(ns_chiwp[i][:,0] - n_data[0])/n_data[0],c=co[i],linestyle='-')
ax.vlines(b_step[-1]+1,-1,3,linestyle='--',linewidth=2.0,zorder=10,label='End of burn-in stage')
ax.set_ylim(-1,3)
ax.set_xlim(-50,2400)
ax.set_xlabel('Step')
ax.set_ylabel(r'$\Delta n_s/n_s$')
ax.legend(loc=1,prop={'size':14})
plt.tight_layout()
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/Delta_ngal_step_0.5An_0.5Awp_0.056nerror_JKwperror_burnin+prod.png',bbox_inches='tight')
plt.show()

f,ax = plt.subplots(1,1,figsize=(7,6))
mc_step = np.arange(0,ns_chiwp.shape[1]+ns_chiwp_b.shape[1])
mc_chi_wp = np.concatenate([chi_wp_b,chi_wp],axis=1)
for i in range(20):
    ax.plot(mc_step,(mc_chi_wp[i]- chiwp_min),c=co[i],linestyle='-')
#ax.set_yscale('log')
ax.vlines(b_step[-1]+1,-50,1e3,linestyle='--',linewidth=2.0,zorder=10,label='End of burn-in stage')
#ax.hlines(100,-10,2500)
ax.set_xlabel('Step')
ax.set_ylabel(r'$\Delta \chi_{wp}^2$')
ax.legend(loc=1,prop={'size':14})
ax.set_ylim(-50,1e3)
ax.set_xlim(-10,2400)
plt.tight_layout()
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/chis_wp_0.5An_0.5Awp_0.056nerror_JKwperror_burnin+prod.png',bbox_inches='tight')
plt.show()

long_chain = np.concatenate([burnin,chains],axis=1)

#ns_chiwp.shape[1]
#for i in range(1):
for i in range(long_chain.shape[-1]):
    f,ax = plt.subplots(1,1,figsize=(7,6))
    for walker in range(long_chain.shape[0]):
        ax.plot(mc_step,long_chain[walker][:,i],c=co[walker],linestyle='-')
    ax.vlines(b_step[-1]+1,prior_range[i][0],prior_range[i][1],linestyle='--',linewidth=2.0,zorder=10,label='End of burn-in stage')
    ax.hlines(theta_test[i],-10,2400,linestyle='-',linewidth=2.0,zorder=10,label=labs[i]+'=%.2lf'%theta_test[i])
    ax.set_xlim(-10,2400)
    ax.set_ylim(prior_range[i][0],prior_range[i][1])
    ax.set_xlabel('Step')
    ax.set_ylabel(labs[i])
    ax.legend(loc=1,prop={'size':14})
    plt.tight_layout()
    plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/parameter_%d_steps_0.5An_0.5Awp_0.056nerror_JKwperror_burnin+prod.png'%i,bbox_inches='tight')
plt.show()

f,ax = plt.subplots(1,1,figsize=(7,6))
ax.errorbar(rp,np.zeros_like(wp_data[:,0]),yerr=wp_data[:,1]/wp_data[:,0],fmt='k.')
for i in [0,2,3,1]:
    ax.plot(rp,(wps[i]-wp_data[:,0])/wp_data[:,0],'-',label=r'$\Delta\chi_{w_p}^2=%.2lf$'%chiwp[ids[i]])
ax.set_xscale('log')
ax.set_ylim(-0.5,0.5)
ax.set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax.set_ylabel(r'$(w_p - w_p^{obs.})/w_p^{obs.}$')
#ax.set_yscale('log')
ax.legend(loc=2,prop={'size':14})
plt.tight_layout()
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/wp_chis_comp.png')
plt.show()