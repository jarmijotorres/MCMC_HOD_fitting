import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import corner
import numpy as np
#
plt.style.use('../images/presentation.mplstyle')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['silver', 'gold', 'darkorange','chartreuse','deeppink','crimson','purple','royalblue','forestgreen','olive'])

labs=[r"$\log\ M_{min}$", r"$\log\ M_1$", r"$\log\ M_0$", r"$\sigma$",r"$\alpha$"]

burnin = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMCpost_burnin_1000it_20wlk_0.1An_0.9Awp.npy')
chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_GR_L768_2000it_20walkers_0.1An_0.9Awp.npy')

samples = chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2])
burnin_samples = burnin.reshape(burnin.shape[0]*burnin.shape[1],burnin.shape[2])

prior_range = np.array([np.min(burnin_samples,axis=0),np.max(burnin_samples,axis=0)]).T

dens_kw = {'plot_density':False,'plot_datapoints':True,'data_kwargs':{'color':'k','marker':'o','ms':3,'alpha':1.0}}

dens_kw2 = {'plot_density':False,'plot_datapoints':True,'data_kwargs':{'color':'r','marker':'o','ms':3,'alpha':1.0}}

fig = corner.corner(
    burnin_samples,
    labels=labs,
    bins=20,
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
    bins=20,
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
#ndim=5
#axes = np.array(fig.axes).reshape((ndim, ndim))
#for yi in range(ndim):
#    for xi in range(yi):
#        ax = axes[yi, xi]
#        ax.plot(theta_test[xi], theta_test[yi], "cs")

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
#plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/mcmc_post_0.5An_0.5Awp_0.056nerror_JKwperror_400burnin_2000prod.png',bbox_inches='tight')

theta_median = np.median(samples,axis=0)

Lbox=768
G0 = HOD_mock(theta_median,haloes,Lbox=Lbox)

b_steps = np.arange(0,1000,1)
c_steps = np.arange(1000,5000,1)

loop_color = np.random.random((20,3))

lklhd_samples = lklhd_chains.flatten()
lklhd_samples_burnin = lklhd_burnin.flatten()
lklhd_burnin = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMClklhd_burnin_1000it_20wlk_0.9An_0.1Awp.npy')
lklhd_chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_GR_L768_2000it_20walkers_0.9An_0.1Awp.npy')

f,ax = plt.subplots(1,1,figsize=(7,6))
for i in range(20):
    ax.plot(b_steps,lklhd_burnin.T[i],c=loop_color[i])
    ax.plot(c_steps,lklhd_chains.T[i],c=loop_color[i])
ax.vlines(1000,10,-500,linestyle='--',linewidth=2.0,zorder=10,label='End of burn-in')
ax.set_ylim(0,-200)    
ax.set_xlim(-10,3010)
ax.set_xlabel('MC step')
ax.set_ylabel('Likelihood')
ax.legend()
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2021/Likelihood_MCsteps_20walkers.pdf',bbox_inches='tight')
plt.show()

wp_sim = wp_from_box(G0,n_threads=16,Lbox=Lbox,Nsigma=20)
wp_rp = wp_data[:,1]/wp_data[:,0]
rp = wp_data[:,0]

f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].errorbar(rp,wp_rp,yerr=wp_data[:,2]/wp_data[:,0],fmt='ko')
ax[0].plot(rp,wp_sim,'r-')

ax[1].errorbar(rp,np.zeros_like(wp_rp),yerr=wp_data[:,2]/wp_rp,fmt='ko')
ax[1].plot(rp,(wp_sim-wp_rp)/wp_rp,'r-')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$w_p(r_p)/r_p$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$rp$ [Mpc $h^{-1}$]')
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.show()