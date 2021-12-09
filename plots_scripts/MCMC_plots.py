import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import corner
import numpy as np

#

mpl.rcParams['axes.prop_cycle'] = cycler(color=['silver', 'gold', 'darkorange','chartreuse','deeppink','crimson','purple','royalblue','forestgreen','olive'])

labs=[r"$\log\ M_{min}$", r"$\log\ M_1$", r"$\log\ M_0$", r"$\sigma$",r"$\alpha$"]

chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_F5_L768_800it_56walkers_0.5An_0.5Awp_target_LOWZ_z0.2_0.4_err_1sigma_sim_subhaloes.npy')


samples = chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2])
#burnin_samples = burnin.reshape(burnin.shape[0]*burnin.shape[1],burnin.shape[2])


#prior_range = np.array([np.min(burnin_samples,axis=0),np.max(burnin_samples,axis=0)]).T
prior_range = np.array([[12.5,13.5],
                        [13.0,14.6],
                        [12.5,13.8],
                        [0.0,0.6],
                        [0.7,1.2]])
ndim=5
theta_max = np.zeros(ndim)
p_bin = np.zeros((ndim,2))
for i in range(ndim):
    tm,_ = np.histogram(samples[:,i],bins=13,range=prior_range[i])
    bm = 0.5*(_[:-1] + _[1:])
    p_bin[i][0] = _[np.argmax(tm)]
    p_bin[i][1] = _[np.argmax(tm)+1]
    theta_max[i] = bm[np.argmax(tm)]

dens_kw = {'plot_density':True,'plot_datapoints':True,'data_kwargs':{'color':'k','marker':'o','ms':3,'alpha':1.0}}

dens_kw2 = {'plot_density':False,'plot_datapoints':True,'data_kwargs':{'color':'r','marker':'o','ms':3,'alpha':1.0}}

fig = corner.corner(
    burnin_samples,
    labels=labs,
    bins=13,
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
    bins=13,
    fig=fig,
    #color='k',
    quantiles=None,
    range = prior_range,
    plot_contours=True,
    show_titles=True, 
    title_kwargs={"fontsize": 16},
    figsize=(7,6),
    hist_kwargs={'color':'k'},
    smooth=0.7,
    **dens_kw)

axes = np.array(fig.axes).reshape((ndim, ndim))
for i in range(ndim):
    ax = axes[i, i]
    #ax.axvline(theta_median[i], color="g")
    #ax.axvline(theta_mean[i], color="b")
    ax.axvline(theta_max[i], color="b")

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
#plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/October2020/mcmc_post_0.5An_0.5Awp_0.056nerror_JKwperror_400burnin_2000prod.png',bbox_inches='tight')

b_steps = np.arange(0,1000,1)
c_steps = np.arange(1000,5000,1)

loop_color = np.random.random((20,3))

lklhd_chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_GR_L768_600it_56walkers_0.5An_0.5Awp_target_LOWZ_z0.2_0.4_err_23igma_sim_subhaloes.npy')
lklhd_burnin = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/burn_in/MCMClklhd_burnin_100it_20wlk_0.1An_0.9Awp_target_LOWZ_z0.2_0.4_err_2sigma.npy')

lklhd_samples = lklhd_chains.flatten()
max_L = np.where(lklhd_samples == lklhd_samples.max())[0][0]
theta_max = samples[max_L]

lklhd_samples_burnin = lklhd_burnin.flatten()

f,ax = plt.subplots(1,1,figsize=(7,6))
i=0
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
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.show()
