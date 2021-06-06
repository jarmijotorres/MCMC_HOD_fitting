import numpy as np
from scipy.special import erf
from hod import HOD_mock
from binning_data import *
from scipy.integrate import quad,trapz
from scipy.interpolate import interp1d
import h5py
#M='GR'
#B='1'
#infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'
#data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))#parent halo data

Lbox=768.0                                     # box size in Mpc/h
#Mhalo_min=1e12
#haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]

haloes_sim = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L768_M200c_pos_R200_c200_logMhalomin_11.2.hdf5','r')
halo = haloes_sim['data']

#define HOD
def HOD_analytic(Mi,logMmin, log_M1, log_M0, sigma_logM, alpha):
    N_cen = 0.5*(1+erf((np.log10(Mi)- logMmin)/(sigma_logM)))
    if np.log10(Mi) > log_M0:
        N_sat = N_cen*((Mi-10**log_M0)/(10**log_M1))**alpha
    else: 
        N_sat = 0.0
    return N_cen + N_sat

HOD_analytic_vec = np.vectorize(HOD_analytic)

#load mass function
#HMF_analytic = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/analytic/Bocquet16_z0.0_dn_dlogM_0.01_logM11_15.5.txt')
HMF_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_GR_z0.0_L768_dlogM0.12.dat')
Mrange_data = HMF_data[:,2]
HMF_interp = interp1d(np.log10(Mrange_data),np.log10(HMF_data[:,2]))

def HODHMF_function(Mi,logMmin, log_M1, log_M0, sigma_logM, alpha):
    N_from_function = HOD_analytic_vec(Mi,logMmin, log_M1, log_M0, sigma_logM, alpha)*10**HMF_interp(np.log10(Mi))
    return N_from_function

dlogM = np.diff(np.log10(Mrange))[0]
def galaxies_fraction(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha):
    Mt = 7.2e11#20M_part LR sim (L1536)
    #I = quad(HODHMF_function,Mlow,Mhigh,args=(logMmin, log_M1, log_M0, sigma_logM, alpha))
    #I_chunk = quad(HODHMF_function,Mlow,1e12,args=(logMmin, log_M1, log_M0, sigma_logM, alpha))
    I = trapz(HODHMF_function(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha),np.log10(Mrange))*dlogM
    I_chunk = trapz(HODHMF_function(Mrange[Mrange<Mt],logMmin, log_M1, log_M0, sigma_logM, alpha),np.log10(Mrange[Mrange<Mt]))*dlogM
    R = 1 - (I-I_chunk)/I
    return R

#target:
theta = np.array([13.15, 14.2,  13.3,  0.6,  1.0 ])
N_hmfhod = 1024**3*HMF_analytic[:,3]*(HOD_analytic_vec(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha))
I = trapz(N_hmfhod,np.log10(Mrange))
N_hmfhod /= I

sigma_vector = np.arange(0.1,0.9,0.02)#sigma
Mmin_vector = np.round(np.linspace(12.5,13.3,len(sigma_vector)),decimals=3)
Param_grid = np.meshgrid(sigma_vector, Mmin_vector, sparse=False, indexing='ij')
#np.arange(0.7,1.5,0.1)#alpha
#param_range = np.arange(0.0,1.0,0.1)#sigma
    
N_Grid = np.zeros((len(sigma_vector),len(Mmin_vector)))
for i in range(len(sigma_vector)):
    for j in range(len(Mmin_vector)):
        theta = np.array([Mmin_vector[j], 14.0,  13.3,  sigma_vector[i],  1.0 ])
        logMmin, log_M1, log_M0, sigma_logM, alpha = theta
        N_Grid[i,j] = galaxies_fraction(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha)
        
        
f,ax = plt.subplots(1,1,figsize=(7.5,6))
clb = ax.imshow(N_Grid.T,origin='lower',norm=colors.LogNorm(vmin=1e-3, vmax=1.0),extent=[sigma_vector[0],sigma_vector[-1],Mmin_vector[0],Mmin_vector[-1]],cmap='Greys')
CS = ax.contour(Param_grid[0],Param_grid[1],N_Grid,[1e-2,2e-2,5e-2,1e-1,2e-1,5e-1],cmap='Greys_r',vmin=1e-2, vmax=0.2)
ax.clabel(CS, CS.levels, inline=True, fmt=r'%.2lf ', fontsize=14)
c1 = plt.colorbar(clb)
c1.set_label('Missing galaxies fraction')
#ax.set_xscale('log')
ax.set_ylabel(r'$\log M_{min}$')
ax.set_xlabel(r'$\sigma_{\log M_{min}}$')
plt.tight_layout()
plt.savefig('/cosma/home/dp004/dc-armi2/sdsswork/figures/January2021/Missing_galaxies_sigma_logMmin.svg',bbox_inches='tight')
plt.show()
   

f,ax = plt.subplots(1,1,figsize=(7,6))
theta_grid = np.array([[13.0, 0.0,  0.0,  1.0,  0.0 ],
                [12.6, 0.0,  0.0,  0.8,  0.0 ],
                [11.6, 0.0,  0.0,  0.0,  0.0 ]])
for theta in theta_grid:
    logMmin, log_M1, log_M0, sigma_logM, alpha = theta
    N_hmfhod = 1536**3*HODHMF_function(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha)
    I = trapz(N_hmfhod,np.log10(Mrange))
    N_hmfhod /= I
    f = galaxies_fraction(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha)
    ax.plot(Mrange,N_hmfhod,'-',label=r'$\log M_{min} = %.1lf,\ \sigma=%.1lf,\ f_{sub} = %.2lf$'%(logMmin,sigma_logM,f))
ax.vlines(7.2e11,-0.5,2.4,linestyle='--')
ax.set_xscale('log')
ax.set_xlabel(r'$M_{200c}\ [M_{\odot} h^{-1}]$')
ax.set_ylabel(r'$N/N_{Total}$')
ax.set_ylim(-.1,2.1)
ax.legend(loc=1)
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/April2021/N_Ntotal_HODs_fraction_subres_haloes_2.pdf',bbox_inches='tight')
plt.show()


deltalogM = 1e-3
Mrange = 10**np.arange(np.log10(HMF_data_L768[2,2])+deltalogM,np.log10(HMF_data_L768[-1,2]),deltalogM)
Mrange2 = 10**np.arange(np.log10(HMF_data_L1536[2,2])+deltalogM,np.log10(HMF_data_L1536[-1,2]),deltalogM)

theta = np.array([12.6, 0.0,  0.0,  0.5,  0.0 ])
logMmin, log_M1, log_M0, sigma_logM, alpha = theta

HMF_interp = interp1d(np.log10(HMF_data_L768[2:,2]),np.log10(HMF_data_L768[2:,3]))
N_hmfhod = 10**HMF_interp(np.log10(Mrange2))*(HOD_analytic_vec(Mrange2,logMmin, log_M1, log_M0, sigma_logM, alpha))
I = trapz(N_hmfhod,np.log10(Mrange2))
HH_L768 = HODHMF_function(HMF_data_L768[2:,2],logMmin, log_M1, log_M0, sigma_logM, alpha)

HMF_interp = interp1d(np.log10(HMF_data_L1536[2:,2]),np.log10(HMF_data_L1536[2:,3]))
N_hmfhod2 = 10**HMF_interp(np.log10(Mrange2))*(HOD_analytic_vec(Mrange2,logMmin, log_M1, log_M0, sigma_logM, alpha))
I2 = trapz(N_hmfhod2,np.log10(Mrange2))
HH_L1536 = HODHMF_function(HMF_data_L1536[2:,2],logMmin, log_M1, log_M0, sigma_logM, alpha)

