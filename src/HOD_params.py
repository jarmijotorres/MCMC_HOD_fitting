import numpy as np
from scipy.special import erf
from hod_GR import HOD_mock
from binning_data import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
M='GR'
B='1'
infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'
data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))#parent halo data

Lbox=1024.0                                     # box size in Mpc/h
Mhalo_min=1e12

haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]

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
HMF_analytic = np.loadtxt('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/files/mVector_WMAP9SOCTinker10z03.txt')
Mrange = HMF_analytic[:,0]
HMF_interp = interp1d(np.log10(Mrange),np.log10(HMF_analytic[:,3]))

def HODHMF_function(Mi,logMmin, log_M1, log_M0, sigma_logM, alpha):
    N_from_function = HOD_analytic_vec(Mi,logMmin, log_M1, log_M0, sigma_logM, alpha)*10**HMF_interp(np.log10(Mi))
    return N_from_function

dlogM = np.diff(np.log10(Mrange))[0]
def galaxies_missing(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha):
    Mt = 3.162e12
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
        N_Grid[i,j] = galaxies_missing(Mrange,logMmin, log_M1, log_M0, sigma_logM, alpha)
        
        
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
   


##### use halo parent catalogue ############
#run on the halo catalogue from the cosmology_checks script

