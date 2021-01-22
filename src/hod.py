import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import time

Lbox=1024.0                                     #Box size in Mpc/h
log_n_log_Mh = np.loadtxt('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/files/mVector_WMAP9SOCTinker10z03.txt')#Analytic halo mass function (Tinker 2010 differential)
Tinker10_HMF_differential = interp1d(np.log10(log_n_log_Mh[:,0]),np.log10(log_n_log_Mh[:,3])) #HMF to compute weights for simulation haloes #Y-axis is a log quantity 
Box_HMF = np.load('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/files/HMF_box1.npy') #simulation HMF 
Mhalo_min= 1e12
#Create cumulative disttibution function. Needed to interpolate
cum_HMF = np.cumsum(Box_HMF[::-1,3])*np.diff(Box_HMF[:,0])[0]
Box_HMF = np.vstack((Box_HMF.T,cum_HMF[::-1])).T


Mlimit = 10**Box_HMF[:,1][Box_HMF[:,1] < 14.0][-1]
def weight_function(M,Mlimit=Mlimit):
    Mbins=np.linspace(Box_HMF[0,0],np.log10(Mlimit),len(Box_HMF[Box_HMF[:,1]<np.log10(Mlimit)])+2)
    M_in_bin = np.digitize(x=np.log10(M),bins=Mbins)
    weight_list = 10**Tinker10_HMF_differential(Box_HMF[:,2][Box_HMF[:,1]<=np.log10(Mlimit)])/Box_HMF[:,3][Box_HMF[:,1]<=np.log10(Mlimit)]
    weight_mass = weight_list[M_in_bin - 1]
    return weight_mass

#diff_interpolation = interp1d(Box_HMF[:,2],np.log10(Box_HMF[:,3]))#global function in this module as Tinker10_HMF_cumulative

#more global variables


def value_locate(rbin,value):
    """
    Function to locate values in different bins. Linear transformation to obtain rbin distributed to obtain a NFW profile.
    parameters:
        rbin: profile x-axis where values should be allocated.
        value: r value to allocate on rbin.
    return:
        vals: bin positions of values in the rbin array.
    """
    vals = np.zeros_like(value)
    for i in range(len(vals)):
        vals[i] = np.where(rbin==np.max(rbin[(rbin - value[i]) < 0]))[0][0]
    return vals.astype(int)

def HOD_mock(theta,haloes,weights_haloes=None):
    """
    Method to create a mock catalogue from a halo parent catalogue and a set of parameters.
    parameters:
        theta: array of 5 parameters for the HOD function. These 5 parameters are: [log M_min, log_M1, log_M0, sigma, alhpa]. For the functional form of the HOD, please consult Manera et al. 2013
        data_r: halo catalogue containing all the haloes. This is the output of a halo finder code (Rockstar in this case).
        weights_haloes: weights for halo masses (see above) with same dimension as the halo catalogue. Default is None.
    return:
        mock_cat: catalogue of HOD galaxies including position and host halo mass.
    """

    logMmin, M1, M0, sigma, alpha = theta #unpack parameters
        
    if weights_haloes is None:
        New_haloes = np.vstack([haloes.T,np.full_like(haloes[:,0],1.)]).T
    else:
        New_haloes = np.vstack([haloes.T,weights_haloes]).T
    Ncen = 0.5*(1.0+erf((np.log10(New_haloes[:,0])-logMmin)/sigma))#function for Number of centrals
#function for Number of centrals (between 0 and 1)

    if (theta == 0.0).any():
        r1 = np.random.random(size=len(Ncen))
        b2 = Ncen >=r1
        New_haloes = New_haloes[b2]
        haloes_cen = np.vstack([New_haloes[:,(1,2,3,0,-1)].T,np.full(len(New_haloes),1.0)]).T
        return haloes_cen
    
    Nsat = np.zeros_like(Ncen)
    b1 = New_haloes[:,0] > 10**M0#boolean 1: haloes that may contain satellites depending on M0
    Nsat[b1] = Ncen[b1]*((New_haloes[:,0][b1]-(10**M0))/(10**M1))**alpha# function for number of satellites
    r1 = np.random.random(size=len(Ncen))#MC random to populate haloes with centrals
    r2 = np.random.poisson(lam=Nsat,size=len(Nsat))# Poisson random for the number of satellites
    is_cen = np.zeros_like(Ncen,dtype=int)# flag: halo in catalogue has central
    b2 = Ncen >=r1#boolean 2: MH step to populate halo with central
    is_cen[b2] = 1
    b3 = (Ncen < r1)&(r2>0)#boolean 3: halo has no central but will produce satellites (in this case 1 satellite becomes the central)
    is_cen[b3] = 1
    r2[b3] = r2[b3] - 1# for those haloes the number of satellites decrease by 1
    b4 = r2 > 0# boolean 4: halo with central has satellites
    has_sat = np.zeros_like(Ncen,dtype=int)#flag: halo has satellites
    has_sat[b4] = 1
    halo_with_sat = New_haloes[b4]# new variable to iterate over haloes with satellites only
    Nsat_perhalo = r2[b4]
    haloes_cen_Nsat = np.vstack([halo_with_sat.T,Nsat_perhalo]).T# Input for sat_occupation routine
    #return haloes_cen_Nsa

    n_bins = 100#bins to replicate NFW profile
    #iteration over haloes with satellites. This may be potetially time consuming as the vector is around 10000 entries large.
    mock = []#list to gather the catalogue
    for i,hi in enumerate(halo_with_sat):
        u = np.random.random(Nsat_perhalo[i])#position in halo coordinates
        v = np.random.random(Nsat_perhalo[i])
        w = np.random.random(Nsat_perhalo[i])
        c = hi[4] / hi[5] #concentration
        if c < 3.0: c = 3.0#set like this on Baojiu's code, consult!
        norm1 = np.log(1.0+c)-c/(1.0+c)#normalization constant 1
        rbins = np.zeros((n_bins,2))
        rbins[:,0] = np.arange(0,1,1./n_bins)
        rbins[:,1] = np.log(1.0+c*rbins[:,0])-c*rbins[:,0]/(1.0+c*rbins[:,0])# bins on the radial direction with the NFW profile given its concentration
        rbins[:,1] /= norm1
        locs = value_locate(rbins[:,1],u)
        rr = (rbins[:,0][locs]+0.5/n_bins)*hi[4]/1000.0# transformation to spherical coordinates
        tt = np.cos(v*2.0 - 1.0)
        pp = w*2.0*np.pi  
        x= hi[1]+rr*np.sin(tt)*np.cos(pp)#transformation to cartesians box coordinates
        y= hi[2]+rr*np.sin(tt)*np.sin(pp)
        z= hi[3]+rr*np.cos(tt)
        bx1 = x > Lbox#boundary conditions
        by1 = y > Lbox
        bz1 = z > Lbox
        bx2 = x < 0
        by2 = y < 0
        bz2 = z < 0
        x[bx1] = x[bx1] - Lbox
        y[by1] = y[by1] - Lbox
        z[bz1] = z[bz1] - Lbox
        x[bx2] = x[bx2] + Lbox
        y[by2] = y[by2] + Lbox
        z[bz2] = z[bz2] + Lbox
        xyz_censat = np.vstack([hi[1:4],np.array([x,y,z]).T])
        mass = np.full(len(xyz_censat),hi[0])
        w_halo = np.full(len(xyz_censat),hi[6])#hi[6] is the weight
        id_cen_sat = np.full_like(mass,-1)
        id_cen_sat[0] = 1
        halo_censat = np.vstack([xyz_censat.T,mass,w_halo,id_cen_sat]).T# all haloes with satellites
        mock.append(halo_censat)
    b5 = (is_cen == 1)&(~b4)#boolean 5: all haloes frome above + haloes with satellites
    haloes_cen = np.vstack([New_haloes[:,(1,2,3,0,-1)][b5].T,np.full(len(New_haloes[b5]),1.0)]).T
    mock.append(haloes_cen)
    mock_cat = np.concatenate(mock)# array containing the catalogue
    return mock_cat

#function to populate with satellites. Needs to be vectorized properly before using it. Not currently in use.
def sat_occupation(hi,n_bins = 100,Ns=16):
    u = np.random.random(int(hi[7]))
    v = np.random.random(int(hi[7]))
    w = np.random.random(int(hi[7]))
    c = hi[4] / hi[5]
    if c < 3.0: c = 3.0
    norm1 = np.log(1.0+c)-c/(1.0+c)
    rbins = np.zeros((n_bins,2))
    rbins[:,0] = np.arange(0,1,1./n_bins)
    rbins[:,1] = np.log(1.0+c*rbins[:,0])-c*rbins[:,0]/(1.0+c*rbins[:,0])
    rbins[:,1] /= norm1
    locs = value_locate(rbins[:,1],u)
    rr = (rbins[:,0][locs]+0.5/n_bins)*hi[4]/1000.0
    tt = np.cos(v*2.0 - 1.0)
    pp = w*2.0*np.pi  
    x= hi[1]+rr*np.sin(tt)*np.cos(pp)
    y= hi[2]+rr*np.sin(tt)*np.sin(pp)
    z= hi[3]+rr*np.cos(tt)
    bx1 = x > Lbox
    by1 = y > Lbox
    bz1 = z > Lbox
    bx2 = x < 0
    by2 = y < 0
    bz2 = z < 0
    x[bx1] = x[bx1] - Lbox
    y[by1] = y[by1] - Lbox
    z[bz1] = z[bz1] - Lbox
    x[bx2] = x[bx2] + Lbox
    y[by2] = y[by2] + Lbox
    z[bz2] = z[bz2] + Lbox
    xyz_censat = np.vstack([np.array([hi[1],hi[2],hi[3]]),np.array([x,y,z]).T])
    mass = np.full(len(xyz_censat),hi[0])
    id_cen_sat = np.full_like(mass,-1)
    id_cen_sat[0] = 1
    halo_censat = np.vstack([xyz_censat.T,mass,id_cen_sat]).T
    
    halo_buffer = np.zeros((Ns+1,halo_censat.shape[1]))
    for im,hm in enumerate(halo_censat):
        halo_buffer[im] = hm
    
    return halo_buffer
    
    