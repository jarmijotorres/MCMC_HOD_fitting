import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import time

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

def HOD_mock(theta,haloes,Lbox,weights_haloes=None):
    """
    Method to create a mock catalogue from a halo parent catalogue and a set of parameters.
    parameters:
        theta: array of 5 parameters for the HOD function. These 5 parameters are: [log M_min, log_M1, log_M0, sigma, alhpa]. For the functional form of the HOD, please consult Manera et al. 2013
        haloes: table with the halo catalogue information. It should be read using haloes['M200'] for the halo mass, haloes['pos'] for the 3D position, haloes['R200'] for the virial radius and haloes['c200'] for concentration
    return:
        mock_cat: catalogue of HOD galaxies including position and host halo mass.
    """
    

    logMmin, logM1, logM0, sigma, alpha = theta #unpack parameters
        
    if weights_haloes is None:
        weights_haloes = np.ones_like(haloes['M200'])
        #New_haloes = np.vstack([haloes,np.full_like(haloes[:,0],1.)]).T
    #else:
        #New_haloes = np.vstack([haloes.T,weights_haloes]).T
    Ncen = 0.5*(1.0+erf((np.log10(haloes['M200'])-logMmin)/sigma))#function for Number of centrals
#function for Number of centrals (between 0 and 1)
    r1 = np.random.random(size=len(Ncen))#MC random to populate haloes with centrals
    if (theta == 0.0).any():
        b2 = Ncen >=r1
        New_haloes = haloes[b2]
        weights_haloes = weights_haloes[b2]
        haloes_cen = np.array([New_haloes['pos'][:,0],New_haloes['pos'][:,1],New_haloes['pos'][:,2],New_haloes['M200'],weights_haloes,np.full(len(New_haloes),1.0)]).T
        return haloes_cen
    
    Nsat = np.zeros_like(Ncen)
    b1 = haloes['M200'] > 10**logM0#boolean 1: haloes that may contain satellites depending on M0
    Nsat[b1] = Ncen[b1]*((haloes['M200'][b1]-(10**logM0))/(10**logM1))**alpha# function for number of satellites
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
    halo_with_sat = haloes[b4]# new variable to iterate over haloes with satellites only
    weights_hsat = weights_haloes[b4]
    Nsat_perhalo = r2[b4]
    #haloes_cen_Nsat = np.array([halo_with_sat['M200'],halo_with_sat['pos'][:,0],halo_with_sat['pos'][:,1],halo_with_sat['pos'][:,2],halo_with_sat['R200'],halo_with_sat['c200'],Nsat_perhalo]).T# Input for sat_occupation routine
    #return haloes_cen_Nsa

    n_bins = 100#bins to replicate NFW profile
    #iteration over haloes with satellites. This may be potetially time consuming as the vector is around 10000 entries large.
    mock = []#list to gather the catalogue
    for i,hi in enumerate(halo_with_sat):
        u = np.random.random(Nsat_perhalo[i])#position in halo coordinates
        v = np.random.random(Nsat_perhalo[i])
        w = np.random.random(Nsat_perhalo[i])
        c = hi[3] #concentration
        #if c < 3.0: c = 3.0#set like this on Baojiu's code, consult!
        norm1 = np.log(1.0+c)-c/(1.0+c)#normalization constant 1
        rbins = np.zeros((n_bins,2))
        rbins[:,0] = np.arange(0,1,1./n_bins)
        rbins[:,1] = np.log(1.0+c*rbins[:,0])-c*rbins[:,0]/(1.0+c*rbins[:,0])# bins on the radial direction with the NFW profile given its concentration
        rbins[:,1] /= norm1
        locs = value_locate(rbins[:,1],u)
        rr = (rbins[:,0][locs]+0.5/n_bins)*hi[2]#/1000.0# transformation to spherical coordinates
        tt = np.cos(v*2.0 - 1.0)
        pp = w*2.0*np.pi  
        x= hi[1][0]+rr*np.sin(tt)*np.cos(pp)#transformation to cartesians box coordinates
        y= hi[1][1]+rr*np.sin(tt)*np.sin(pp)
        z= hi[1][2]+rr*np.cos(tt)
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
        xyz_censat = np.vstack([hi[1],np.array([x,y,z]).T])
        mass = np.full(len(xyz_censat),hi[0])
        w_halo = np.full(len(xyz_censat),weights_hsat[i])#hi[6] is the weight
        id_cen_sat = np.full_like(mass,-1)
        id_cen_sat[0] = 1
        halo_censat = np.vstack([xyz_censat.T,mass,w_halo,id_cen_sat]).T# all haloes with satellites
        mock.append(halo_censat)
    b5 = (is_cen == 1)&(~b4)#boolean 5: all haloes frome above - haloes with satellites
    New_haloes = haloes[b5]
    haloes_cen = np.vstack([New_haloes['pos'][:,0],New_haloes['pos'][:,1],New_haloes['pos'][:,2],New_haloes['M200'],weights_haloes[b5],np.full(len(weights_haloes[b5]),1.0)]).T
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
    
    