import sys,os
import numpy as np
from scipy.special import erf
import time

Lbox=1024.0                                     # box size in Mpc/h
Mhalo_min=4.0E11  

def value_locate(rbin,value):
    vals = np.zeros_like(value)
    for i in range(len(vals)):
        vals[i] = np.where(rbin==np.max(rbin[(rbin - value[i]) < 0]))[0][0]
    return vals.astype(int)

def HOD_mock(theta,data_r):
    n_lines_r = len(data_r)
    #nhaloes = len(data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)])

    haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]
    haloes=haloes[np.argsort(haloes[:,0])[::-1]]
    
    logMmin, M1, M0, sigma, alpha = theta
    
    mock = []
    
    Ncen = 0.5*(1.0+erf((np.log10(haloes[:,0])-logMmin)/sigma))
    Nsat = np.zeros_like(Ncen)
    b1 = haloes[:,0] > 10**M0
    Nsat[b1] = Ncen[b1]*((haloes[:,0][b1]-(10**M0))/(10**M1))**alpha
    r1 = np.random.random(size=len(Ncen))
    r2 = np.random.poisson(lam=Nsat,size=len(Nsat))
    is_cen = np.zeros_like(Ncen,dtype=int)
    b2 = Ncen >=r1
    is_cen[b2] = 1
    b3 = (Ncen < r1)&(r2>0)
    is_cen[b3] = 1
    r2[b3] = r2[b3] - 1
    b4 = r2 > 0
    has_sat = np.zeros_like(Ncen,dtype=int)
    has_sat[b4] = 1
    halo_with_sat = haloes[b4]
    m=halo_with_sat[:,0]
    x=halo_with_sat[:,1]
    y=halo_with_sat[:,2]
    z=halo_with_sat[:,3]
    rv=halo_with_sat[:,4]
    rs=halo_with_sat[:,5]
    Nsat_perhalo = r2[b4]
    
    n_bins = 100
    for i,hi in enumerate(halo_with_sat):
        u = np.random.random(Nsat_perhalo[i])
        v = np.random.random(Nsat_perhalo[i])
        w = np.random.random(Nsat_perhalo[i])
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
        xyz_censat = np.vstack([hi[1:4],np.array([x,y,z]).T])
        mass = np.full(len(xyz_censat),hi[0])
        id_cen_sat = np.full_like(mass,-1)
        id_cen_sat[0] = 1
        halo_censat = np.vstack([xyz_censat.T,mass,id_cen_sat]).T
        
        mock.append(halo_censat)
    b5 = (is_cen == 1)&(~b4)
    halos_cen = np.vstack([haloes[:,(1,2,3,0)][b5].T,np.full(len(haloes[b5]),1.0)]).T
    mock.append(halos_cen)
    mock_cat = np.concatenate(mock)
    
    return mock_cat

def sat_occupation(m,xx,yy,zz,rv,rs,Nsat_in,n_bins = 100):
    u = np.random.random(Nsat_in)
    v = np.random.random(Nsat_in)
    w = np.random.random(Nsat_in)
    c = rv/rs
    if c < 3.0: c = 3.0
    norm1 = np.log(1.0+c)-c/(1.0+c)
    rbins = np.zeros((n_bins,2))
    rbins[:,0] = np.arange(0,1,1./n_bins)
    rbins[:,1] = np.log(1.0+c*rbins[:,0])-c*rbins[:,0]/(1.0+c*rbins[:,0])
    rbins[:,1] /= norm1
    locs = value_locate(rbins[:,1],u)
    rr = (rbins[:,0][locs]+0.5/n_bins)*rv/1000.0
    tt = np.cos(v*2.0 - 1.0)
    pp = w*2.0*np.pi  
    x= xx+rr*np.sin(tt)*np.cos(pp)
    y= yy+rr*np.sin(tt)*np.sin(pp)
    z= zz+rr*np.cos(tt)
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
    xyz_censat = np.vstack([np.array([xx,yy,zz]),np.array([x,y,z]).T])
    mass = np.full(len(xyz_censat),m)
    id_cen_sat = np.full_like(mass,-1)
    id_cen_sat[0] = 1
    halo_censat = np.vstack([xyz_censat.T,mass,id_cen_sat]).T
    return halo_censat
    