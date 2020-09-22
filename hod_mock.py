import sys,os
import numpy as np
from scipy.special import erf
from astropy.table import Table

class HOD_mock:
    def __init__(self,parent_haloes,Mhalo_min,box_size):
        self.box_size = box_size
        self.Mhalo_threshold = Mhalo_min
        self.halo_parent_catalogue = parent_haloes
        
    def HOD_centrals(self,theta):
        self.parameters = theta
        self.logMmin = self.parameters[0]
        self.M1 = self.parameters[1]
        self.M0 = self.parameters[2]
        self.sigma = self.parameters[3]
        self.alpha = self.parameters[4]
        data_r = self.halo_parent_catalogue
        n_lines_r = len(data_r)
        haloes = data_r[(data_r[:,1]<0.0)&(data_r[:,0]>self.Mhalo_threshold)][:,(0,2,3,4,5,6)]
        
        haloes=haloes[np.argsort(haloes[:,0])[::-1]]
        self.Nhaloes = len(haloes)
        
        Ncen = 0.5*(1.0+erf((np.log10(haloes[:,0])-self.logMmin)/self.sigma))
        Nsat = np.zeros_like(Ncen)
        b1 = haloes[:,0] > 10**self.M0
        Nsat[b1] = Ncen[b1]*((haloes[:,0][b1]-(10**self.M0))/(10**self.M1))**self.alpha
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
        Nsat_perhalo = r2[b4]
        
        self.central_masses = haloes[:,0]
        self.central_pos = haloes[:,(1,2,3)]
        self.central_Rv = haloes[:,4]
        self.central_Rs = haloes[:,5]
        self.central_Nsat = r2
        
        #return central information + the number of satellites per halo
        #self.centrals_info = np.vstack([haloes[b2|b3].T,r2[b2|b3]]).T
        centrals_info = np.vstack([haloes[b2|b3].T,r2[b2|b3]]).T
        return centrals_info      
    
    def create_mock_catalogue(self,centrals_info):
        add_satellites = HOD_satellites_population(centrals_info[:,6].astype(int),centrals_info[:,1],centrals_info[:,2],centrals_info[:,3],centrals_info[:,4],centrals_info[:,5])
        self.mock_table = Table(data=centrals_info,names=['Mass_halo','X','Y','Z','R_vir','R_s','N_sat'])
        self.mock_table['cen_sat_pos'] = add_satellites
        cat_from_table = np.concatenate(self.mock_table['cen_sat_pos'])
        return cat_from_table
        
def HOD_satellites_population(Ni,xi,yi,zi,Rvi,Rsi,n_bins=100,Lbox=1024.):
    u = np.random.random(Ni)
    v = np.random.random(Ni)
    w = np.random.random(Ni)
    c = Rvi / Rsi
    if c < 3.0: c = 3.0
    norm1 = np.log(1.0+c)-c/(1.0+c)
    rbins = np.zeros((n_bins,2))
    rbins[:,0] = np.arange(0,1,1./n_bins)
    rbins[:,1] = np.log(1.0+c*rbins[:,0])-c*rbins[:,0]/(1.0+c*rbins[:,0])
    rbins[:,1] /= norm1
    locs = value_locate(rbins[:,1],u)        
    rr = (rbins[:,0][locs]+0.5/n_bins)*Rvi/1000.0
    tt = np.cos(v*2.0 - 1.0)
    pp = w*2.0*np.pi
    x= rr*np.sin(tt)*np.cos(pp)
    y= rr*np.sin(tt)*np.sin(pp)
    z= rr*np.cos(tt)
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
    xyz_censat = np.vstack([[xi,yi,zi],np.array([x,y,z]).T])
    return xyz_censat
HOD_satellites_population = np.vectorize(HOD_satellites_population,otypes=[object])

def value_locate(rbin,value):
    vals = np.zeros_like(value)
    for i in range(len(vals)):
        vals[i] = np.where(rbin==np.max(rbin[(rbin - value[i]) < 0]))[0][0]
    return vals.astype(int)