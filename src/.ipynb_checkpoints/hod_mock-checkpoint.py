import sys,os
import numpy as np
from scipy.special import erf
from astropy.table import Table
from scipy.interpolate import interp1d
#import multiprocessing
#from pyina.launchers import MpiPool, MpiScatter
from mpi4py import MPI
from schwimmbad import MPIPool,JoblibPool
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Box_HMF = np.load('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/files/HMF_box1.npy') 
log_n_log_Mh = np.loadtxt('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/files/mVector_WMAP9SOCTinker10z03.txt')#Analytic halo mass function (Tinker 2010 differential)
Tinker10_HMF_differential = interp1d(np.log10(log_n_log_Mh[:,0]),np.log10(log_n_log_Mh[:,3]))
Mlimit = 10**Box_HMF[:,1][Box_HMF[:,1] < 14.0][-1]

class HOD_mock():
    def __init__(self,parent_haloes,Mhalo_min,box_size):
        self.Lbox = box_size
        self.Mhalo_min = Mhalo_min
        self.halo_parent_catalogue = parent_haloes
        self.weights_haloes = np.ones(len(self.halo_parent_catalogue))
        
    def weight_HMF(self,Mlimit=Mlimit):
        self.Mhalo = self.halo_parent_catalogue[:,0]
        weight_flag = self.Mhalo < Mlimit
        self.weights_haloes[weight_flag] = weight_function(self.Mhalo[weight_flag],Mlimit=Mlimit)
        no_weight_flag =self.Mhalo < self.Mhalo_min
        self.weights_haloes[no_weight_flag] = 0.0
        
    def centrals(self,theta):
        self.HOD_parameters = theta
        self.logMmin = self.HOD_parameters[0]
        self.M1 = self.HOD_parameters[1]
        self.M0 = self.HOD_parameters[2]
        self.sigma = self.HOD_parameters[3]
        self.alpha = self.HOD_parameters[4]
        self.Nhaloes = len(self.halo_parent_catalogue)
        
        New_haloes = np.vstack([self.halo_parent_catalogue.T,self.weights_haloes]).T
        Ncen = 0.5*(1.0+erf((np.log10(self.Mhalo)-self.logMmin)/self.sigma))
        r1 = np.random.random(size=len(Ncen))
        if (theta == 0.0).any():
            b2 = Ncen >= r1
            New_haloes = New_haloes[b2]
            haloes_cen = np.vstack([New_haloes[:,(1,2,3,0,-1)].T,np.full(len(New_haloes),1.0)]).T
            self.centrals_catalogue = haloes_cen
        
        Nsat = np.zeros_like(Ncen)
        b1 = self.Mhalo > 10**self.M0
        Nsat[b1] = Ncen[b1]*((self.Mhalo[b1]-(10**self.M0))/(10**self.M1))**self.alpha
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
        halo_with_sat = New_haloes[b4]
        Nsat_perhalo = r2[b4]
        self.Nsat_max = Nsat_perhalo.max()
        b5 = (is_cen == 1)&(~b4)
        haloes_cen = np.vstack([New_haloes[:,(1,2,3,0,-1)][b5].T,np.full(len(New_haloes[b5]),1.0)]).T
        self.centrals_catalogue = haloes_cen
        haloes_cen_Nsat = np.vstack([halo_with_sat.T,Nsat_perhalo]).T
        return haloes_cen_Nsat

    def sat_occupation(self,hi,n_bins = 100):
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
        bx1 = x > self.Lbox
        by1 = y > self.Lbox
        bz1 = z > self.Lbox
        bx2 = x < 0
        by2 = y < 0
        bz2 = z < 0
        x[bx1] = x[bx1] - self.Lbox
        y[by1] = y[by1] - self.Lbox
        z[bz1] = z[bz1] - self.Lbox
        x[bx2] = x[bx2] + self.Lbox
        y[by2] = y[by2] + self.Lbox
        z[bz2] = z[bz2] + self.Lbox
        xyz_censat = np.vstack([np.array([hi[1],hi[2],hi[3]]),np.array([x,y,z]).T])
        mass = np.full(len(xyz_censat),hi[0])
        w_halo = np.full(len(xyz_censat),hi[6])
        id_cen_sat = np.full_like(mass,-1)
        id_cen_sat[0] = 1
        halo_censat = np.vstack([xyz_censat.T,mass,w_halo,id_cen_sat]).T
        
        halo_buffer = np.zeros((self.Nsat_max+1,halo_censat.shape[1]))
        for im,hm in enumerate(halo_censat):
            halo_buffer[im] = hm

        return halo_buffer
    
    def create_HOD_mock_catalogue(self,theta):
        Haloes_with_sat = self.centrals(theta)
        #pool = multiprocessing.Pool(processes=4)#
        Buffer_sat = np.zeros((len(Haloes_with_sat),self.Nsat_max+1,6))
        with JoblibPool(28) as pool:
            Buffer_sat[:] = pool.map(self.sat_occupation,Haloes_with_sat)
        #pool.close() 
        #pool.join()
        sat_list = []
        for Hs in Buffer_sat:
            hs = Hs[~((Hs==0.0).all(axis=1))]
            sat_list.append(hs)
        sat_cat = np.concatenate(sat_list,axis=0)
        self.satellites_catalogue = sat_cat
        mock_catalogue = np.append(self.centrals_catalogue,self.satellites_catalogue,axis=0)
        return mock_catalogue
            

def weight_function(M,Mlimit=Mlimit):
    Mbins=np.linspace(Box_HMF[0,0],np.log10(Mlimit),len(Box_HMF[Box_HMF[:,1]<np.log10(Mlimit)])+2)
    M_in_bin = np.digitize(x=np.log10(M),bins=Mbins)
    weight_list = 10**Tinker10_HMF_differential(Box_HMF[:,2][Box_HMF[:,1]<=np.log10(Mlimit)])/Box_HMF[:,3][Box_HMF[:,1]<=np.log10(Mlimit)]
    weight_mass = weight_list[M_in_bin - 1]
    return weight_mass

def value_locate(rbin,value):
    vals = np.zeros_like(value)
    for i in range(len(vals)):
        vals[i] = np.where(rbin==np.max(rbin[(rbin - value[i]) < 0]))[0][0]
    return vals.astype(int)
