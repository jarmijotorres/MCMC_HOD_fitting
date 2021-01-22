import numpy as np
import time,sys,os
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod_mock import *


halo_file = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/GR/box1/31/Rockstar_M200c_GR_B1_B1024_NP1024_S31.dat'
Mhalo_min = 1e12
data_r = np.loadtxt(halo_file,usecols=(2,33,8,9,10,5,6))
haloes = data_r[(data_r[:,1]<0)&(data_r[:,0]>Mhalo_min)][:,(0,2,3,4,5,6)]

theta = np.array([13.1,14.2,13.3,0.5,1.0])

#multithread version
t=time.time()
Halo_cat = HOD_mock(haloes,Mhalo_min,box_size=1024.)
Halo_cat.weight_HMF()
cen_with_sat = Halo_cat.centrals(theta)
HOD1 = Halo_cat.create_HOD_mock_catalogue(theta)
et=time.time()-t
print("using 4 threads the computation time was:%.2lfs creating a catalogue with %d galaxies"%(et,len(HOD1)))

#serial version
from hod import weight_HMF,value_locate
from hod import HOD_mock as HOD_mock_serial

t=time.time()
G1 = HOD_mock_serial(theta,haloes)
et=time.time()-t
print("Serial version computation time was:%.2lfs creating a catalogue with %d galaxies"%(et,len(G1)))