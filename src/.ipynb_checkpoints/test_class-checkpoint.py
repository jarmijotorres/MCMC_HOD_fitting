import numpy as np
from hod_mock import HOD_mock

M = 'GR'
B = 1

infile_r = '/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/'+str(M)+'/box'+str(B)+'/31/Rockstar_M200c_'+str(M)+'_B'+str(B)+'_B1024_NP1024_S31.dat'

data_r = np.loadtxt(infile_r,usecols=(2,33,8,9,10,5,6))

Lbox=1024.0                                     # box size in Mpc/h
Mhalo_min=4.0E11 
theta_test = np.array([13.15,14.2,13.3,0.6,1.0])

H = HOD_mock(data_r,Mhalo_min,Lbox)
centrals = H.HOD_centrals(theta_test)