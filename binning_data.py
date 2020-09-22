# coding: utf-8
# %load 36
# %load 32

import numpy as np
#import sys

def binning_function(X,Y,bl,bu,Nb,percentile=50):
    """
    binning scattered data between bl, bu with Nb number of bins.
    
    """
    p = percentile #50 + percentile/2.
    #p = 16.
    #bl = 0.11#sys.argv[1]
    #bu = 0.9#sys.argv[2]
    #Nb = 16#sys.argv[3]
    BINS = np.linspace(bl,bu,Nb+1)#np.linspace(0.01,0.95,5)
    b = BINS[:-1] + np.diff(BINS)[0]/2.#bin centre
    bmedian = np.zeros_like(b)
    xmean = np.zeros_like(b)
    bmean = np.zeros_like(b)
    pert = np.zeros_like(b)
    perb = np.zeros_like(b)
    N = np.zeros_like(b)
    Ybn = []
    for i in np.arange(Nb):
        xx = X[(X>BINS[i])&(X<BINS[i+1])]#x axis
        yy = Y[(X>BINS[i])&(X<BINS[i+1])]
        Ybn.append(yy)
        bmedian[i] = np.median(yy)
        xmean[i] = np.mean(xx)
        bmean[i] = np.mean(yy)
        #pert[i] = np.percentile(yy,p)
        #perb[i] = np.percentile(yy,100.-p)
        N[i] = len(yy)
    S = np.array([b,xmean,bmean,bmedian,N]).T
    return S,Ybn


# =============================================================================
 # =============================================================================
# =============================================================================
# f,ax = plt.subplots(1,1,figsize=(7,6))
# ax.plot(xx,yy,'k.')
# ax.plot(b,p16,'g--',alpha=1.,linewidth=3.5,label='68% confidence')
# ax.plot(b,p84,'g--',alpha=1.,linewidth=3.5)
# ax.plot(b,p5,'r--',alpha=1.,linewidth=3.5,label='95% confidence')
# ax.plot(b,p95,'r--',alpha=1.,linewidth=3.5)
# ax.plot(b,bmedian,'m*',ms=12.)
# ax.legend(loc=2,prop={'size':15})
# ax.set_ylabel(r'$(z_{ph} - z_s)/(1+z_s)$')
# ax.set_xlabel(r'$n$(<Qz)')
# ax.hlines(0.0,0.,1.,linestyles='--',colors='b',linewidth=2.)
# ax.set_ylim(-.15,.15)
# ax.set_xlim(-0.01,1.01)
# plt.show()
#   
# =============================================================================
 # =============================================================================
# 
# =============================================================================
