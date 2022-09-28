'''
Created on Jan 23, 2016

- estimate diffusion curve the envelopes seismicit cloud in r-t plot

@author: tgoebel
'''
from __future__ import division
import matplotlib as mpl
import os
import pylab as plb
import numpy as np
#import dateUtils
#from obspy import UTCDateTime


#===================================================================================
#                         dir, file, and parameter
#===================================================================================
fD = 0.004 # hydraulic diffusivity in m2/s
afD = np.arange(0.001, 0.01, 0.002)
t0        = 2004.39344262  #start of injection in seconds 1085349600 after 1970/1/1
tShift    = 0 # use this parameter to shift seismicity relative to diffusion curves

catName   = 'KTB'
dataDir   = '%s/Dropbox/Obspy/python_obspy/data'%( os.path.expanduser( '~'))

eqFile    = '%s_seism_txyz.txt'%( catName)
wellFile  = '%s_wells_xyz.txt'%( catName)

plotFile  = '%s_perm.png'%( catName)

def computeDiffCurve( aT, D):
    """

    :param aT: time vector
    :param D: hydrolic diff.
    :return:
    """
    return np.sqrt( 4*np.pi*D*aT)


#===========================1========================================================
#                       load data
#====================================================================================
os.chdir( dataDir)

mData = np.loadtxt( eqFile).T
aT, aX, aY, aZ = mData[1], mData[2], mData[3], mData[4]

mData = np.loadtxt( wellFile).T
aX_well, aY_well, aZ_well = mData[4], mData[5], mData[3]

#===========================2========================================================
#                       compute distance
#====================================================================================
aR = np.sqrt( (aX_well-aX)**2 + (aY_well-aY)**2 + (aZ_well-aZ)**2 )
aR *= 1e3

aT = (aT - t0)*365*24 - tShift
#===========================3========================================================
#                       r-t plot and diffusion curves
#====================================================================================
plb.figure(1)
ax1 = plb.subplot( 111)
### plot seismicity
ax1.plot( aT, aR, 'bo', ms = 10, mfc = 'none', mec = 'b', mew = 2)


### plot diffusion curve
# get x-limits in hours
aXlim    = np.array( ax1.get_xlim())
aT_model = np.arange( 0, aXlim[1], 1)
###TODO: write this into a for loop to simulate different diffusivities
for i in range( len(afD)):
    print( afD[i])
    aD       = computeDiffCurve(aT_model*3600, afD[i])
    ax1.plot( aT_model, aD , label  = 'D = %.3f'%(afD[i]), lw = 2)# color = 'r')


# legend
ax1.legend( loc = 'upper left', frameon = False)

ax1.set_xlabel( 'Time [hr]')
ax1.set_ylabel( 'Distance [m]') 
ax1.set_xlim( -500, 9000)
ax1.set_ylim( 0, 2000)

#plb.savefig( plotFile, dpi = 250)
plb.show()













