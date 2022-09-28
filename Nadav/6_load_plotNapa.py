'''
Created on Feb 23, 2017

- map and rate plots of Napa earthquake activity from 08 -09/2014
- ANSS catalog is imported with np.genfromtxt, resulting matrix has following format:
    YR MO DY HR Lat Lon Depth MAG

@author: tgoebel
'''
from __future__ import division
#from StringIO import StringIO
import os
import numpy as np
import matplotlib.pyplot as plb
import matplotlib as mpl
#from obspy import UTCDateTime

#===================================================================================
#                         dir, file, and parameter
#===================================================================================
eqDir  = '%s/Dropbox/Obspy/python_obspy/data'%(os.path.expanduser('~'))
eqFile = 'Napa_2014_08_09_M2.txt'
binsize= 4/365# in dec. year for rate computation
Mc     = 2.0# cut catalog below completenes

#===================================================================================
#                         new function : better move to new file and use 'import ...'
#===================================================================================
def toDecYear( datetime_in, **kwargs ):
    """
    input: datetime_in = array containing time columns year - second
	       out = date in decimal year
    """
    import mx.DateTime
    datetime = mx.DateTime.DateTime( int( datetime_in[0] ), int( datetime_in[1] ), int( datetime_in[2] ),
						int( datetime_in[3] ), int( datetime_in[4] ), float( datetime_in[5] ) )

    year_seconds = ( datetime.day_of_year - 1 ) * 86400.0 + datetime.abstime
    if datetime.is_leapyear:
	    year_fraction = year_seconds / ( 86400.0 * 366 )
    else:
	    year_fraction = year_seconds / ( 86400.0 * 365 )
    return datetime.year + year_fraction

def plotCA( ax):
    import os
    dir_file = '%s/Dropbox/Obspy/python_obspy/data/CA_stateBoundary.txt'%( os.path.expanduser('~'))
    mData = np.loadtxt( dir_file).T
    ax.plot( mData[0], mData[1], 'k-', lw=2)
#====================================1===============================================
#                      load eq. catalog with np.genfromtxt
#====================================================================================
os.chdir( eqDir)
#
mEq = np.genfromtxt( eqFile,  comments='#', dtype = float,#delimiter = '\t',
                          usecols = (0,   1,   2,    3,     4,  5,      6,  7,    8,     9,    16)).T
                            #        YR   MO    Dy   HR    MN, SC,      Lat Lon   Depth MAG    ID
# dtype = ['i8','i8','i8', 'i8','i8','f8', 'f8','f8','f8','f8','i8'],
#mDateTime = mEq[0:6].T
#aDecYr = np.array([ toDecYear( mDateTime[i]) for i in range( len(mEq[0]))])#,mEq[1],mEq[2],mEq[3])
aDecYr = mEq[0] + mEq[1]/12 + mEq[2]/365
### cut catalog and time vector
bSelMag = mEq[9] >= Mc
aDecYr = aDecYr[bSelMag]
mEq    = mEq.T[bSelMag].T
### or for combined matrix, i.e. aDecYr attached to the end
#mEq = np.hstack((mEq.T, aDecYr.reshape(len(aDecYr),1)))
#mEq = mEq[mEq[:,9]>=Mc].T


#====================================2===============================================
#                         plot seismicity map
#====================================================================================
plb.figure(1, figsize=(6,8))
ax1 = plb.subplot(111)
ax1.plot( mEq[7], mEq[6], 'ro')
plotCA( ax1)

#=====================================3==============================================
#                             plot rate
#====================================================================================

aRate, aBins = np.histogram( aDecYr, np.arange( aDecYr.min(), aDecYr.max(), binsize))
plb.figure(2, figsize=(12, 8))
ax2 = plb.subplot(111)
ax2.set_title( 'Seismicity Rate and Magnitudes')

ax2.plot(  aBins[0:-1], aRate)
ax2.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter( '%.2f'))
#mpl.ticker.ScalarFormatter(useOffset=False)
## plot magnitudes
twinx = ax2.twinx()
twinx.plot( aDecYr, mEq[9], 'ro')


#=====================================4==============================================
#                             plot mag distribution
#====================================================================================
aCumSum = np.cumsum( np.ones(mEq[0].shape[0]))[::-1]
plb.figure( 3,figsize=(5,5))
ax3 = plb.subplot( 111)
ax3.set_title( 'Cumulative Mag Distribution')
ax3.semilogy( sorted( mEq[9]), aCumSum, 'ko', markersize = 3)
plb.show()















