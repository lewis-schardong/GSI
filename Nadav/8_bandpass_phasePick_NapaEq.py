'''
Created on Feb 28, 2017

- pre-processing (MAG, Time selection)
- download waveforms using obspy and visualize waveforms

see https://www.fdsn.org/webservices/
    https://service.iris.edu/irisws/   for details about data clients
and https://docs.obspy.org/packages/obspy.clients.fdsn.html for obspy tutorial

@author: tgoebel
'''
from __future__ import division
#--------------py modules----------------------------
import os
import numpy as np

import matplotlib.pyplot as plb
import matplotlib as mpl

#import obspy
#from obspy.clients.fdsn import Client
from obspy.fdsn import Client
from obspy import UTCDateTime
from obspy.core import read
from obspy.signal import filter
#--------------my modules----------------------------
import utils_NCEDC as utils
#===================================================================================
#                         dir, file, and parameter
#===================================================================================
eqDir  = '%s/Documents/teaching/python_obspy/data'%(os.path.expanduser('~'))
eqFile = 'Napa_2014_08_09_M2.txt'
phaseFile = 'Napa_2014_08_09_M2_phase.txt'
Mc     = 3# cut catalog below completenes
sNet   = 'NC'
sSta   = 'N016' #JUC - UC Santa Cruz
minFr, maxFr = 2, 20
timeShift = 0 #-6.42
#sCha   = 'HHZ' # UCSC = EHZ and SHZ other: 'BHZ', 'LHZ'
#aEvID  = [72282711, 72282716, 72282751] # so far only sta-picks and wfs for these
#                                           three events have been downloaded

bDownloadEqCat = False
v = np.arange( 10, dtype = float)
v = np.int8( v)
print(type( v))
#change to working dir, and import catalog params from .txt file
os.chdir( eqDir)
dCatPar = utils.loadCatParams( 'Napa') # get catalog parameters
print('cat params', dCatPar)

#====================================1===============================================
#                      load eq. catalog with np.genfromtxt
#====================================================================================
starttime = UTCDateTime( dCatPar['starttime'])
endtime   = UTCDateTime( dCatPar['endtime'])
os.chdir( eqDir)
#
mEq = np.genfromtxt( eqFile,  comments='#', dtype = float,#delimiter = '\t',
                          usecols = (0,   1,   2,    3,     4,  5,      6,  7,    8,     9,    16)).T

### cut catalog and time vector
bSelMag = mEq[9] >= Mc
mEq    = mEq.T[bSelMag].T
print('total number of selected eqs in NC catalog', mEq[0].shape[0])
print('event IDs', mEq[-1])
print('event Mag', mEq[-2])


#client = Client( 'NCEDC') #IRIS, ETH, GFZ, SCEDC, USGS ... etc
client = Client( 'NCEDC') #IRIS, ETH, GFZ, SCEDC, USGS ... etc

if bDownloadEqCat == True:
    cat = client.get_events(starttime=dCatPar['starttime'], endtime=dCatPar['endtime'],minmagnitude=Mc,
                            minlatitude=dCatPar['minlatitude'],maxlatitude=dCatPar['maxlatitude'],
                            minlongitude=dCatPar['minlongitude'],maxlongitude=dCatPar['maxlongitude'],
                            )# catalog="ANSS")
    print( cat)
    # cat.plot() # plot catalog downloaded with obspy
#====================================2===============================================
#                      get waveform for each event in mEq
#====================================================================================
# -----------------2.1 stations-----------------------------------------
# obspy has a simple way of downloading station info
# TODO 3: write a short function in utils_NCEDC.py that opens NCEDC_station.txt
# and return the lon/lat of station N016
inventory = client.get_stations(network=sNet, station = sSta)
#see: http://docs.obspy.org/archive/0.10.2/packages/autogen/obspy.station.inventory.Inventory.html#obspy.station.inventory.Inventory
print(inventory.get_contents())

#isn_inv.plot() # plot station location, requires mpl - basemap

for iEv in xrange( 3): #mEq[0].shape[0]:
    # origin time of ev
    f_tOri = UTCDateTime( int(mEq[0][iEv]), int(mEq[1][iEv]), int(mEq[2][iEv]), int(mEq[3][iEv]), int(mEq[4][iEv]),
                          float(mEq[5][iEv]) )
    # -----------------2.2 phase data-----------------------------------------
    # TODO 4:  use the info in Napa_2014_08_09_M2_phase.txt to plot P arrival for the first three events
    os.chdir( eqDir)
    # ! make sure event ID is an integer
    #UTC_tP, UTC_tS = utils.getPhasePicks( phaseFile, int( mEq[-1][iEv]), sSta)

    # -----------------2.3  waveforms-----------------------------------------
    # TODO 2: I only downloaded waveforms for the mainshock and the first two M>3 aftershocks
    # go to: http://service.ncedc.org/ncedcws/eventdata/1/
    # and downdload the waveforms for the other 4 M>3 events
    SEIS = read(  'ncedcws-eventdata_%s_%s_%i.mseed'%( sNet, sSta, int(mEq[-1][iEv])))
    print('number of traces', len( SEIS))
    #lStaTr.plot() # this plots all traces within the miniseed file
    aTime = np.arange(0, SEIS[0].stats.npts/SEIS[0].stats.sampling_rate, SEIS[0].stats.delta)
    #TODO 1: we can also loop over each trace individually, uncomment the following three lines and see what happens
    for iTr in xrange( len(SEIS)):
        #print SEIS[iTr].stats #.starttime # this returns sampling rate and dt in addition to other sta infos
        # subtract start time of time series
        f_tP_s, f_tS_s = UTC_tP-SEIS[iTr].stats.starttime, UTC_tS-SEIS[iTr].stats.starttime

        ##### now we can plot the time series in sec and display pick information
        SEIS[iTr].detrend( 'linear')
        aTr_bp = filter.bandpass(SEIS[iTr], minFr, maxFr, SEIS[iTr].stats.sampling_rate, corners=4, zerophase=True)


        plb.figure(2, figsize = (12,4))
        ax = plb.subplot(111)
        ax.set_title( '%s , %s'%( sSta, SEIS[iTr].stats.channel))
        ax.plot( aTime, np.array( SEIS[iTr]), 'k-', alpha =.4, lw =2, label = 'raw wf.')
        ax.plot( aTime, aTr_bp, 'k-', alpha =1, lw =3, label = 'bp, $f$=%s-%s'%( minFr, maxFr))
        ax.plot( [f_tP_s+timeShift, f_tP_s+timeShift], ax.get_ylim(), 'r--', label = '$P$')
        ax.plot( [f_tS_s+timeShift, f_tS_s+timeShift], ax.get_ylim(), 'g--', label = '$S$')
        # TODO: find a better time window to reveal details of wf.
        # ax.set_xlim( f_tP_s - 2, f_tP_s + 10)
        ax.legend( loc = 'upper right', frameon = True)
        ax.set_xlabel( 'Time [s]')
        plb.show()











