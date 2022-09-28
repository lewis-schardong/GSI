'''
Created on Feb 28, 2017

- pre-processing (MAG, Time selection)
- download waveforms using obspy and visualize waveforms

see https://www.fdsn.org/webservices/
    https://service.iris.edu/irisws/   for details about data clients
and https://docs.obspy.org/packages/obspy.clients.fdsn.html for obspy tutorial

@author: tgoebel
'''
#--------------py modules----------------------------
import os
import numpy as np
import matplotlib.pyplot as plb
import matplotlib as mpl

import obspy
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
eqDir  = '%s/Dropbox/Obspy/python_obspy/data'%(os.path.expanduser('~'))
eqFile = 'Napa_2014_08_09_M2.txt'
Mc     = 3# cut catalog below completenes
sNet   = 'NC'
sSta   = 'N016' #JUC - UC Santa Cruz
minFr, maxFr = .1, 10
#sCha   = 'HHZ' # UCSC = EHZ and SHZ other: 'BHZ', 'LHZ'
#aEvID  = [72282711, 2282716, 72282751] # so far only sta-picks and wfs for these
#                                           three events have been downloaded

bDownloadEqCat = False

#change to working dir, and import catalog params from .txt file
os.chdir( eqDir)
dCatPar = utils.loadCatParams( 'Napa') # get catalog parameters
print('cat params', dCatPar)

#====================================1===============================================
#                      load eq. catalog with np.genfromtxt
#====================================================================================
starttime = UTCDateTime( dCatPar['starttime'])
endtime   = UTCDateTime( dCatPar['endtime'])

#
mEq = np.genfromtxt( eqFile,  comments='#',#delimiter = '\t',
                          usecols = (0,   1,   2,    3,     4,  5,      6,  7,    8,     9,    16)).T
                            #        YR   MO    Dy   HR    MN, SC,      Lat Lon   Depth MAG    ID

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
inventory = client.get_stations(network=sNet, station = sSta,
                                starttime=starttime,  endtime=endtime)
print(inventory)
inventory.plot()

for iEv in xrange( 3): #mEq[0].shape[0]:
    # -----------------2.2 phase data-----------------------------------------
    # TODO 4:  use the info in Napa_2014_08_09_M2_phase.txt to plot P arrival for the first three events
    # (col: 30-35, i.e. in py 29-34 arrival times are in sec.)
    # -----------------2.3  waveforms-----------------------------------------
    # TODO 2: I only downloaded waveforms for the mainshock and the first two M>3 aftershocks
    # go to: http://service.ncedc.org/ncedcws/eventdata/1/
    # and downdload the waveforms for the other 4 M>3 events
    SEIS = read(  'ncedcws-eventdata_%s_%s_%i.mseed'%( sNet, sSta, int(mEq[-1][iEv])))
    print('number of traces', len( SEIS))
    SEIS.plot() # this plots all traces within the miniseed file
    #TODO 1: we can also loop over each trace individually, uncomment the following three lines and see what happens
    # for i in xrange( len(SEIS)):
    #     print SEIS[i].stats # this returns sampling rate and dt in addition to other sta infos
    #     SEIS[i].plot()












