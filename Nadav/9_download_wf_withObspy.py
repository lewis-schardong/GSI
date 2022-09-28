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
from obspy.core.event import Catalog

client = Client('NCEDC')

#--------------my modules----------------------------
import utils_NCEDC as utils
#===================================================================================
#                         dir, file, and parameter
#===================================================================================
eqDir  = '%s/Documents/teaching/python_obspy/data'%(os.path.expanduser('~'))
eqFile = 'Napa_2014_08_09_M2.txt'
Mc     = 3# cut catalog below completenes
sNet   = 'NC'
sSta   = 'N016' #
sLoc   = '*' # '01', '' does not work, maybe in more recen versions
sCha   = 'HNZ' # try alos 'HN*'
tPrePick = 5
dt_wf    = 60 # in seconds

#change to working dir, and import catalog params from .txt file
os.chdir( eqDir)
dCatPar = utils.loadCatParams( 'Napa') # get catalog parameters
print 'cat params', dCatPar
client = Client( 'NCEDC') #IRIS, ETH, GFZ, SCEDC, USGS ... etc
#client = Client( 'IRIS') #IRIS, ETH, GFZ, SCEDC, USGS ... etc

#====================================1===============================================
#                      load eq. catalog with np.genfromtxt
#====================================================================================
starttime = UTCDateTime( dCatPar['starttime'])
#endtime   = UTCDateTime( dCatPar['endtime'])

#
mEq = np.genfromtxt( eqFile,  comments='#',#delimiter = '\t',
                          usecols = (0,   1,   2,    3,     4,  5,      6,  7,    8,     9,    16)).T
                            #        YR   MO    Dy   HR    MN, SC,      Lat Lon   Depth MAG    ID

### cut catalog and time vector
bSelMag = mEq[9] >= Mc
mEq    = mEq.T[bSelMag].T
print 'total number of selected eqs in NC catalog', mEq[0].shape[0]
print 'event IDs', mEq[-1]
print 'event Mag', mEq[-2]



#====================================2===============================================
#                      get waveform for each event in mEq
#====================================================================================
for iEv in xrange( mEq[0].shape[0]):
    starttime = UTCDateTime( int(mEq[0][iEv]), int(mEq[1][iEv]), int(mEq[2][iEv]), int(mEq[3][iEv]), int(mEq[4][iEv]), mEq[5][iEv])-tPrePick
    # dt_wf has to be in seconds for this to work
    endtime   = starttime + dt_wf
    lStaTr  = client.get_waveforms( sNet, sSta, sLoc, sCha, starttime, endtime)
    lStaTr.plot()

    #if you uncomment the code below, obspy will download event and pick info for all stations that
    # recorded the Napa earthquake and the first two M3 aftershocks
    # catalog = client.get_events( eventid= str(int(mEq[-1][iEv])), includearrivals=True)
    # print int(mEq[-1][iEv]), catalog
    # for event in catalog:
    #     for pick in event.picks:
    #         if pick.waveform_id.network_code == sNet:
    #             starttime = UTCDateTime( int(mEq[0][iEv]), int(mEq[1][iEv]), int(mEq[2][iEv]), int(mEq[3][iEv]), int(mEq[4][iEv]), mEq[5][iEv])-tPrePick#dt_wf*.5
    #             # dt_wf has to be in seconds for this to work
    #             endtime   = starttime + dt_wf# dt_wf*.5
    #             print starttime, event.origins[0].time
    #
    #             # -----------------2.3  waveforms-----------------------------------------
    #             #lStaTr = client.get_waveforms( network = sNet, station = sSta, channel = '', location='',
    #             #                               starttime = starttime, endtime = endtime)
    #             #lStaTr  = client.get_waveforms( network = "TA", station="637A", location="00",channel="BH*", startime=starttime,    endtime=endtime)
    #             if pick.waveform_id.station_code == sSta:
    #                 print pick.waveform_id.network_code, pick.waveform_id.station_code,pick.waveform_id.location_code, pick.waveform_id.channel_code, starttime, endtime
    #                 lStaTr  = client.get_waveforms( pick.waveform_id.network_code, pick.waveform_id.station_code,
    #                                                 pick.waveform_id.location_code, pick.waveform_id.channel_code, starttime, endtime)
    #                 lStaTr.plot()
    #                 #lStaTr  = client.get_waveforms( network = "IU", station="ANMO", location="00",channel="LHZ", startime=starttime,    endtime=endtime)
    #                 #lStaTr  = client.get_waveforms("IU", "ANMO", "00", "LHZ", starttime, endtime)
    #                 # get_waveforms options:
    #                 #lStaTr  = client.get_waveforms(network=sNet, station=sSta, location='',channel=sCha, starttime=starttime,endtime=endtime,
    #                 #                        attach_response=False)
    #










