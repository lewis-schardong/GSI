########################################################################################################################
import os
from os import path
import re
import filecmp
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
import statistics
from scipy import signal
from datetime import datetime, timedelta
import xml.etree.ElementTree as ETree
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def read_autopick_xml(xml_path, phase='P', picker=''):
    """
    :param xml_path: path to .XML file containing picks to read
    :param phase: seismic phase of interest
    :param picker: picker used
    :return: DataFrame containing automatic picks
    """
    atab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                         'chn': pd.Series(dtype='string'), 'pick': pd.Series(dtype='string')})
    # read resulting xml file
    for line_xml in ETree.parse(xml_path).getroot()[0]:
        if re.search('pick', str(line_xml.tag)):
            # check picker
            if picker and not re.search(picker, str(line_xml.attrib)):
                continue
            # check phase
            if (line_xml[3].tag == 'phaseHint' and line_xml[3].text != phase) or\
                    (line_xml[4].tag == 'phaseHint' and line_xml[4].text != phase):
                continue
            # station metadata
            str1 = re.search("'publicID': '(.*?)'", str(line_xml.attrib)).group(1)
            str2 = str1.split('.')
            str3 = str2[2].split('-')
            if not picker:
                pik_net = str3[1]
            else:
                pik_net = str3[2]
            if str2[4] is None:
                pik_loc = ''
            else:
                pik_loc = str2[4]
            atab.loc[atab.shape[0]] = [pik_net, str2[3], pik_loc, str2[5], datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ')]
    return atab


def get_catalogue_picks(client, t_beg, t_end):
    """
    :param client: Obspy FDSN client to retrieve data from
    :param t_beg: starting time
    :param t_end: ending time
    :return: data streamer containing catalogue picks in trace headers
    """
    # DataFrame to contain catalogue picks
    ctab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                         'chn': pd.Series(dtype='string'), 'pick': pd.Series(dtype='datetime64[ms]')})
    # retrieve event data
    evt_lst = client.get_events(starttime=t_beg, endtime=t_end, includearrivals=True)
    for evt in evt_lst:
        for pik in evt.picks:
            if pik.phase_hint[0] == 'P':
                if pik.waveform_id.location_code is None:
                    pik_loc = ''
                else:
                    pik_loc = pik.waveform_id.location_code
                ctab.loc[ctab.shape[0]] = [pik.waveform_id.network_code, pik.waveform_id.station_code,
                                           pik_loc, pik.waveform_id.channel_code]
    return ctab


########################################################################################################################
# input parameters
exp = 0
ntw = 'IS'
chn = '(B|H|E)(H|N)Z'
pic = 'AIC'

# area of interest
rrad = 6371.
igrd = [29., 34., 34., 36.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 25., 45.]
vel_mod = 'gitt05'

# working directory
wdir = f'/mnt/c/Users/lewiss/Documents/Research/Autopicker'
print(f'Working directory: {wdir}\n')
mpl.rcParams['savefig.directory'] = f"{wdir}"
# data archive directory
adir = '/net/172.16.46.200/archive/jqdata/archive'

# FDSN database
isn_client = Client('http://172.16.46.102:8181/')

# autopicker parameters
rmhp =    [10., 10., 10., 10.]
taper =   [30., 30., 30., 30.]
bworder = [4,   4,   4,   4  ]
bwminf =  [.7,  3.,  4.,  4. ]
bwmaxf =  [2.,  6.,  8.,  8. ]
sta =     [2.,  2.,  2.,  .2 ]
lta =     [80., 80., 80., 10.]
trigon =  [3.,  3.,  3.,  3. ]
trigoff = [1.5, 1.5, 1.5, 1.5]
fpar = {'rmhp': rmhp[exp], 'taper': taper[exp], 'bworder': bworder[exp], 'bwminf': bwminf[exp], 'bwmaxf': bwmaxf[exp],
        'sta': sta[exp], 'lta': lta[exp], 'trigon': trigon[exp], 'trigoff': trigoff[exp]}

########################################################################################################################
# RETRIEVE WAVEFORM DATA
# starting and ending dates for Jan 2022 swarm
tbeg = datetime.strptime('2022-01-22 00:00:00', '%Y-%m-%d %H:%M:%S')
for i in range(1, 4):
    print(tbeg, tbeg + timedelta(days=1))
    tbeg += timedelta(days=1)
exit()
tend = datetime.strptime('2022-01-25 23:59:59', '%Y-%m-%d %H:%M:%S')
mseed = f'{wdir}/swarm/22-01-2022_25-01-2022.raw.mseed'
if path.exists(mseed) == 0:
    # import miniSEED file
    os.system(f'scart -dsE -n "{ntw}" -c "{chn}" -t "{tbeg}~{tend}" {adir} > {mseed}')
# read raw miniSEED file
print('Reading...')
isn_traces = read(mseed)
# ensuring 50-Hz sampling rate for all traces
print('Resampling...')
isn_traces.resample(50.0, window='hann')
# merging all traces
print('Merging...')
isn_traces.merge(fill_value='interpolate')
# remove problematic channels
print('Cleaning...')
for t in isn_traces:
    if t.stats.station == 'KRPN' or (t.stats.station == 'EIL' and t.stats.channel == 'BHZ') or (t.stats.station == 'GEM' and t.stats.channel == 'BHZ') \
            or (t.stats.station == 'KFSB' and t.stats.channel == 'HHZ' and t.stats.location == '22')\
            or (t.stats.station == 'HRFI' and t.stats.channel == 'HHZ' and t.stats.location == '')\
            or (t.stats.channel == 'HHZ' and ('MMA' in t.stats.station or 'MMB' in t.stats.station or 'MMC' in t.stats.station)):
        isn_traces.remove(t)
# remove trend from all traces
print('Detrending...')
isn_traces.detrend('spline', order=3, dspline=500)
# write new .mseed file for processed data
print('Writing...')
isn_traces.write(mseed.replace('.raw', ''))

########################################################################################################################
# AUTOPICKER RUN
# output file
oxml = mseed.replace('mseed', 'xml')
# check the autopicker was run
if path.exists(f"{wdir}/swarm/{oxml}") != 0:
    print(f"Experiment #{exp} already ran:")
    os.system(f"ls -lh {wdir}/swarm/{oxml}")
else:
    # check the right configuration file is the current one
    if not filecmp.cmp(f'{wdir}/config_autop.xml', f'{wdir}/config_autop_{exp}.xml'):
        os.system(f'cp -p {wdir}/config_autop_{exp}.xml {wdir}/config_autop.xml')
    # autopicker command (using raw .mseed file)
    cmd = f"scautopick --ep --config-db {wdir}/config_autop.xml --inventory-db {wdir}/inventory_autop.xml" \
          f" --playback -I file://{mseed} > {wdir}/swarm/{oxml}"
    print(f"Running experiment #{exp}:")
    print(' ' + cmd)
    os.system(cmd)

########################################################################################################################
# RETRIEVE AUTOMATIC PICKS
# load waveform data to plot (not .raw.mseed)
isn_traces = read(mseed.replace('.raw', ''))
# read output file
atab = read_autopick_xml(f"{wdir}/swarm/{oxml}", 'P', 'AIC')
print(f'{len(atab)} automatic picks')

########################################################################################################################
# MORE DATA PROCESSING (FOR PLOTTING ONLY)
# apply taper to all traces
isn_traces.taper(max_percentage=.5, type='cosine', max_length=fpar['taper'], side='left')
# apply high-pass filter to all traces
isn_traces.filter('highpass', freq=1./fpar['rmhp'])
# apply Butterworth band-pass filter to all traces
isn_traces.filter('bandpass', freqmin=fpar['bwminf'], freqmax=fpar['bwmaxf'], corners=fpar['bworder'])
# downsample data (to avoid memory issues when plotting)
isn_traces.resample(1.0, window='hann')

#######################################################################################################################
# RETRIEVE CATALOGUE PICKS
ctab = get_catalogue_picks(isn_client, tbeg, tend)
print(f'{len(ctab)} catalogue picks')
