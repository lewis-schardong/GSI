import os
from os import path
import re
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
from calendar import monthrange
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from obspy.core import UTCDateTime
import xml.etree.ElementTree as ETree
from obspy.clients.fdsn import Client
from obspy import read, read_events
from obspy.taup import TauPyModel


# working directory
wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker'

# read input table
atab = pd.read_csv(f'{wdir}/autopicker_M2.8_5.7_27.10.csv')
# replace blanks with '_' in column names
atab.columns = atab.columns.str.replace(' ', '_')
# prepare output table with local catalogue earthquakes
otab = atab[atab.Cat_Type == "EQ"].reset_index(drop=True)

# FIGURE
# magnitude limit
mlim = 2.8
# geographical boundaries
lmap = [27., 36., 32., 38.]
rmap = [11., 51., 15., 55.]
# in-net seismogenic zones
izon = ['Arava', 'Arif fault', 'Barak fault', 'Carmel Tirza', 'Central Israel', 'Dead Sea Basin', 'East Shomron',
        'Eilat Deep', 'Galilee', 'Gaza', 'HaSharon', 'Hula Kinneret', 'Jordan Valley', 'Judea Samaria', 'Negev',
        'Paran', 'West Malhan']
# PLOT RESULTS
# create figure: three maps for (1) stations, (2) in-net events and (3) out-net events
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200)
# define map boundaries & resolution
m1 = Basemap(projection='cyl', llcrnrlon=lmap[2], llcrnrlat=lmap[0],
             urcrnrlon=lmap[3], urcrnrlat=lmap[1], resolution='i', ax=ax1)
m2 = Basemap(projection='cyl', llcrnrlon=lmap[2]+2., llcrnrlat=lmap[0]+2.5,
             urcrnrlon=lmap[3]-2., urcrnrlat=lmap[1]-2.5, resolution='i', ax=ax2)
m3 = Basemap(projection='cyl', llcrnrlon=lmap[2], llcrnrlat=lmap[0],
             urcrnrlon=lmap[3], urcrnrlat=lmap[1], resolution='i', ax=ax3)
# draw map
m1.drawmapboundary(fill_color='none')
m2.drawmapboundary(fill_color='none')
m3.drawmapboundary(fill_color='none')
# fill continents
m1.fillcontinents(color='0.8', lake_color='white')
m2.fillcontinents(color='0.8', lake_color='white')
m3.fillcontinents(color='0.8', lake_color='white')
# faults
flts_id = open(f"/home/{os.environ['LOGNAME']}/.seiscomp/bna/ActiveFaults/activefaults.bna", 'r')
flts = flts_id.readlines()
flts_id.close()
flt = []
for iii in range(len(flts)):
    if re.search('"', flts[iii]):
        flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
    elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
        ax1.plot(flt.lon, flt.lat, '.6', linewidth=.1)
        ax2.plot(flt.lon, flt.lat, '.6', linewidth=.1)
        ax3.plot(flt.lon, flt.lat, '.6', linewidth=.1)
    else:
        l_line = flts[iii].split(',')
        flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
# STATION INVENTORY
if path.exists(f'{wdir}/autopicker_M2.8_inventory_cd.csv') == 0:
    # FDSN station inventory
    isn_inv = Client('http://172.16.46.102:8181/')\
        .get_stations(network='IS,GE', channel='ENZ,HHZ,BHZ,SHZ', level='channel')
    # read inventory file
    stab = pd.read_csv(f'{wdir}/autopicker_M2.8_inventory.csv', dtype={'Location': str}).fillna("")
    # add columns for station coordinates
    stab = stab.assign(Latitude=pd.Series([np.nan] * len(stab), dtype='float64'))\
        .assign(Longitude=pd.Series([np.nan] * len(stab), dtype='float64'))\
        .assign(Elevation=pd.Series([np.nan] * len(stab), dtype='float64'))
    # retrieve coordinates from FDSN inventory
    for i, st in stab.iterrows():
        # in case location is empty
        if not st.Location:
            sta = isn_inv.select(network=st.Network, station=st.Station, location='', channel=st.Channel+'Z')
        else:
            sta = isn_inv.select(network=st.Network, station=st.Station,
                                 location=st.Location, channel=st.Channel+'Z')
        if not sta or not sta.networks[0] or not sta.networks[0].stations[0]:
            print('Missing station to add manually:')
            print(st)
            continue
        # add coordinates to station table
        stab.loc[i, 'Latitude'] = sta.networks[0].stations[0].latitude
        stab.loc[i, 'Longitude'] = sta.networks[0].stations[0].longitude
        stab.loc[i, 'Elevation'] = sta.networks[0].stations[0].elevation
    # write new inventory file containing station coordinates
    stab[['Network', 'Station', 'Location', 'Channel', 'Latitude', 'Longitude', 'Elevation']]\
        .to_csv(f'{wdir}/autopicker_M2.8_inventory_cd.csv', float_format='%.4f', index=False)
else:
    # read inventory file including station coordinates
    stab = pd.read_csv(f'{wdir}/autopicker_M2.8_inventory_cd.csv')
# plot seismometers
tab1 = stab[(stab.Channel == 'SH') | (stab.Channel == 'HH') | (stab.Channel == 'BH')]
h11 = ax1.scatter(tab1.Longitude, tab1.Latitude, s=9, edgecolors='red', marker='^', linewidth=.5,
                  facecolors='none', label=f'Seismometer ({len(tab1)})')
# plot acceletometers
tab2 = stab[stab.Channel == 'EN']
h12 = ax1.scatter(tab2.Longitude, tab2.Latitude, s=4, c='blue', marker='s', linewidth=.5,
                  edgecolors='none', label=f'Accelerometer ({len(tab2)})')
# title
ax1.title.set_text('Station inventory')
# legend
ax1.legend(handles=[h11, h12], loc='lower right', fontsize=5)
# IN-NETWORK (ACCORDING TO SEISMIC ZONES)
# show seismic zones
for i, z in enumerate(izon):
    ax2.text(lmap[2]+2.03, lmap[1]-2.57-i*0.07, z, ha='left', va='center', fontsize=5)
# plot area of middle map
ax1.plot([lmap[2]+2, lmap[2]+2, lmap[3]-2, lmap[3]-2, lmap[2]+2],
         [lmap[0]+2.5, lmap[1]-2.5, lmap[1]-2.5, lmap[0]+2.5, lmap[0]+2.5], color='saddlebrown', linewidth=1)
# M<[mlim] detected in-net events
tab1 = otab[~pd.isna(otab.Auto_ID) & (otab.Cat_M < mlim) & otab.Cat_Region.isin(izon)]
h21 = ax2.scatter(tab1.Cat_Lon, tab1.Cat_Lat, s=4, c='lime',
                  edgecolor='none', alpha=.5, label=f'True M<{mlim} ({len(tab1)})')
# M>=[mlim] detected in-net events
tab2 = otab[~pd.isna(otab.Auto_ID) & (otab.Cat_M >= mlim) & otab.Cat_Region.isin(izon)]
h22 = ax2.scatter(tab2.Cat_Lon, tab2.Cat_Lat, s=16, c='green',
                  edgecolor='none', alpha=.5, label=f'True M>={mlim} ({len(tab2)})')
# M<[mlim] missed in-net events
tab3 = otab[pd.isna(otab.Auto_ID) & (otab.Cat_M < mlim) & otab.Cat_Region.isin(izon)]
h23 = ax2.scatter(tab3.Cat_Lon, tab3.Cat_Lat, s=4, c='orange',
                  marker='x', linewidth=.5, alpha=.5, label=f'Missed M<{mlim} ({len(tab3)})')
# M>=[mlim] missed in-net events
tab4 = otab[pd.isna(otab.Auto_ID) & (otab.Cat_M >= mlim) & otab.Cat_Region.isin(izon)]
h24 = ax2.scatter(tab4.Cat_Lon, tab4.Cat_Lat, s=16, c='red',
                  marker='x', linewidth=.5, alpha=.5, label=f'Missed M>={mlim} ({len(tab4)})')
# title
ax2.title.set_text('In-network events')
# legend
ax2.legend(handles=[h21, h22, h23, h24], loc='lower left', fontsize=5)
# OUT-NETWORK (ACCORDING TO SEISMIC ZONES)
# plot area of middle map
ax3.plot([lmap[2]+2, lmap[2]+2, lmap[3]-2, lmap[3]-2, lmap[2]+2],
         [lmap[0]+2.5, lmap[1]-2.5, lmap[1]-2.5, lmap[0]+2.5, lmap[0]+2.5], color='saddlebrown', linewidth=1)
# M<[mlim] detected in-net events
tab1 = otab[~pd.isna(otab.Auto_ID) & (otab.Cat_M < mlim) & ~otab.Cat_Region.isin(izon)]
h31 = ax3.scatter(tab1.Cat_Lon, tab1.Cat_Lat, s=4, c='lime',
                  edgecolor='none', alpha=.5, label=f'True M<{mlim} ({len(tab1)})')
# M>=[mlim] detected in-net events
tab2 = otab[~pd.isna(otab.Auto_ID) & (otab.Cat_M >= mlim) & ~otab.Cat_Region.isin(izon)]
h32 = ax3.scatter(tab2.Cat_Lon, tab2.Cat_Lat, s=16, c='green',
                  edgecolor='none', alpha=.5, label=f'True M>={mlim} ({len(tab2)})')
# M<[mlim] missed in-net events
tab3 = otab[pd.isna(otab.Auto_ID) & (otab.Cat_M < mlim) & ~otab.Cat_Region.isin(izon)]
h33 = ax3.scatter(tab3.Cat_Lon, tab3.Cat_Lat, s=4, c='orange',
                  marker='x', linewidth=.5, alpha=.5, label=f'Missed M<{mlim} ({len(tab3)})')
# M>=[mlim] missed in-net events
tab4 = otab[pd.isna(otab.Auto_ID) & (otab.Cat_M >= mlim) & ~otab.Cat_Region.isin(izon)]
h34 = ax3.scatter(tab4.Cat_Lon, tab4.Cat_Lat, s=16, c='red',
                  marker='x', linewidth=.5, alpha=.5, label=f'Missed M>={mlim} ({len(tab4)})')
# title
ax3.title.set_text('Out-of-network events')
# legend
ax3.legend(handles=[h31, h32, h33, h34], loc='lower left', fontsize=5)
# maximise figure
plt.get_current_fig_manager().full_screen_toggle()
# adjust plots
fig.subplots_adjust(left=.05, bottom=.05, wspace=.05)
# show figure
plt.show()
