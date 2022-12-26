########################################################################################################################
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from obspy import read_inventory
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

# area of interest
rrad = 6371.
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 20., 50.]

# input files
if os.environ['LOGNAME'] == 'sysop':
    wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/autopicker-pb'
else:
    wdir = '/home/lewis/autopicker-pb'
# tectonic plates (Bird, 2002)
if os.environ['LOGNAME'] == 'sysop':
    bird = '/mnt/c/Users/lewiss/Documents/Research/Data/mapping/PB2002/PB2002_boundaries'
else:
    bird = '/home/lewis/mapping/PB2002/PB2002_boundaries'
# local faults
if os.environ['LOGNAME'] == 'sysop':
    fid = open('/mnt/c/Users/lewiss/Documents/Research/Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz', 'r')
else:
    fid = open('/home/lewis/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz', 'r')
flts = fid.readlines()
fid.close()
# FDSN database
isn_client = Client('http://172.16.46.102:8181/')
# station inventory
# isn_inv = isn_client.get_stations(starttime=UTCDateTime('2021-01-01 00:00:00'))
isn_inv = read_inventory(f"{wdir}/../inventory_fdsn.xml", format='STATIONXML')
# read M>3.0 events from FDSNws for period June 2021 - June 2022
etab = pd.read_csv(f"{wdir}/01-06-2021_01-06-2022_M3.csv", parse_dates=['OriginTime'])

# output file name
fig_name = ''
# fig_name = '/home/lewis/autopicker-rt/map-evt-stn.png'
if fig_name:
    mpl.use('Agg')
# create figure & axes
fig, axis1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(18, 9), dpi=200)
if not fig_name:
    plt.show(block=False)
# MAIN MAP
# define global map projection & resolution
m1 = Basemap(projection='cyl', llcrnrlon=mgrd[2], llcrnrlat=mgrd[0], urcrnrlon=mgrd[3], urcrnrlat=mgrd[1], resolution='i', ax=axis1)
# # shaded relief
# m1.shadedrelief(scale=1, alpha=.7, zorder=0)
# grey continents with white background
m1.fillcontinents(color='.8', lake_color='white', zorder=1)
m1.drawmapboundary(fill_color='white', zorder=0)
m1.drawcountries()
# INSET MAP
# define inset map position, projection & resolution
axis2 = inset_axes(axis1, '30%', '18%', loc='lower left')
m2 = Basemap(projection='cyl', llcrnrlon=rgrd[2], llcrnrlat=rgrd[0], urcrnrlon=rgrd[3], urcrnrlat=rgrd[1], resolution='l', ax=axis2)
# topography (ETOPO1.0)
m2.etopo(scale=.5, alpha=.7, zorder=0)
# draw plate boundaries (Bird, 2002)
m2.readshapefile(bird, name='tectonic_plates', drawbounds=True, color='red', linewidth=.5, zorder=2)
# local area of interest
axis2.plot([mgrd[2], mgrd[2], mgrd[3], mgrd[3], mgrd[2]], [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]], linewidth=.5, color='black')
# fault lines
flt_x = []
flt_y = []
hf = []
for iii in range(len(flts)):
    if re.search('NaN', flts[iii]):
        flt_x = []
        flt_y = []
    elif iii < len(flts) - 1 and re.search('NaN', flts[iii + 1]):
        hf, = axis1.plot(flt_x, flt_y, '.6', label='Faults', linewidth=.5, zorder=2)
    else:
        l_line = flts[iii].split()
        flt_x.append(float(l_line[0]))
        flt_y.append(float(l_line[1]))
# seismic station locations with different symbols for each network/channel combination
i1 = isn_inv.select(network='IS', channel='ENZ')
hs1 = axis1.scatter([st.longitude for st in i1.networks[0].stations], [st.latitude for st in i1.networks[0].stations],
                    s=2, marker='s', c='blue', edgecolors='none', label='IS.*.ENZ', zorder=10)
i2 = isn_inv.select(network='IS', channel='HHZ')
hs2 = axis1.scatter([st.longitude for st in i2.networks[0].stations], [st.latitude for st in i2.networks[0].stations],
                    s=15, marker='^', c='none', edgecolors='red', linewidths=.5, label='IS.*.HHZ', zorder=10)
i3 = isn_inv.select(network='IS', channel='BHZ')
hs3 = axis1.scatter([st.longitude for st in i3.networks[0].stations], [st.latitude for st in i3.networks[0].stations],
                    s=15, marker='d', c='none', edgecolors='green', linewidths=.5, label='IS.*.BHZ', zorder=10, alpha=.5)
i4 = isn_inv.select(network='IS', channel='SHZ')
hs4 = axis1.scatter([st.longitude for st in i4.networks[0].stations], [st.latitude for st in i4.networks[0].stations],
                    s=15, marker='v', c='none', edgecolors='yellow', linewidths=.5, label='IS.*.SHZ', zorder=10)
i5 = isn_inv.select(network='GE', channel='HHZ')
hs5 = axis1.scatter([st.longitude for st in i5.networks[0].stations], [st.latitude for st in i5.networks[0].stations],
                    s=15, marker='^', c='none', edgecolors='purple', linewidths=.5, label='GE.*.HHZ', zorder=10)
i6 = isn_inv.select(network='GE', channel='BHZ')
hs6 = axis1.scatter([st.longitude for st in i6.networks[0].stations], [st.latitude for st in i6.networks[0].stations],
                    s=15, marker='d', c='none', edgecolors='magenta', linewidths=.5, label='GE.*.BHZ', zorder=10)
i7 = isn_inv.select(network='GE', channel='SHZ')
hs7 = axis1.scatter([st.longitude for st in i7.networks[0].stations], [st.latitude for st in i7.networks[0].stations],
                    s=15, marker='v', c='none', edgecolors='orange', linewidths=.5, label='GE.*.SHZ', zorder=10)
# seismic event locations
he = axis1.scatter(etab.sort_values(by=['Magnitude']).Longitude, etab.sort_values(by=['Magnitude']).Latitude,
                   c=etab.sort_values(by=['Magnitude']).Magnitude, s=20, cmap='hot_r', vmin=3., vmax=4.5, edgecolors='black', linewidths=.5, alpha=.7, label='Mw > 3', zorder=10)
axis2.scatter(etab.sort_values(by=['Magnitude']).Longitude, etab.sort_values(by=['Magnitude']).Latitude,
              c=etab.sort_values(by=['Magnitude']).Magnitude, s=5, cmap='hot', vmin=3., vmax=4.5, edgecolors='black', linewidths=.5, alpha=.7, zorder=10)
# show parallel and meridians (labels=[left,right,top,bottom])
m1.drawparallels(np.arange(mgrd[0], mgrd[1], 2.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, False], fontsize=5, zorder=15)
m1.drawmeridians(np.arange(mgrd[2], mgrd[3], 2.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, False], fontsize=5, zorder=15)
m2.drawparallels(np.arange(rgrd[0], rgrd[1], 5.),
                 linewidth=.1, dashes=(None, None), labels=[False, False, False, False], fontsize=5, zorder=15)
m2.drawmeridians(np.arange(rgrd[2], rgrd[3], 5.),
                 linewidth=.1, dashes=(None, None), labels=[False, False, False, False], fontsize=5, zorder=15)
# legend
axis1.legend(handles=[he, hs1, hs2, hs3, hs4, hs5, hs6, hs7], loc='lower right', fontsize=5)
hc = plt.colorbar(he, ax=axis1, location='bottom', fraction=.015, pad=.02)
hc.ax.set_xticklabels([3., 3.5, 4., 4.5], fontsize=5)
hc.set_label('Mw', fontsize=7)
# maximise figure
plt.get_current_fig_manager().full_screen_toggle()
# adjust plots
fig.subplots_adjust(left=.07, right=.95, wspace=.1)
# show or save figure
if fig_name:
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    print(f" Figure saved: {fig_name}")
    plt.close()
else:
    plt.show()
