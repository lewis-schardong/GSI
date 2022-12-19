########################################################################################################################
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

# area of interest
rrad = 6371.
agrd = [20., 40., 20., 40.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 20., 50.]

# input files
# file = '/home/sysop/seiscomp/share/scautoloc/grid.conf'
file = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/grid-regional.conf'
gtab = pd.read_csv(file, names=['lat', 'lon', 'dep', 'rad', 'dis', 'pha'], sep='\\s+')
if os.environ['LOGNAME'] == 'sysop':
    bird = '/mnt/c/Users/lewiss/Documents/Research/Data/mapping/PB2002/PB2002_boundaries'
else:
    bird = '/home/lewis/mapping/PB2002/PB2002_boundaries'
if re.search('local', file):
    xgrd = agrd
else:
    xgrd = rgrd

# output file name
fig_name = ''
# fig_name = '/home/lewis/autopicker-rt/autoloc-grid.png'
if fig_name:
    mpl.use('Agg')
# create figure & axes
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
if not fig_name:
    plt.show(block=False)
# define global map projection & resolution
m1 = Basemap(projection='eck4', lon_0=0, resolution='l', ax=axis1)
# define regional map boundaries and resolution
m2 = Basemap(projection='cyl', llcrnrlon=xgrd[2], llcrnrlat=xgrd[0],
             urcrnrlon=xgrd[3], urcrnrlat=xgrd[1], resolution='i', ax=axis2)
# fill continents
m1.fillcontinents(color='.8', lake_color='white')
m2.fillcontinents(color='.8', lake_color='white')
# draw map
m1.drawmapboundary(fill_color='none')
m2.drawmapboundary(fill_color='none', color='.5')
# draw plate boundaries
m1.readshapefile(bird, name='tectonic_plates', drawbounds=True, color='red')
m2.readshapefile(bird, name='tectonic_plates', drawbounds=True, color='black')
# project grid lat/lon coordinates
gtab['x'], gtab['y'] = m1(gtab.lon, gtab.lat)
# marker sizes for map symbols
ms1 = [4, 2, 1, .5, .1]
ms2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
mc1 = ['purple', 'blue', 'green', 'yellow', 'red']
# loop over unique depth values
hh1 = []
hh2 = []
i1 = 0
i2 = 0
for d in gtab.dep.drop_duplicates().sort_values(ascending=False).to_list():
    if d == 33 or d == 150 or d == 300 or d == 450 or d == 600:
        # GLOBAL
        h1, = axis1.plot(gtab.x[gtab.dep == d], gtab.y[gtab.dep == d], 'o', mec=mc1[i1], mfc='none',
                         markersize=ms1[i1], linewidth=.00000001, alpha=.5, label=f'{d} km')
        hh1.append(h1)
        i1 += 1
    else:
        # REGIONAL + LOCAL
        h2, = axis2.plot(gtab.lon[gtab.dep == d], gtab.lat[gtab.dep == d], 'o', mfc='none',
                         markersize=ms2[i2]/1.5, linewidth=.0, alpha=.5, clip_on=False, label=f'{d} km')
        hh2.append(h2)
        i2 += 1
# regional and local areas of interest
x, y = m1([xgrd[2], xgrd[2], xgrd[3], xgrd[3], xgrd[2]], [xgrd[0], xgrd[1], xgrd[1], xgrd[0], xgrd[0]])
axis1.plot(x, y, color='.5', linewidth=.5, alpha=.7)
x, y = m1([mgrd[2], mgrd[2], mgrd[3], mgrd[3], mgrd[2]], [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]])
axis1.plot(x, y, color='brown', linewidth=.5, alpha=.7)
axis2.plot([mgrd[2], mgrd[2], mgrd[3], mgrd[3], mgrd[2]], [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]],
           color='brown', linewidth=.5, alpha=.7)
# show parallel and meridians (labels=[left,right,top,bottom])
m1.drawparallels(np.arange(-90., 120., 30.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, True], fontsize=5)
m2.drawparallels(np.arange(xgrd[0], xgrd[1], 2.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, True], fontsize=5)
m1.drawmeridians(np.arange(0., 360., 60.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, True], fontsize=5)
m2.drawmeridians(np.arange(xgrd[2], xgrd[3], 2.),
                 linewidth=.1, dashes=(None, None), labels=[True, True, True, True], fontsize=5)
# legend
axis1.legend(handles=hh1, loc='best', fontsize=5)
axis2.legend(handles=hh2, loc='best', fontsize=5)
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
