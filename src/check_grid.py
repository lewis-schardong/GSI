########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

# area of interest
rrad = 6371.
igrd = [29., 34., 34., 36.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 20., 50.]

# input files
gtab = pd.read_csv("/home/sysop/seiscomp/share/scautoloc/grid.conf",
                   names=['lat', 'lon', 'dep', 'rad', 'dis', 'pha'], sep='\\s+')

# output file name
fig_name = ''
# fig_name = '/home/lewis/autopicker-rt/autoloc-grid.png'
if fig_name:
    mpl.use('Agg')
# create figure & axes
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
if not fig_name:
    plt.show(block=False)
# GLOBAL
# define map projection & resolution
m1 = Basemap(projection='eck4', lon_0=0, resolution='l', ax=axis1)
# draw map
m1.drawmapboundary(fill_color='none')
# fill continents
m1.fillcontinents(color='0.8', lake_color='white')
# show grid
# project lat/lon coordinates
gtab['x'], gtab['y'] = m1(gtab.lon, gtab.lat)
h5, = axis1.plot(gtab.x[gtab.dep == 600.], gtab.y[gtab.dep == 600.], 'o', mec='purple', mfc='none', markersize=4, linewidth=.001, label='600 km', alpha=.7)
h4, = axis1.plot(gtab.x[gtab.dep == 450.], gtab.y[gtab.dep == 450.], 'o', mec='blue', mfc='none', markersize=2, linewidth=.001, label='450 km', alpha=.7)
h3, = axis1.plot(gtab.x[gtab.dep == 300.], gtab.y[gtab.dep == 300.], 'o', mec='green', mfc='none', markersize=1, linewidth=.001, label='300 km', alpha=.7)
h2, = axis1.plot(gtab.x[gtab.dep == 150.], gtab.y[gtab.dep == 150.], 'o', mec='yellow', mfc='none', markersize=.5, linewidth=.001, label='150 km', alpha=.7)
h1, = axis1.plot(gtab.x[gtab.dep == 33.], gtab.y[gtab.dep == 33.], 'o', mec='red', mfc='none', markersize=.1, linewidth=.001, label='33 km', alpha=.7)
# local area of interest
x, y = m1([rgrd[2], rgrd[2], rgrd[3], rgrd[3], rgrd[2]], [rgrd[0], rgrd[1], rgrd[1], rgrd[0], rgrd[0]])
axis1.plot(x, y, color='orange')
# show parallel and meridians
m1.drawparallels(np.arange(-90., 120., 30.), linewidth=.1, dashes=(None, None))
m1.drawmeridians(np.arange(0., 360., 60.), linewidth=.1, dashes=(None, None))
# legend
axis1.legend(handles=[h1, h2, h3, h4, h5], loc='best', fontsize=5)
# LOCAL
# define map bounadries and resolution
m2 = Basemap(projection='cyl', llcrnrlon=rgrd[2], llcrnrlat=rgrd[0],
             urcrnrlon=rgrd[3], urcrnrlat=rgrd[1], resolution='i', ax=axis2)
# draw map
m2.drawmapboundary(fill_color='none')
# fill continents
m2.fillcontinents(color='0.8', lake_color='white')
# show grid
axis2.plot(gtab.lon[gtab.dep == 33.], gtab.lat[gtab.dep == 33.], 'o', mec='red', mfc='none', markersize=5, linewidth=.1)
axis2.plot(gtab.lon[gtab.dep == 150.], gtab.lat[gtab.dep == 150.], 'o', mec='orange', mfc='none', markersize=10, linewidth=.1)
axis2.plot(gtab.lon[gtab.dep == 300.], gtab.lat[gtab.dep == 300.], 'o', mec='green', mfc='none', markersize=15, linewidth=.1)
axis2.plot(gtab.lon[gtab.dep == 450.], gtab.lat[gtab.dep == 450.], 'o', mec='blue', mfc='none', markersize=20, linewidth=.1)
axis2.plot(gtab.lon[gtab.dep == 600.], gtab.lat[gtab.dep == 600.], 'o', mec='purple', mfc='none', markersize=25, linewidth=.1)
# show parallel and meridians
m1.drawparallels(np.arange(-90., 120., 30.), linewidth=.1, dashes=(None, None))
m1.drawmeridians(np.arange(0., 360., 60.), linewidth=.1, dashes=(None, None))
# maximise figure
plt.get_current_fig_manager().full_screen_toggle()
# adjust plots
fig.subplots_adjust(left=.07, right=.98, wspace=.1)
# show or save figure
if fig_name:
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    print(f" Figure saved: {fig_name}")
    plt.close()
else:
    plt.show()
