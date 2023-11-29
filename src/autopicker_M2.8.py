import os
from os import path
import re
import matplotlib.pyplot as plt
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from datetime import datetime
from obspy.clients.fdsn import Client


def plot_map_auto(tab):
    """
    :param tab: DataFrame containing automatic and catalogue event parameters
    :return: nothing
    """
    # create figure: three maps for (1) in-net events and (2) out-net events
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=rmap[2], llcrnrlat=rmap[0],
                 urcrnrlon=rmap[3], urcrnrlat=rmap[1], resolution='i', ax=ax1)
    m2 = Basemap(projection='cyl', llcrnrlon=lmap[2], llcrnrlat=lmap[0],
                 urcrnrlon=lmap[3], urcrnrlat=lmap[1], resolution='i', ax=ax2)
    # draw map
    m1.drawmapboundary(fill_color='none')
    m2.drawmapboundary(fill_color='none')
    # fill continents
    m1.fillcontinents(color='0.8', lake_color='white')
    m2.fillcontinents(color='0.8', lake_color='white')
    # faults
    flts_id = open(f"/home/{os.environ['LOGNAME']}/.seiscomp/bna/ActiveFaults/activefaults.bna", 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            ax1.plot(flt.lon, flt.lat, c='black', linewidth=.1, zorder=1)
            ax2.plot(flt.lon, flt.lat, c='black', linewidth=.1, zorder=1)
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # plot events
    ax1.scatter(tab.Auto_Lon, tab.Auto_Lat, s=tab.Auto_M, c=tab.Auto_Depth, edgecolors='none', cmap='jet', vmin=0., vmax=100.)
    ax2.scatter(tab.Auto_Lon, tab.Auto_Lat, s=tab.Auto_M, c=tab.Auto_Depth, edgecolors='none', cmap='jet', vmin=0., vmax=100.)
    # highlight Tele/Reg events
    ax2.scatter(tab[tab.Auto_ID.str.contains('tele')].Auto_Lon, tab[tab.Auto_ID.str.contains('tele')].Auto_Lat,
                s=tab[tab.Auto_ID.str.contains('tele')].Auto_M, facecolors='none', edgecolors='magenta', linewidth=.5)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.05, bottom=.05, wspace=.05)
    # show figure
    plt.show()
    return


def plot_hist_detec(tab, zones):
    """
    :param tab: DataFrame containing automatic and catalogue event parameters
    :param zones: list of seismic zones for in/out of network selections
    :return: nothing
    """
    # select detected in-net events
    tab1 = tab[~pd.isna(tab.Auto_ID) & ~tab.Cat_M.isnull() & tab.Cat_Region.isin(zones)]
    # select detected out-of-net events
    tab2 = tab[~pd.isna(tab.Auto_ID) & ~tab.Cat_M.isnull() & ~tab.Cat_Region.isin(zones)]
    # add columns for origin parameter errors
    tab1 = tab1.assign(dLoc=pd.Series([np.nan] * len(tab1), dtype='float64'))\
        .assign(dTim=pd.Series([np.nan] * len(tab1), dtype='float64'))\
        .assign(dDep=pd.Series([np.nan] * len(tab1), dtype='float64'))\
        .assign(dMag=pd.Series([np.nan] * len(tab1), dtype='float64'))
    tab2 = tab2.assign(dLoc=pd.Series([np.nan] * len(tab2), dtype='float64'))\
        .assign(dTim=pd.Series([np.nan] * len(tab2), dtype='float64'))\
        .assign(dDep=pd.Series([np.nan] * len(tab2), dtype='float64'))\
        .assign(dMag=pd.Series([np.nan] * len(tab2), dtype='float64'))
    # calculate origin parameter errors
    # origin-time error
    tab1.dTim = (tab1.Auto_Date_Time-tab1.Cat_Date_Time).dt.total_seconds()
    tab2.dTim = (tab2.Auto_Date_Time-tab2.Cat_Date_Time).dt.total_seconds()
    Tbeg = -10.
    Tend = 10.
    Tstp = 1.
    # location error
    for i, ev in tab1.iterrows():
        dis = gdist.distance((ev.Cat_Lat, ev.Cat_Lon), (ev.Auto_Lat, ev.Auto_Lon))
        tab1.loc[i, 'dLoc'] = dis.km
    for i, ev in tab2.iterrows():
        dis = gdist.distance((ev.Cat_Lat, ev.Cat_Lon), (ev.Auto_Lat, ev.Auto_Lon))
        tab2.loc[i, 'dLoc'] = dis.km
    Lbeg = 0.
    Lend = 50.
    Lstp = 2.5
    # depth error
    tab1.dDep = tab1.Auto_Depth-tab1.Cat_Depth
    tab2.dDep = tab2.Auto_Depth-tab2.Cat_Depth
    Dbeg = -10.
    Dend = 10.
    Dstp = 1.
    # magnitude error
    tab1.dMag = tab1.Auto_M-tab1.Cat_M
    tab2.dMag = tab2.Auto_M-tab2.Cat_M
    Mbeg = -1.
    Mend = 1.
    Mstp = .1
    # prepare histogram variables
    Lbar = pd.DataFrame({'In-Net': pd.Series(dtype='float64'), 'Out-Net': pd.Series(dtype='float64')})
    tab1['Loc Error [km]'] = pd.cut(tab1['dLoc'], bins=np.arange(Lbeg-Lstp/2., Lend+Lstp, Lstp), include_lowest=True)
    Lbar['In-Net'] = tab1.groupby('Loc Error [km]')['Loc Error [km]'].count()
    tab2['Loc Error [km]'] = pd.cut(tab2['dLoc'], bins=np.arange(Lbeg-Lstp/2., Lend+Lstp, Lstp), include_lowest=True)
    Lbar['Out-Net'] = tab2.groupby('Loc Error [km]')['Loc Error [km]'].count()
    # origin-time error
    Tbar = pd.DataFrame({'In-Net': pd.Series(dtype='float64'), 'Out-Net': pd.Series(dtype='float64')})
    tab1['OT Error [s]'] = pd.cut(tab1['dTim'], bins=np.arange(Tbeg-Tstp/2., Tend+Tstp, Tstp), include_lowest=True)
    Tbar['In-Net'] = tab1.groupby('OT Error [s]')['OT Error [s]'].count()
    tab2['OT Error [s]'] = pd.cut(tab2['dTim'], bins=np.arange(Tbeg-Tstp/2., Tend+Tstp, Tstp), include_lowest=True)
    Tbar['Out-Net'] = tab2.groupby('OT Error [s]')['OT Error [s]'].count()
    # depth error
    Dbar = pd.DataFrame({'In-Net': pd.Series(dtype='float64'), 'Out-Net': pd.Series(dtype='float64')})
    tab1['Dep Error [km]'] = pd.cut(tab1['dDep'], bins=np.arange(Dbeg-Dstp/2., Dend+Dstp, Dstp), include_lowest=True)
    Dbar['In-Net'] = tab1.groupby('Dep Error [km]')['Dep Error [km]'].count()
    tab2['Dep Error [km]'] = pd.cut(tab2['dDep'], bins=np.arange(Dbeg-Dstp/2., Dend+Dstp, Dstp), include_lowest=True)
    Dbar['Out-Net'] = tab2.groupby('Dep Error [km]')['Dep Error [km]'].count()
    # magnitude error
    Mbar = pd.DataFrame({'In-Net': pd.Series(dtype='float64'), 'Out-Net': pd.Series(dtype='float64')})
    tab1['Mag Error'] = pd.cut(tab1['dMag'], bins=np.arange(Mbeg-Mstp/2., Mend+Mstp, Mstp), include_lowest=True)
    Mbar['In-Net'] = tab1.groupby('Mag Error')['Mag Error'].count()
    tab2['Mag Error'] = pd.cut(tab2['dMag'], bins=np.arange(Mbeg-Mstp/2., Mend+Mstp, Mstp), include_lowest=True)
    Mbar['Out-Net'] = tab2.groupby('Mag Error')['Mag Error'].count()

    # create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    # BAR HISTOGRAMS
    # origin-time error
    Tbar[['In-Net', 'Out-Net']].plot(kind='bar', rot=0, width=.8, ax=ax1)
    ax1.set_xticklabels(np.arange(Tbeg, Tend+Tstp, Tstp))
    ax1.set_xlabel(ax1.get_xlabel(), fontsize=8)
    ax1.set_ylabel('Number of events', fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=5)
    ax1.legend(fontsize=8)
    # location error
    Lbar[['In-Net', 'Out-Net']].plot(kind='bar', rot=0, width=.8, ax=ax2)
    ax2.set_xticklabels(np.arange(Lbeg, Lend+Lstp, Lstp))
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=5)
    ax2.legend(fontsize=8)
    # depth error
    Dbar[['In-Net', 'Out-Net']].plot(kind='bar', rot=0, width=.8, ax=ax3)
    ax3.set_xticklabels(np.arange(Dbeg, Dend+Dstp, Dstp))
    ax3.set_xlabel(ax3.get_xlabel(), fontsize=8)
    ax3.set_ylabel('Number of events', fontsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=5)
    ax3.legend(fontsize=8)
    # magnitude error
    Mbar[['In-Net', 'Out-Net']].plot(kind='bar', rot=0, width=.8, ax=ax4)
    ax4.set_xticklabels(np.arange(Mbeg*10, Mend*10+Mstp*10, Mstp*10)/10.)
    ax4.set_xlabel(ax4.get_xlabel(), fontsize=8)
    ax4.tick_params(axis='both', which='major', labelsize=5)
    ax4.legend(fontsize=8)
    # adjust plot
    fig.subplots_adjust(left=.05, right=.98, top=.97, bottom=.07, wspace=.1)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show plot
    plt.show()
    return


def plot_maps_detec(tab, zones, lim1, lim2, lim3):
    """
    :param tab: DataFrame containing automatic and catalogue event parameters
    :param zones: list of seismic zones for in/out of network selections
    :param lim1: upper magnitude limited
    :param lim2: intermediate magnitude limit
    :param lim3: lower magnitude limit
    :return: nothing
    """
    # create figure: three maps for (1) stations, (2) in-net events and (3) out-net events
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200)
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=lmap[2], llcrnrlat=lmap[0],
                 urcrnrlon=lmap[3], urcrnrlat=lmap[1], resolution='i', ax=ax1)
    m2 = Basemap(projection='cyl', llcrnrlon=lmap[2]+2., llcrnrlat=lmap[0]+2.,
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
            ax1.plot(flt.lon, flt.lat, c='black', linewidth=.1, zorder=1)
            ax2.plot(flt.lon, flt.lat, c='black', linewidth=.1, zorder=1)
            ax3.plot(flt.lon, flt.lat, c='black', linewidth=.1, zorder=1)
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
    h11 = ax1.scatter(tab1.Longitude, tab1.Latitude, s=4, c='red', marker='^',
                      edgecolors='none', label=f'Seismometer ({len(tab1)})')
    h12 = []
    if tab.Auto_Date_Time.max() >= datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S'):
        # plot acceletometers
        tab2 = stab[stab.Channel == 'EN']
        h12 = ax1.scatter(tab2.Longitude, tab2.Latitude, s=4, c='blue', marker='s',
                          edgecolors='none', label=f'Accelerometer ({len(tab2)})')
    # title
    ax1.title.set_text('Station inventory')
    # legend
    h1 = [h11]
    if h12:
        h1.append(h12)
    ax1.legend(handles=h1, loc='lower right', fontsize=4)
    # IN-NETWORK (ACCORDING TO SEISMIC ZONES)
    # show seismic zones
    for i, z in enumerate(izon):
        ax2.text(lmap[2]+2.03, lmap[1]-2.57-i*0.07, z, ha='left', va='center', fontsize=5)
    # plot area of middle map
    ax1.plot([lmap[2]+2, lmap[2]+2, lmap[3]-2, lmap[3]-2, lmap[2]+2],
             [lmap[0]+2., lmap[1]-2.5, lmap[1]-2.5, lmap[0]+2., lmap[0]+2.], color='saddlebrown', linewidth=1)
    # select detected in-net events
    tab1 = tab[~pd.isna(tab.Auto_ID) & ~pd.isna(tab.Cat_M) & tab.Cat_Region.isin(zones)]
    # plot M≥[lim*] detected in-net events
    h21 = ax2.scatter(tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)].Cat_Lon,
                      tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)].Cat_Lat, s=4, c='black', marker='.', linewidth=.5,
                      alpha=.7, label=f'True {lim3}≤M<{lim2} EQ ({len(tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)])})')
    h22 = ax2.scatter(tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)].Cat_Lon,
                      tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)].Cat_Lat, s=4, c='lime', edgecolors='none',
                      alpha=.7, label=f'True {lim2}≤M<{lim1} EQ ({len(tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)])})')
    h23 = ax2.scatter(tab1[tab1.Cat_M >= lim1].Cat_Lon, tab1[tab1.Cat_M >= lim1].Cat_Lat, s=16, c='green',
                      edgecolors='none', alpha=.7, label=f'True M≥{lim1} EQ ({len(tab1[tab1.Cat_M >= lim1])})')
    # select missed in-net events
    tab2 = tab[pd.isna(tab.Auto_ID) & ~pd.isna(tab.Cat_M) & tab.Cat_Region.isin(zones)]
    # plot M≥[lim*] missed in-net events
    h24 = ax2.scatter(tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)].Cat_Lon,
                      tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)].Cat_Lat, s=4, c='yellow', marker='.', linewidth=.5,
                      alpha=.7, label=f'Missed {lim3}≤M<{lim2} EQ ({len(tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)])})')
    h25 = ax2.scatter(tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)].Cat_Lon,
                      tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)].Cat_Lat, s=4, c='orange', marker='x', linewidth=.5,
                      alpha=.7, label=f'Missed {lim2}≤M<{lim1} EQ ({len(tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)])})')
    h26 = ax2.scatter(tab2[tab2.Cat_M >= lim1].Cat_Lon, tab2[tab2.Cat_M >= lim1].Cat_Lat, s=16, c='red',
                      marker='x', alpha=.7, label=f'Missed M≥{lim1} EQ ({len(tab2[tab2.Cat_M >= lim1])})')
    # detected in-net felt events
    tab3 = tab[~pd.isna(tab.Auto_ID) & (tab.Cat_Type == 'F') & tab.Cat_Region.isin(zones)]
    h27 = ax2.scatter(tab3.Cat_Lon, tab3.Cat_Lat, s=36, c='red',
                      marker='*', edgecolors='black', linewidth=.7, alpha=.7, label=f'True felt EQ ({len(tab3)})')
    # missed in-net felt events
    tab4 = tab[pd.isna(tab.Auto_ID) & (tab.Cat_Type == 'F') & tab.Cat_Region.isin(zones)]
    h28 = ax2.scatter(tab4.Cat_Lon, tab4.Cat_Lat, s=36, c='red',
                      marker='*', edgecolors='orange', linewidth=.7, alpha=.7, label=f'Missed felt EQ ({len(tab4)})')
    # in-net events detected by tele/reg detector
    tab5 = tab[tab.Auto_ID.str.contains('tele') & ~pd.isna(tab.Cat_Lat) & ~pd.isna(tab.Auto_ID) & tab.Cat_Region.isin(zones)]
    if not tab5.empty:
        for _, ev in tab5.iterrows():
            ax2.text(ev.Cat_Lon, ev.Cat_Lat, 'T', fontsize=5, weight='bold', alpha=.5,
                     color='purple', clip_on=False, ha='center', va='bottom')
    # in-net events with duplicate(s)
    tab6 = tab[tab.Cat_ID.isin(dble) & ~pd.isna(tab.Cat_Lat) & ~pd.isna(tab.Auto_ID) & tab.Cat_Region.isin(zones)]
    if not tab6.empty:
        for _, ev in tab6.iterrows():
            ax2.text(ev.Cat_Lon, ev.Cat_Lat, 'D', fontsize=5, weight='bold', alpha=.5,
                     color='orange', clip_on=False, ha='center', va='bottom')
    # in-net detected explosions
    tab7 = tab[~pd.isna(tab.Auto_ID) & tab.Cat_Type.str.match('EX') & ~pd.isna(tab.Cat_Lat) & tab.Cat_Region.isin(zones)]
    h29 = ax2.scatter(tab7.Cat_Lon, tab7.Cat_Lat, s=9,
                      edgecolors='blue', facecolor='none', linewidth=.3, alpha=.7, label=f'True EX ({len(tab7)})')
    # in-net missed explosions
    tab8 = tab[pd.isna(tab.Auto_ID) & tab.Cat_Type.str.match('EX') & ~pd.isna(tab.Cat_Lat) & tab.Cat_Region.isin(zones)]
    h20 = ax2.scatter(tab8.Cat_Lon, tab8.Cat_Lat, s=9,
                      edgecolors='red', facecolor='none', linewidth=.3, alpha=.7, label=f'Missed EX ({len(tab8)})')
    # title
    ax2.title.set_text('In-network events')
    # legend
    h2 = []
    if h21:
        h2.append(h21)
    if h22:
        h2.append(h22)
    if h23:
        h2.append(h23)
    if h24:
        h2.append(h24)
    if h25:
        h2.append(h25)
    if h26:
        h2.append(h26)
    if h27:
        h2.append(h27)
    if h28:
        h2.append(h28)
    if h29:
        h2.append(h29)
    if h20:
        h2.append(h20)
    ax2.legend(handles=h2, loc='lower right', bbox_to_anchor=(1.2, .02), fontsize=4)
    # OUT-OF-NETWORK (ACCORDING TO SEISMIC ZONES)
    # plot area of middle map
    ax3.plot([lmap[2]+2, lmap[2]+2, lmap[3]-2, lmap[3]-2, lmap[2]+2],
             [lmap[0]+2., lmap[1]-2.5, lmap[1]-2.5, lmap[0]+2., lmap[0]+2.], color='saddlebrown', linewidth=1)
    # select detected out-of-net events
    tab1 = tab[~pd.isna(tab.Auto_ID) & ~pd.isna(tab.Cat_M) & ~tab.Cat_Region.isin(zones)]
    # plot M≥[lim*] detected out-of-net events
    h31 = ax3.scatter(tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)].Cat_Lon,
                      tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)].Cat_Lat, s=4, c='black', marker='.', linewidth=.5,
                      alpha=.7, label=f'True {lim3}≤M<{lim2} EQ ({len(tab1[(tab1.Cat_M >= lim3) & (tab1.Cat_M < lim2)])})')
    h32 = ax3.scatter(tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)].Cat_Lon,
                      tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)].Cat_Lat, s=4, c='lime', edgecolors='none',
                      alpha=.7, label=f'True {lim2}≤M<{lim1} EQ ({len(tab1[(tab1.Cat_M >= lim2) & (tab1.Cat_M < lim1)])})')
    h33 = ax3.scatter(tab1[tab1.Cat_M >= lim1].Cat_Lon, tab1[tab1.Cat_M >= lim1].Cat_Lat, s=16, c='green',
                      edgecolors='none', alpha=.7, label=f'True M≥{lim1} EQ ({len(tab1[tab1.Cat_M >= lim1])})')
    # select missed out-of-net events
    tab2 = tab[pd.isna(tab.Auto_ID) & ~pd.isna(tab.Cat_M) & ~tab.Cat_Region.isin(zones)]
    # plot M≥[lim*] missed out-of-net events
    h34 = ax3.scatter(tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)].Cat_Lon,
                      tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)].Cat_Lat, s=4, c='yellow', marker='.', linewidth=.5,
                      alpha=.7, label=f'Missed {lim3}≤M<{lim2} EQ ({len(tab2[(tab2.Cat_M >= lim3) & (tab2.Cat_M < lim2)])})')
    h35 = ax3.scatter(tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)].Cat_Lon,
                      tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)].Cat_Lat, s=4, c='orange', marker='x', linewidth=.5,
                      alpha=.7, label=f'Missed {lim2}≤M<{lim1} EQ ({len(tab2[(tab2.Cat_M >= lim2) & (tab2.Cat_M < lim1)])})')
    h36 = ax3.scatter(tab2[tab2.Cat_M >= lim1].Cat_Lon, tab2[tab2.Cat_M >= lim1].Cat_Lat, s=16, c='red',
                      marker='x', alpha=.7, label=f'Missed M≥{lim1} EQ ({len(tab2[tab2.Cat_M >= lim1])})')
    # detected out-of-net felt events
    tab3 = tab[~pd.isna(tab.Auto_ID) & (tab.Cat_Type == 'F') & ~tab.Cat_Region.isin(zones)]
    h37 = ax3.scatter(tab3.Cat_Lon, tab3.Cat_Lat, s=36, c='red',
                      marker='*', edgecolors='black', linewidth=.7, alpha=.7, label=f'True felt EQ ({len(tab3)})')
    # missed out-of-net felt events
    tab4 = tab[pd.isna(tab.Auto_ID) & (tab.Cat_Type == 'F') & ~tab.Cat_Region.isin(zones)]
    h38 = ax3.scatter(tab4.Cat_Lon, tab4.Cat_Lat, s=36, c='red',
                      marker='*', edgecolors='orange', linewidth=.7, alpha=.7, label=f'Missed felt EQ ({len(tab4)})')
    # out-of-net events detected by tele/reg detector
    tab5 = tab[tab.Auto_ID.str.contains('tele') & ~pd.isna(tab.Cat_Lat) & ~pd.isna(tab.Auto_ID) & ~tab.Cat_Region.isin(zones)]
    if not tab5.empty:
        for _, ev in tab5.iterrows():
            ax3.text(ev.Cat_Lon, ev.Cat_Lat, 'T', fontsize=5, weight='bold', alpha=.5,
                     color='purple', clip_on=False, ha='center', va='bottom')
    # out-of-net events with duplicate(s)
    tab6 = tab[tab.Cat_ID.isin(dble) & ~pd.isna(tab.Cat_Lat) & ~pd.isna(tab.Auto_ID) & ~tab.Cat_Region.isin(zones)]
    if not tab6.empty:
        for _, ev in tab6.iterrows():
            ax3.text(ev.Cat_Lon, ev.Cat_Lat, 'D', fontsize=5, weight='bold', alpha=.5,
                     color='orange', clip_on=False, ha='center', va='bottom')
    # out-of-net detected explosions
    tab7 = tab[~pd.isna(tab.Auto_ID) & tab.Cat_Type.str.match('EX') & ~pd.isna(tab.Cat_Lat) & ~tab.Cat_Region.isin(zones)]
    h39 = ax3.scatter(tab7.Cat_Lon, tab7.Cat_Lat, s=9,
                      edgecolors='blue', facecolors='none', linewidth=.3, alpha=.7, label=f'True EX ({len(tab7)})')
    # out-of-net missed explosions
    tab8 = tab[pd.isna(tab.Auto_ID) & tab.Cat_Type.str.match('EX') & ~pd.isna(tab.Cat_Lat) & ~tab.Cat_Region.isin(zones)]
    h30 = ax3.scatter(tab8.Cat_Lon, tab8.Cat_Lat, s=9,
                      edgecolors='red', facecolors='none', linewidth=.3, alpha=.7, label=f'Missed EX ({len(tab8)})')
    # title
    ax3.title.set_text('Out-of-network events')
    # legend
    h3 = []
    if h31:
        h3.append(h31)
    if h32:
        h3.append(h32)
    if h33:
        h3.append(h33)
    if h34:
        h3.append(h34)
    if h35:
        h3.append(h35)
    if h36:
        h3.append(h36)
    if h37:
        h3.append(h37)
    if h38:
        h3.append(h38)
    if h39:
        h3.append(h39)
    if h30:
        h3.append(h30)
    ax3.legend(handles=h3, loc='lower right', fontsize=4)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.05, bottom=.05, wspace=.05)
    # show figure
    plt.show()
    return


# working directory
wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker'

# read input table
itab = pd.read_csv(f'{wdir}/autopicker_M2.8_5.7_28.11.csv', parse_dates=['Auto Date Time', 'Cat Date Time'])
# replace blanks with '_' in column names
itab.columns = itab.columns.str.replace(' ', '_')
# replace missing magnitude(s) ('-') with nans and convert to floats
itab.Auto_M = itab.Auto_M.replace('-', 'nan').astype(float)
itab.Cat_M = itab.Cat_M.replace('-', 'nan').astype(float)
# convert lat/lon columns (some empty -> strings)
itab.Auto_Lat = itab.Auto_Lat.astype(float)
itab.Auto_Lon = itab.Auto_Lon.astype(float)
itab.Cat_Lat = itab.Cat_Lat.astype(float)
itab.Cat_Lon = itab.Cat_Lon.astype(float)
# list of duplicate events
dble = itab[~pd.isna(itab.Cat_Type) & itab.Cat_Type.str.match('D') & itab.Cat_ID.str.contains('gsi')].Cat_ID.to_list()
# prepare event table of automatic events
atab = itab[~pd.isna(itab.Auto_M)]
# prepare event table of detections
otab = itab[itab.Cat_ID.str.contains('gsi') &
            ((itab.Cat_Type == 'EQ') | (itab.Cat_Type == 'F') | (itab.Cat_Type == 'EX'))].reset_index(drop=True)
# # select events for 1st time period (5.7-6.8): initial configuration
# otab = otab[((otab.Auto_Date_Time >= datetime.strptime('2023-07-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (otab.Auto_Date_Time < datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S'))) |
#             ((otab.Cat_Date_Time >= datetime.strptime('2023-07-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (otab.Cat_Date_Time < datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S')))]
# # select events for 2nd time period (7.8-20.9): min. no. of stations per event changed from 10 to 7
# atab = atab[((atab.Auto_Date_Time >= datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (atab.Auto_Date_Time < datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S'))) |
#             ((atab.Cat_Date_Time >= datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (atab.Cat_Date_Time < datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')))]
# otab = otab[((otab.Auto_Date_Time >= datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (otab.Auto_Date_Time < datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S'))) |
#             ((otab.Cat_Date_Time >= datetime.strptime('2023-08-07 00:00:00', '%Y-%m-%d %H:%M:%S')) &
#             (otab.Cat_Date_Time < datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')))]
# select events for 3rd time period (20.9-14.11): added 25 accelerometers
atab = atab[((atab.Auto_Date_Time >= datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')) &
            (atab.Auto_Date_Time < datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S'))) |
            ((atab.Cat_Date_Time >= datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')) &
            (atab.Cat_Date_Time < datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S')))]
# select events for 3rd time period (20.9-14.11): added 25 accelerometers
otab = otab[((otab.Auto_Date_Time >= datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')) &
            (otab.Auto_Date_Time < datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S'))) |
            ((otab.Cat_Date_Time >= datetime.strptime('2023-09-20 12:00:00', '%Y-%m-%d %H:%M:%S')) &
            (otab.Cat_Date_Time < datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S')))]
# # select events for 4th time period (14.11): lowered detection threshold from 4.5 to 2.5
# otab = otab[(otab.Auto_Date_Time >= datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S')) |
#             (otab.Cat_Date_Time >= datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S'))]
# atab = atab[(atab.Auto_Date_Time >= datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S')) |
#             (atab.Cat_Date_Time >= datetime.strptime('2023-11-14 11:00:00', '%Y-%m-%d %H:%M:%S'))]
# # write output table to check
# otab.to_csv(f'{wdir}/eq.csv')

# FIGURES
# magnitude limits
lim3 = 1.0
lim2 = 2.0
lim1 = 2.8
# geographical boundaries
lmap = [27., 36., 32., 38.]
rmap = [11., 51., 15., 55.]
# in-net seismogenic zones
izon = ['Arava', 'Arif fault', 'Barak fault', 'Carmel Tirza', 'Central Israel', 'Dead Sea Basin', 'East Shomron',
        'Eilat Deep', 'Galilee', 'Gaza', 'HaSharon', 'Hula Kinneret', 'Jordan Valley', 'Judea Samaria', 'Negev',
        'Paran', 'West Malhan']

# plot automatic events map
plot_map_auto(atab)
# # plot detection histograms
# plot_hist_detec(otab, izon)
# # plot detection maps
# plot_maps_detec(otab, izon, lim1, lim2, lim3)
