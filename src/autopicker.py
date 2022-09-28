########################################################################################################################
import xml.etree.ElementTree as ETree
import os
from os import path
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
import statistics
from scipy import signal
from datetime import datetime
from datetime import timedelta
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def read_autopick_xml(file_path, evt_dict, phase=None, picker=None):
    # ________________________________________________________ #
    # file_path: path to .XML file containing picks to read
    # phase: pick's seismic phase
    # picker: picker used
    # ________________________________________________________ #
    # initialise table
    ptab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                         'chn': pd.Series(dtype='string'), 'pha': pd.Series(dtype='string'), 'pic': pd.Series(dtype='float64'),
                         'pid': pd.Series(dtype='string')})
    # read resulting xml file
    root_xml = ETree.parse(file_path).getroot()
    for line_xml in root_xml[0]:
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
                str_net = str3[1]
            else:
                str_net = str3[2]
            if str2[4] is None:
                str_loc = ''
            else:
                str_loc = str2[4]
            # fill output table
            ptab.loc[ptab.shape[0]] = [str_net, str2[3], str_loc, str2[5], line_xml[4].text,
                                       (datetime.strptime(str(line_xml[0][0].text), '%Y-%m-%dT%H:%M:%S.%fZ') -
                                        evt_dict['eori']).total_seconds(), str1]
    ptab = ptab.assign(snr=pd.Series([None] * len(ptab), dtype='float'))
    # read second time to get matching SNR values (because file is not sorted)
    for line_xml in root_xml[0]:
        if re.search('amplitude', str(line_xml.tag)):
            # check picker
            if picker and not re.search(picker, str(line_xml.attrib)):
                continue
            p = re.search("'publicID': '(.*?)'", str(line_xml.attrib)).group(1)
            pid = p.split('.snr')[0]
            ip = ptab.index[ptab.pid == pid].to_list()
            ptab.loc[ip, 'snr'] = float(line_xml[1][0].text)
    return ptab


def plot_autopick_sec(stream, auto_tab, hand_tab, evt_param, filt_param, index=None, fig_name=None):
    # ________________________________________________________ #
    # stream: data streamer
    # evt_ot: event origin time to use as reference (UTCDateTime)
    # ________________________________________________________ #
    # map area (slightly different from main code)
    # plotting mode
    if fig_name:
        mpl.use('Agg')
    # define axis limits
    theo_tab = [s_tr.stats.theo_tt for s_tr in stream]
    if min(theo_tab)-10 >= 0.:
        tmin = min(theo_tab) - 10.
    else:
        tmin = 0.
    xlim1 = [tmin, max(theo_tab)+50.]
    xlim2 = [-5., 5.]
    # create figure & axis
    ff, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200,
                                             gridspec_kw={'width_ratios': [2, 1, 1]})
    if not fig_name:
        plt.show(block=False)
    # AXIS 1: VELOCITY WAVEFORMS
    axis1.grid(which='both', axis='both')
    axis1.set_xlim(xlim1)
    # AXIS 2: RESIDUAL OVER DISTANCE PLOT
    axis2.grid(which='both', axis='both')
    axis2.set_xlim(xlim2)
    # AXIS 3: LOCAL STATIONS & EVENT MAP
    m = Basemap(projection='cyl', llcrnrlon=mgrd[2]+.5, llcrnrlat=mgrd[0], urcrnrlon=mgrd[3]-.5, urcrnrlat=mgrd[1], resolution='l', ax=axis3)
    # draw map
    m.drawmapboundary(fill_color='none')
    m.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels = [left,right,top,bottom])
    m.drawparallels(np.arange(m.llcrnrlat, m.urcrnrlat + 1, 2.), labels=[True, False, True, False])
    m.drawmeridians(np.arange(m.llcrnrlon, m.urcrnrlon + 1, 2.), labels=[True, False, False, True])
    n_hpic = 0
    n_apic = 0
    n_tres = 0
    n_theo = 0
    tres_tab = []
    theo_tab = []
    h1 = []
    h2 = []
    h3 = []
    h4 = []
    # initialise counter
    n_trace = 0
    # loop over stream channels
    stn_lbl = []
    # sorting index (if any)
    if index is not None:
        ind = index
    else:
        ind = range(len(stream))
    for jjj in ind:
        stn_lbl.append(f"{stream[jjj].stats.network}.{stream[jjj].stats.station}."
                       f"{stream[jjj].stats.location}.{stream[jjj].stats.channel}")
        n_trace += 1
        # time vector
        t_vec = stream[jjj].times('relative', reftime=UTCDateTime(evt_param['eori']))
        # plot waveforms
        h1, = axis1.plot(t_vec, stream[jjj].data / stream[jjj].max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # theoretical travel time
        if not np.isnan(stream[jjj].stats.theo_tt):
            h2, = axis1.plot([stream[jjj].stats.theo_tt, stream[jjj].stats.theo_tt], [n_trace - 1, n_trace + 1],
                             color='blue', linestyle='dotted', label=vel_mod)
        # show hand picks
        k_hpic = None
        if not hand_tab.empty:
            if not hand_tab[(hand_tab.sta == stream[jjj].stats.station)].empty:
                k_hpic = hand_tab.index[(hand_tab.sta == stream[jjj].stats.station)
                                        & (hand_tab.net == stream[jjj].stats.network) & (hand_tab.chn == stream[jjj].stats.channel)].to_list()
                if len(k_hpic) == 1:
                    k_hpic = k_hpic[0]
                else:
                    continue
                if k_hpic is not None:
                    n_hpic += 1
                    # show on both axes
                    h3, = axis1.plot([hand_tab.pic[k_hpic], hand_tab.pic[k_hpic]], [n_trace - 1, n_trace + 1], color='orange', label='Handpick')
                    axis1.plot(axis1.get_xlim()[1]+(axis1.get_xlim()[1]-axis1.get_xlim()[0])/25., n_trace, 'o',
                               markersize=5, mfc='orange', mec='none', alpha=.7, clip_on=False)
                    # station name (MAP)
                    for st in isn_inv.networks[0]:
                        if stream[jjj].stats.station == st.code:
                            axis3.plot(st.longitude, st.latitude, 's', markersize=7, color='orange', mfc='none', alpha=.7, label='Autopick')
        # show automatic picks
        k_apic = None
        if not auto_tab.empty:
            if not auto_tab[(auto_tab.sta == stream[jjj].stats.station)].empty:
                # indexing
                temp_tab = auto_tab[(auto_tab.sta == stream[jjj].stats.station)
                                    & (auto_tab.net == stream[jjj].stats.network) & (auto_tab.chn == stream[jjj].stats.channel)]
                temp_tab = temp_tab.assign(tdif=pd.Series([None] * len(temp_tab), dtype='float'))
                if temp_tab.empty:
                    continue
                if len(tab) > 1:
                    k_apic = None
                    # find pick closest to theoretical arrival
                    temp_tab['tdif'] = [abs(xx - stream[jjj].stats.theo_tt) for xx in temp_tab.pic.to_list()]
                    for kk in temp_tab.index:
                        if temp_tab.pic[kk] > 0 and kk == temp_tab['tdif'].idxmin():
                            k_apic = kk
                            break
                else:
                    k_apic = temp_tab.index[0]
                n_apic += 1
                for kk in temp_tab.index:
                    # show on both axes
                    h4, = axis1.plot([temp_tab.pic[kk], temp_tab.pic[kk]], [n_trace - 1, n_trace + 1], color='purple', label='Autopick')
                    axis1.plot(axis1.get_xlim()[1]+(axis1.get_xlim()[1]-axis1.get_xlim()[0])/40., n_trace, 'o',
                               markersize=5, mfc='purple', mec='none', alpha=.7, clip_on=False)
                    # station name (MAP)
                    for st in isn_inv.networks[0]:
                        if stream[jjj].stats.station == st.code:
                            axis3.plot(st.longitude, st.latitude, 'o', markersize=7, color='purple', mfc='none', alpha=.7)
        # residual w.r.t. hand pick (if both exist)
        if k_hpic is not None and k_apic is not None:
            # table for residuals (for statistics)
            tres_tab.append(auto_tab.pic[k_apic] - hand_tab.pic[k_hpic])
            # residual plot
            axis2.plot(auto_tab.pic[k_apic] - hand_tab.pic[k_hpic], n_trace, 'o', markersize=5, mfc='orange', mec='none', alpha=.7)
            # # station names
            # axis2.text(auto_tab.pic[k_apic] - hand_tab.pic[k_hpic] - .1, n_trace, stream[jjj].stats.station,
            #            ha='right', va='center', clip_on=True, fontsize=5)
            n_tres += 1
        # residual w.r.t. theoretical pick (if automatic pick exists)
        if k_apic is not None:
            # table for residuals (for statistics)
            theo_tab.append(auto_tab.pic[k_apic] - stream[jjj].stats.theo_tt)
            # residual plot
            axis2.plot(auto_tab.pic[k_apic] - stream[jjj].stats.theo_tt, n_trace, 'o', markersize=5, mfc='blue', mec='none', alpha=.7)
            n_theo += 1
    # display statistics on residuals
    if n_tres > 1:
        axis2.text(.95 * xlim2[0], 4, f"N={n_tres}: {statistics.mean(tres_tab):.2f} \u00B1 {statistics.stdev(tres_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='orange')
    if n_theo > 1:
        axis2.text(.95 * xlim2[0], 1, f"N={n_theo}: {statistics.mean(theo_tab):.2f} \u00B1 {statistics.stdev(theo_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='blue')
    # axis limits
    axis1.set_ylim([0, n_trace + 1])
    axis2.set_ylim([0, n_trace + 1])
    # legend
    if not h4:
        if not h3:
            axis1.legend(handles=[h1, h2], loc='lower left', fontsize=8)
        else:
            axis1.legend(handles=[h1, h2, h3], loc='lower left', fontsize=8)
    else:
        if not h3:
            axis1.legend(handles=[h1, h2, h4], loc='lower left', fontsize=8)
        else:
            axis1.legend(handles=[h1, h2, h3, h4], loc='lower left', fontsize=8)
    # station and pick numbers
    axis1.text(xlim1[0] - .025 * xlim1[1], 1.01 * axis1.get_ylim()[1], f"N={n_trace}", ha='right', va='center')
    # replace numerical tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace+1, 1))
    axis1.set_yticklabels(stn_lbl, fontsize=5)
    axis2.set_yticklabels([])
    # axis labels
    axis1.set_xlabel('Time - OT [s]', fontweight='bold')
    axis1.set_ylabel('Station', fontweight='bold')
    axis2.set_xlabel('\u0394t [s]', fontweight='bold')
    # MAP
    # fault lines
    fid = open(f"{wdir}/../Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz", 'r')
    flts = fid.readlines()
    fid.close()
    flt_x = []
    flt_y = []
    hf = []
    for iii in range(len(flts)):
        if re.search('NaN', flts[iii]):
            flt_x = []
            flt_y = []
        elif iii < len(flts) - 1 and re.search('NaN', flts[iii + 1]):
            hf, = axis3.plot(flt_x, flt_y, '.6', label='Faults')
        else:
            l_line = flts[iii].split()
            flt_x.append(float(l_line[0]))
            flt_y.append(float(l_line[1]))
    # stations
    hm1 = []
    hm2 = []
    for jjj in ind:
        for st in isn_inv.networks[0]:
            if stream[jjj].stats.station == st.code:
                if stream[jjj].stats.network == 'IS':
                    hm1, = axis3.plot(st.longitude, st.latitude, 'b^', markersize=5, alpha=.7, mec='none', label=stream[jjj].stats.network)
                elif stream[jjj].stats.network == 'GE':
                    hm2, = axis3.plot(st.longitude, st.latitude, 'cs', markersize=5, alpha=.7, mec='none', label=stream[jjj].stats.network)
    # event
    he, = axis3.plot(evt_param['elon'], evt_param['elat'], 'r*', markersize=10, markeredgecolor='black', label='Event')
    if evt_param['emag']:
        axis3.text(evt_param['elon'] + .1, evt_param['elat'] - .1, f"M{evt_param['emag']:3.1f}",
                   ha='left', va='bottom', color='red', clip_on=True, fontsize=8)
    # legend
    if not hm1:
        hm = [hm2, he, hf]
    elif not hm2:
        hm = [hm1, he, hf]
    else:
        hm = [hm1, hm2, he, hf]
    axis3.legend(handles=hm, loc='upper left', fontsize=8)
    # AXIS 4: INSET MAP
    axis4 = inset_axes(axis3, '30%', '18%', loc='lower left')
    m = Basemap(projection='cyl', llcrnrlon=rgrd[2], llcrnrlat=rgrd[0], urcrnrlon=rgrd[3], urcrnrlat=rgrd[1], resolution='l', ax=axis4)
    # draw map
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8', lake_color='white')
    # area of interest
    axis4.plot([mgrd[2]+.5, mgrd[2]+.5, mgrd[3]-.5, mgrd[3]-.5, mgrd[2]+.5], [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]], 'r')
    # distance from centre of local map
    axis4.set_title(f"{gdist.distance((mgrd[0]+(mgrd[1]-mgrd[0])/2., mgrd[2]+(mgrd[3]-mgrd[2])/2.), (evt_param['elat'], evt_param['elon'])).km:.2f} km")
    # event
    axis4.plot(evt_param['elon'], evt_param['elat'], 'r*', markersize=10, markeredgecolor='black')
    # figure title
    tit1 = f"{evt_param['eori'].strftime('%d/%m/%Y %H:%M:%S')} \u2013 {evt_param['edep']:.2f} km \u2013 M{evt_param['emag']:3.1f}"
    tit2 = f"HP: {filt_param['rmhp']:.2f} [s] \u2013 Taper: {filt_param['taper']:.2f} [s] \u2013 BP: " \
           f"{filt_param['bworder']} / {filt_param['bwminf']:.2f} / {filt_param['bwmaxf']:.2f} [Hz]"
    tit3 = f"STA/LTA: {filt_param['sta']:.2f} / {filt_param['lta']:.2f} [s] " \
           f"\u2013 Trigger: {filt_param['trigon']:.2f} / {filt_param['trigoff']:.2f}"
    ff.suptitle(tit1 + '\n' + tit2 + '\n' + tit3, fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    ff.subplots_adjust(left=.07, right=.98, wspace=.1)
    # show or save figure
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()
    return tres_tab, n_tres, theo_tab, n_theo


########################################################################################################################
# input parameters
# selected events
# evt = '20210208205947606'         # M2.0
# evt = '20210126194914826'         # M2.0
# evt = '20210122064437843'         # M2.2
# evt = '20210717101444540'         # M2.3
# evt = '20210628231231690'         # M2.5
# evt = '20210511125045570'         # M3.1
# evt = '20210615230854375'         # M4.1
exp = 0
ntw = 'GE, IS'
chn = '(B|H|E)(H|N)Z'
pic = 'AIC'

# area of interest
rrad = 6371.
igrd = [29., 34., 34., 36.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 25., 45.]

# working directory
wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker'
mpl.rcParams['savefig.directory'] = f"{wdir}/fig"
# data archive directory
adir = '/net/jarchive/archive/jqdata/archive'
# input data
idat = '01-06-2021_01-06-2022_M3'

# FDSN database
# isn_client = Client('http://172.16.46.102:8181/')       # jfdsn
# isn_client = Client('http://172.16.46.140:8181/')       # jtfdsn

# initialising TauP
vel_mod = 'gitt05'
model = TauPyModel(model=vel_mod)

# autopicker parameters
# varying: Def  Frequenccy               STA/LTA                  Thresholds               Order?
# exp:     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16 ...
# rmhp =    [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]
# taper =   [30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.]
# bworder = [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4]
# bwminf =  [.7,  1.,  2.,  3.,  4.,  1.,  1.,  4.,  3.,  4.,  1.,  1.,  2.,  3.,  4.,  1.]
# bwmaxf =  [2.,  3.,  4.,  6.,  8.,  5.,  3.,  8.,  6.,  8.,  5.,  3.,  4.,  6.,  8.,  5.]
# sta =     [2.,  2.,  2.,  2.,  2.,  2.,  2.,  .2,  .4,  .1,  .5,  2.,  .2,  .4,  .1,  .5]
# lta =     [80., 80., 80., 80., 80., 80., 80., 10., 12., 4.,  3.,  80., 10., 12.,  4., 3.]
# trigon =  [3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  1.,  3.,  3.,  2.,  1.,  .5,  10.]
# trigoff = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .5,  1.5, 1.5, 1.,  2.,  1.,  1.]
# exp:     0    1(3) 2(4) 3(7)
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

print(f"HP: {fpar['rmhp']:.2f}")
print(f"Taper: {fpar['taper']:.2f}")
print(f"STA/LTA: {fpar['sta']:.2f} / {fpar['lta']:.2f}")
print(f"BW: {fpar['bworder']} / {fpar['bwminf']:.2f} / {fpar['bwmaxf']:.2f}")
print(f"Triggers: {fpar['trigon']:.2f} / {fpar['trigoff']:.2f}")
print()

# # choosing frequencies based on experiments
# if exp > 5:
#     if epar['emag'] >= 3.0:
#         fmin = 3.0
#         fmax = 6.0
#     else:
#         fmin = 4.0
#         fmax = 8.0
#     fpar['bwminf'] = fmin
#     fpar['bwmaxf'] = fmax
# if exp > 10:
#     if epar['emag'] >= 3.0:
#         swin = 2.0
#         lwin = 80.0
#     else:
#         swin = 0.2
#         lwin = 10.0
#     fpar['sta'] = swin
#     fpar['lta'] = lwin

# retrieve station inventory
if path.exists(f"{wdir}/inventory.xml") != 0:
    print('ISN inventory file already exists:')
    os.system(f"ls -lh {wdir}/inventory.xml")
    print()
else:
    isn_inv = isn_client.get_stations(network=ntw, channel='ENZ, HHZ, BHZ', level='response')
    isn_inv.write('%s/inventory.xml' % wdir, level='response', format='STATIONXML')
# read station inventory
isn_inv = read_inventory(f"{wdir}/inventory.xml", format='STATIONXML')
# isn_inv_new = read_inventory("{wdir}/Autopicker/inventory_sc.xml", format='STATIONXML')

# # import station inventory (SEISCOMP)
# if path.exists('%s/inventory_sc.xml' % (wdir)) != 0:
#     print('ISN inventory file already exists:')
#     os.system('ls -lh %s/inventory_sc.xml' % (wdir))
#     print()
# else:
#     cmd = 'scxmldump -fI -o %s/inventory_sc.xml -d postgresql://' % (wdir)
#     print(cmd)
#     print()
#     os.system(cmd)

# read M>3.0 events from FDSNws for period June 2021 - June 2022
etab = pd.read_csv(f"{wdir}/{idat}.csv", parse_dates=['OriginTime'])

########################################################################################################################
# TESTS WITH SPECTRA
if_yes = False
if if_yes:
    twin = 3.
    evt = '20220216032228454'
    eid = 'gsi202202160321'
    elat = 32.59742
    elon = 35.60145
    edep = 5.
    emag = 3.2
    # evt = '20210615230854375'
    # eid = 'gsi202106152307'
    # elat = 30.09941
    # elon = 35.17768
    # edep = 21.
    # emag = 4.1
    # e = isn_client.get_events(eventid=eid, includearrivals=False)[0].preferred_origin()
    # load data
    isn_traces = read(f"{wdir}/{idat}/{evt}.raw.mseed").merge()
    # initialise tables
    fdom = []
    evst = []
    for tr in isn_traces:
        print(f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}')
        stn = tr.stats.station
        chn = tr.stats.channel
        tr1 = isn_traces.select(station=stn, channel=chn)
        if not tr1:
            continue
        else:
            tr1 = tr1[0]
        # if path.exists(f'{wdir}/{idat}/{evt}/{tr1.stats.network}.{stn}.{chn}.png') != 0:
        #     continue
        # theoretical arrival
        st = isn_inv.select(station=stn, channel=chn)
        d = gdist.distance((elat, elon), (st.networks[0].stations[0].latitude, st.networks[0].stations[0].longitude))
        evst.append(d.km)
        tt = model.get_travel_times(source_depth_in_km=edep/1000., distance_in_degree=d.km / (2. * np.pi * rrad / 360.),
                                    phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])[0]
        # time vector
        tv = tr1.times('relative', reftime=UTCDateTime(etab.OriginTime[etab.EventID == eid].iloc[0]))
        # # FIGURE
        # mpl.use('Agg')
        # _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(18, 9), dpi=200)
        # ax1.grid(which='both', axis='both')
        # ax2.grid(which='both', axis='both')
        # # # TIME DOMAIN
        # # waveform
        # ax1.plot(tv, tr1.data, alpha=.7)
        # # theoretical arrival
        # ax1.plot([tt.time, tt.time], [-abs(tr1.max()), abs(tr1.max())], color='red', alpha=.7)
        # ax1.text(tt.time, abs(tr1.max())*1.05, tt.name, color='red', ha='center', va='bottom')
        # signal time window
        ts1 = tt.time - 1.
        ts2 = tt.time + twin
        # ax1.plot([ts1, ts1], [-abs(tr1.max()), abs(tr1.max())], color='orange', alpha=.7)
        # ax1.plot([ts2, ts2], [-abs(tr1.max()), abs(tr1.max())], color='orange', alpha=.7)
        # noise time window
        tn1 = tt.time - twin - 2.
        tn2 = tt.time - 1.
        # ax1.plot([tn1, tn1], [-abs(tr1.max()), abs(tr1.max())], '--', color='purple', alpha=.7)
        # ax1.plot([tn2, tn2], [-abs(tr1.max()), abs(tr1.max())], '--', color='purple', alpha=.7)
        # time window indexes
        is1 = list(abs(tv-ts1)).index(min(abs(tv-ts1)))
        is2 = list(abs(tv-ts2)).index(min(abs(tv-ts2)))
        in1 = list(abs(tv-tn1)).index(min(abs(tv-tn1)))
        in2 = list(abs(tv-tn2)).index(min(abs(tv-tn2)))
        # # highlight waveforms in windows
        # ax1.plot(tv[is1:is2], tr1.data[is1:is2], ':', color='orange', alpha=.7)
        # ax1.plot(tv[in1:in2], tr1.data[in1:in2], ':', color='purple', alpha=.7)
        # # labels
        # ax1.set_xlabel('Time [s]', fontweight='bold')
        # ax1.set_ylabel('Velocity [m/s]', fontweight='bold')
        # ax1.set_xlim([-50., 100.])
        # ax1.set_title(f'{stn}.{chn}', fontweight='bold', fontsize=15)
        # # FREQUENCY DOMAIN
        # power spectral density
        # f, Pxx = signal.periodogram(tr1.data, tr1.stats.sampling_rate)
        f_s, Pxx_s = signal.periodogram(tr1.data[is1:is2], tr1.stats.sampling_rate)
        f_n, Pxx_n = signal.periodogram(tr1.data[in1:in2], tr1.stats.sampling_rate)
        # remove 0-frequency
        f_s = f_s[1:]
        Pxx_s = Pxx_s[1:]
        f_n = f_n[1:]
        Pxx_n = Pxx_n[1:]
        # # ax2.loglog(f, Pxx, alpha=.7, label='All')
        # ax2.loglog(f_s, Pxx_s, alpha=.7, color='orange', label='Signal')
        # ax2.loglog(f_n, Pxx_n, alpha=.7, color='purple', label='Noise')
        # log difference between noise and signal spectra
        dlog = list(np.log(Pxx_s)-np.log(Pxx_n))
        # display max difference
        im = dlog.index(max(dlog))
        fdom.append(f_s[im])
        # ax2.plot([f_s[im], f_s[im]], ax2.get_ylim(), color='red')
        # ax2.text(f_s[im]*1.05, ax2.get_ylim()[0]*10, f'fmax={f_s[im]} Hz', ha='left', va='bottom', color='red', fontsize=8)
        # # labels
        # ax2.set_xlabel('Frequency [Hz]', fontweight='bold')
        # ax2.set_ylabel('PSD', fontweight='bold')
        # ax2.set_xlim([.2, 100.])
        # ax2.legend(loc='lower right')
        # # adjustments
        # # plt.get_current_fig_manager().full_screen_toggle()
        # # plt.show()
        # plt.savefig(f'{wdir}/{idat}/{evt}/{tr1.stats.network}.{stn}.{chn}.png', bbox_inches='tight', dpi='figure')
        # plt.close()
    ########################
    # EVENT SUMMARY FIGURE #
    _, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)
    ax.grid(which='both', axis='both')
    ax.plot(evst, fdom, 'o')
    ax.set_xlabel('Distance [km]', fontweight='bold', fontsize=15)
    ax.set_ylabel('Dominant frequency [Hz]', fontweight='bold', fontsize=15)
    ax.set_title(f'{evt} (M{emag:3.1f})')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()
    exit()

########################################################################################################################
# AUTOPICKER RESULTS SUMMARY
if_res = True
if if_res:
    # INITIALISE FIGURE
    XLim = [-10., 10.]
    ff, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, squeeze=True)
    ax1.grid(which='both', axis='both')
    ax1.set_xlim(XLim)
    ax2.grid(which='both', axis='both')
    ax2.set_xlim(XLim)
    ax3.grid(which='both', axis='both')
    ax3.set_xlim(XLim)
    ax4.grid(which='both', axis='both')
    ax4.set_xlim(XLim)
    # event list
    xevt = [datetime.strftime(e, '%Y%m%d%H%M%S%f')[:-3] for e in etab.OriginTime.to_list()]
    xmag = etab.Magnitude.to_list()
    # distance to center of local map
    xdis = [gdist.distance((mgrd[0] + (mgrd[1]-mgrd[0])/2., mgrd[2] + (mgrd[3]-mgrd[2])/2.),
                           (etab.Latitude[i], etab.Longitude[i])).km for i in range(len(etab))]
    # sort lists with decreasing magnitude
    xevt = [i for _, i in sorted(zip(xmag, xevt))][::-1]
    xdis = [i for _, i in sorted(zip(xmag, xdis))][::-1]
    xmag = np.sort(xmag)[::-1]
    # sort according to magnitude
    sorted_ind = np.argsort(xmag)[::-1]
    # loop over experiments
    bx = []
    for j in range(len(bwminf)):
        print(f'Experiment #{j}')
        # difference between auto and hand picks
        hdif = dict.fromkeys(xevt)
        # theo picks
        tdif = dict.fromkeys(xevt)
        # loop over events
        for evt in xevt:
            if pic != 'AIC':
                in_file = f"{wdir}/{idat}/{evt}_{j}.txt"
            else:
                in_file = f"{wdir}/{idat}/{evt}_{j}_AIC.txt"
            if path.exists(in_file):
                f_in = open(in_file)
            else:
                print(f" No output: {in_file.replace(wdir+'/'+idat+'/', '')}")
                continue
            lines = f_in.readlines()
            f_in.close()
            # header from 1st line
            s = lines[0]
            x = s.split()
            e = datetime.strftime(datetime.strptime(x[0] + ' ' + x[1], '%d/%m/%Y %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
            apar = [float(k) for k in x[6:]]
            # check input parameters
            if e != evt or apar[0] != rmhp[j] or apar[1] != taper[j] or apar[2] != bworder[j] or apar[3] != bwminf[j] or apar[4] != bwmaxf[j] \
                    or apar[5] != sta[j] or apar[6] != lta[j] or apar[7] != trigon[j] or apar[8] != trigoff[j]:
                print(f" Check input parameters for {evt}")
            else:
                n_pic = 0
                # read picks from other lines
                hand = []
                auto = []
                theo = []
                for line in lines[1:]:
                    if 'ENZ' in line:
                        hand.append(float(line.split('ENZ')[1].split()[1]))
                        auto.append(float(line.split('ENZ')[1].split()[2]))
                        theo.append(float(line.split('ENZ')[1].split()[3]))
                    elif 'HHZ' in line:
                        hand.append(float(line.split('HHZ')[1].split()[1]))
                        auto.append(float(line.split('HHZ')[1].split()[2]))
                        theo.append(float(line.split('HHZ')[1].split()[3]))
                    elif 'BHZ' in line:
                        hand.append(float(line.split('BHZ')[1].split()[1]))
                        auto.append(float(line.split('BHZ')[1].split()[2]))
                        theo.append(float(line.split('BHZ')[1].split()[3]))
                    n_pic += 1
                ix = [i for i, x in enumerate(hand) if not np.isnan(x)]
                hdif[evt] = [auto[idx]-hand[idx] for idx in ix]
                tdif[evt] = np.subtract(auto, theo)
        # axis selection based on experiment number
        ax = 0
        if j == 0:
            ax = ax1
        if j == 1:
            ax = ax2
        if j == 2:
            ax = ax3
        if j == 3:
            ax = ax4
        # choose which data to plot
        pdif = dict(hdif)
        ff.suptitle('Comparison with catalogue picks', fontweight='bold')
        # pdif = dict(tdif)
        # ff.suptitle('Comparison with theoretical picks', fontweight='bold')
        print(f" {len(pdif)} events to plot")
        # count number of data for each event
        nn = [len(tdif[e]) if tdif[e] is not None else 0 for e in xevt]
        # for empty events, replace None with 0
        for evt in xevt:
            if tdif[evt] is None:
                pdif[evt] = [0]
        # plot boxplots
        bx = ax.boxplot(pdif.values(), vert=False, showfliers=False)
        ie = 0
        evt_lbl = []
        for evt in xevt:
            # event label
            evt_lbl.append(f"{evt:16s} | M{xmag[ie]:3.1f} | {xdis[ie]:6.2f} km")
            # highlight distant events
            if j == 0 and xdis[ie] > 400.:
                ax.plot([ax.get_xlim()[0]-.5, ax.get_xlim()[0]-4.5], [ie+.7, ie+.7], 'r', clip_on=False)
            if tdif[evt] is None:
                # mark empty events with red cross
                ax.plot(0., ie+1, 'xr', markersize=10, linewidth=2)
            # display no. of data
            ax.text(ax.get_xlim()[1]+(ax.get_xlim()[1]-ax.get_xlim()[0])/30., ie+1, str(nn[ie]),
                    ha='left', va='center', color='orange', fontsize=8)
            ie += 1
        # labels
        ax.set_xlabel('Travel-time difference [s]', fontweight='bold')
        ax.set_yticklabels([])
        ax.set_title(f"BP: {bworder[j]} / {bwminf[j]:.2f} / {bwmaxf[j]:.2f} [Hz]\nSTA/LTA: {sta[j]:.2f} / {lta[j]:.2f} [s]")
        if ax == ax1:
            # replace numerical tick labels with event info
            ax.set_yticks(np.arange(1, ie+1, 1))
            ax.set_yticklabels(evt_lbl, fontsize=8)
        print()
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()
    exit()

# loop over events
for i in range(len(etab)):
    # read event parameters
    evt = datetime.strftime(datetime.strptime(str(etab.OriginTime[i]).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
    # if evt != '20220216032228454':
    #     continue
    epar = {'elat': etab.Latitude[i], 'elon': etab.Longitude[i], 'edep': etab.Depth[i],
            'eori': datetime.strptime(str(etab.OriginTime[i]).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), 'emag': etab.Magnitude[i]}
    print(f"{evt} M{epar['emag']:3.1f}")
    # check whether figure and output file already exist or not
    if pic == 'AIC':
        oxml = f"{evt}_{exp}_AIC.xml"
    else:
        oxml = f"{evt}_{exp}.xml"
    if path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}") != 0 and \
            path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}") != 0:
        print()
        continue
    # starting and ending times
    tbeg = datetime.strptime(str(etab.OriginTime[i]).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f') - timedelta(minutes=5)
    tend = datetime.strptime(str(etab.OriginTime[i]).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f') + timedelta(minutes=5)
    print(f" Time window: {tbeg.strftime('%d/%m/%Y %H:%M:%S.%f')} \u2013 {tend.strftime('%d/%m/%Y %H:%M:%S.%f')}")

    #######################################################################################################################
    # RETRIEVE RAW MINISEED DATA
    if path.exists(f"{wdir}/{idat}/{evt}.mseed") == 0:
        # import miniSEED file
        t1 = str(datetime.strftime(tbeg, '%Y-%m-%d %H:%M:%S'))
        t2 = str(datetime.strftime(tend, '%Y-%m-%d %H:%M:%S'))
        cmd1 = f'scart -dsE -n "{ntw}" -c "{chn}" -t "{t1}Z~{t2}Z" {adir} > {wdir}/v1.mseed'
        print(' ' + cmd1)
        os.system(cmd1)
        # repack miniSEED file
        cmd2 = f"msrepack -R 512 -i -a {wdir}/v1.mseed -o {wdir}/v2.mseed"
        print(' ' + cmd2)
        os.system(cmd2)
        # sort miniSEED file
        cmd3 = f"scmssort -uE {wdir}/v2.mseed > {wdir}/{idat}/{evt}.mseed"
        print(' ' + cmd3)
        os.system(cmd3)
        # clean up
        os.system(f"rm {wdir}/v?.mseed")
        # read resulting miniSEED file for waveform processing
        isn_traces = read(f"{wdir}/{idat}/{evt}.mseed").merge()
        # remove problematic channels
        for t in isn_traces.select(network='IS', station='EIL', channel='BHZ'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='IS', station='GEM', channel='BHZ'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='IS', station='KFSB', channel='HHZ', location='22'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='IS', station='HRFI', channel='HHZ', location=''):
            isn_traces.remove(t)
        # remove far away stations from streamer
        for t in isn_traces.select(network='GE', station='CSS'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='GE', station='ISP'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='GE', station='APE'):
            isn_traces.remove(t)
        for t in isn_traces.select(network='GE', station='ARPR'):
            isn_traces.remove(t)
        # remove Meiron stations with HHZ channel (not in inventory)
        for t in isn_traces.select(network='IS', channel='HHZ'):
            if 'MMA' in t.stats.station or 'MMB' in t.stats.station or 'MMC' in t.stats.station:
                isn_traces.remove(t)
        # remove response from all traces
        isn_traces.remove_response(output='VEL', inventory=isn_inv)
        # apply taper to all traces
        isn_traces.taper(max_percentage=.5, type='cosine', max_length=taper[exp], side='left')
        # apply high-pass filter to all traces
        isn_traces.filter('highpass', freq=1./rmhp[exp])
        # remove trend from all traces
        isn_traces.detrend('spline', order=3, dspline=500)
        # apply Butterworth band-pass filter to all traces
        isn_traces.filter('bandpass', freqmin=bwminf[exp], freqmax=bwmaxf[exp], corners=bworder[exp])
        # write miniSEED file
        isn_traces.write(f"{wdir}/{idat}/{evt}.mseed")
    else:
        print(' MiniSEED data for event %s already exists:' % evt)
        os.system(f"ls -lh {wdir}/{idat}/{evt}.mseed")
    # read miniSEED file
    isn_traces = read(f"{wdir}/{idat}/{evt}.mseed").merge()
    if len(isn_traces) == 0:
        print(f"No data found for {evt}")
        print()
        continue

    #######################################################################################################################
    # RUN AUTOPICKER
    # check the autopicker was run
    if path.exists(f"{wdir}/{idat}/{oxml}") != 0:
        print(f" Experiment #{exp} already ran for {evt}:")
        os.system(f"ls -lh {wdir}/{idat}/{oxml}")
    else:
        # autopicker command
        cmd = f"scautopick --ep --config-db {wdir}/config_autop.xml --inventory-db {wdir}/inventory_autop.xml" \
              f" --playback -I file://{wdir}/{idat}/{evt}.mseed > {wdir}/{idat}/{oxml}"
        print(f" Running experiment #{exp} for {evt}")
        print(' ' + cmd)
        os.system(cmd)
    # read output file
    if pic == 'AIC':
        atab = read_autopick_xml(f"{wdir}/{idat}/{oxml}", epar, 'P', 'AIC')
    else:
        atab = read_autopick_xml(f"{wdir}/{idat}/{oxml}", epar, 'P', '')
    # remove problematic channels
    atab = atab.drop(atab[(atab.net == 'IS') & (atab.chn == 'BHZ') & ((atab.sta == 'EIL') | (atab.sta == 'GEM'))].index).reset_index(drop=True)
    if len(atab) < 2:
        print(' Not enough automatic picks')
        print()
        continue
    else:
        print(f" {len(atab)} automatic picks")

    #######################################################################################################################
    # RETRIEVE HAND PICKS
    evt_lst = isn_client.get_events(eventid=etab.EventID[i], includearrivals=True)[0]
    n = 0
    htab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                         'chn': pd.Series(dtype='string'), 'pha': pd.Series(dtype='string'), 'pic': pd.Series(dtype='float64')})
    for pik in evt_lst.picks:
        if pik.phase_hint[0] == 'S':
            continue
        if pik.waveform_id.location_code is None:
            htab.loc[htab.shape[0]] = [pik.waveform_id.network_code, pik.waveform_id.station_code, '',
                                       pik.waveform_id.channel_code, pik.phase_hint, pik.time -
                                       evt_lst.preferred_origin().time]
        else:
            htab.loc[htab.shape[0]] = [pik.waveform_id.network_code, pik.waveform_id.station_code,
                                       pik.waveform_id.location_code, pik.waveform_id.channel_code,
                                       pik.phase_hint, pik.time - evt_lst.preferred_origin().time]
        n += 1
    print(f" {n} hand picks")

    #######################################################################################################################
    # DATA PRE-PROCESSING
    # array initialisation for common arrivals (hand/auto)
    rtab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                         'chn': pd.Series(dtype='string'), 'dis': pd.Series(dtype='float64'), 'harr': pd.Series(dtype='float64'),
                         'aarr': pd.Series(dtype='float64'), 'tarr': pd.Series(dtype='float64')})
    # traces to delete
    to_del = []
    nr = 0
    k = 0
    while k < len(isn_traces):
        # print(k, f"{isn_traces[k].stats.network}.{isn_traces[k].stats.station}."
        #          f"{isn_traces[k].stats.location}.{isn_traces[k].stats.channel}")
        # selection of best available channel
        lst = isn_traces.select(station=isn_traces[k].stats.station)
        ss = None
        if len(lst) == 1:
            ss = lst.select(station=isn_traces[k].stats.station)
        elif len(lst) > 1:
            # in case multiple channels available
            pp = None
            jp = None
            # in case pick exists
            if not atab[atab.sta == isn_traces[k].stats.station].empty:
                jj = atab.index[atab.sta == isn_traces[k].stats.station].to_list()
                if len(jj) > 1:
                    # in case several picks exist, test each acceptable option (ranked with decreasing priority)
                    for j in jj:
                        if pp is None and atab.net[j] == 'IS' and atab.chn[j] == 'HHZ':
                            pp = 1
                            jp = j
                        if pp is None and atab.net[j] == 'GE' and atab.chn[j] == 'HHZ':
                            pp = 1
                            jp = j
                        if pp is None and atab.net[j] == 'IS' and atab.chn[j] == 'BHZ':
                            pp = 1
                            jp = j
                        if pp is None and atab.net[j] == 'GE' and atab.chn[j] == 'BHZ':
                            pp = 1
                            jp = j
                        if pp is None and atab.net[j] == 'IS' and atab.chn[j] == 'ENZ':
                            pp = 1
                            jp = j
                        if pp is None and atab.net[j] == 'GE' and atab.chn[j] == 'ENZ':
                            pp = 1
                            jp = j
            if not ss or ss is None:
                if pp is not None:
                    ss = lst.select(station=atab.sta[jp], network=atab.net[jp], channel=atab.chn[jp])
                else:
                    # if no pick, and in case several available channels, test each acceptable option (ranked with decreasing priority)
                    s1 = lst.select(station=isn_traces[k].stats.station, network='IS', channel='HHZ')
                    if s1:
                        ss = s1
                    s2 = lst.select(station=isn_traces[k].stats.station, network='GE', channel='HHZ')
                    if not s1 and s2:
                        ss = s2
                    s3 = lst.select(station=isn_traces[k].stats.station, network='IS', channel='BHZ')
                    if not s1 and not s2 and s3:
                        ss = s3
                    s4 = lst.select(station=isn_traces[k].stats.station, network='GE', channel='BHZ')
                    if not s1 and not s2 and not s3 and s4:
                        ss = s4
                    s5 = lst.select(station=isn_traces[k].stats.station, network='IS', channel='ENZ')
                    if not s1 and not s2 and not s3 and not s4 and s5:
                        ss = s5
                    s6 = lst.select(station=isn_traces[k].stats.station, network='GE', channel='ENZ')
                    if not s1 and not s2 and not s3 and not s4 and not s5 and s6:
                        ss = s6
            # remove unselected channels (only getting here if >1 channels)
            for item in lst:
                if item != ss[0]:
                    # isn_traces.remove(item)
                    to_del.append(item)
        elif len(lst) == 0:
            # print(' Not in stream')
            k += 1
            continue
        # find station in inventory
        stn = isn_inv.select(network=isn_traces[k].stats.network, station=isn_traces[k].stats.station,
                             channel=isn_traces[k].stats.channel, location=isn_traces[k].stats.location)
        if len(stn) == 0:
            # print(' Not in inventory')
            # isn_traces.remove(isn_traces[k])
            to_del.append(isn_traces[k])
            k += 1
            continue
        elif len(stn) > 1:
            print(f" Multiple matches in station inventory: {isn_traces[k].stats.station}")
            exit()
        # calculate event-station distance
        d = gdist.distance((epar['elat'], epar['elon']), (stn[0].stations[0].channels[0].latitude, stn[0].stations[0].channels[0].longitude))
        isn_traces[k].stats.distance = d.m
        # compute theoretical travel times
        x = model.get_travel_times(source_depth_in_km=epar['edep'], distance_in_degree=d.km / (2 * np.pi * rrad / 360),
                                   phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        if len(x) != 0:
            isn_traces[k].stats['theo_tt'] = x[0].time
        else:
            # print(' No theoretical arrival')
            isn_traces[k].stats['theo_tt'] = np.nan
        # selection of picks for statistics and output file
        kh = None
        # hand picks
        if n > 0:
            if not htab[(htab.sta == isn_traces[k].stats.station)].empty:
                kh = htab.index[(htab.sta == isn_traces[k].stats.station) &
                                (htab.net == isn_traces[k].stats.network) & (htab.chn == isn_traces[k].stats.channel)].to_list()
                if len(kh) == 1:
                    kh = kh[0]
                elif len(kh) > 1:
                    print(' Multiple results in hand picks')
                    exit()
                else:
                    # print(' No result in hand picks')
                    k += 1
                    continue
        ka = None
        # automatic picks
        if not atab.empty:
            if not atab[(atab.sta == isn_traces[k].stats.station)].empty:
                # indexing
                tab = atab[(atab.sta == isn_traces[k].stats.station) &
                           (atab.net == isn_traces[k].stats.network) & (atab.chn == isn_traces[k].stats.channel)]
                tab = tab.assign(tdif=pd.Series([None] * len(tab), dtype='float'))
                if tab.empty:
                    # print(' No result in auto picks')
                    k += 1
                    continue
                if len(tab) > 1:
                    ka = None
                    # find pick closest to theoretical arrival
                    tab['tdif'] = [abs(xx - x[0].time) for xx in tab.pic.to_list()]
                    for j in tab.index:
                        if tab.pic[j] > 0 and j == tab['tdif'].idxmin():
                            ka = j
                            break
                else:
                    ka = tab.index[0]
        # table for residuals (for statistics)
        if kh is not None and ka is not None:
            rtab.loc[rtab.shape[0]] = [isn_traces[k].stats.network, isn_traces[k].stats.station, isn_traces[k].stats.location,
                                       isn_traces[k].stats.channel, isn_traces[k].stats.distance/1000.,
                                       htab.pic[kh], atab.pic[ka], isn_traces[k].stats.theo_tt]
            nr += 1
        elif kh is None and ka is not None:
            rtab.loc[rtab.shape[0]] = [isn_traces[k].stats.network, isn_traces[k].stats.station, isn_traces[k].stats.location,
                                       isn_traces[k].stats.channel, isn_traces[k].stats.distance/1000.,
                                       np.nan, atab.pic[ka], isn_traces[k].stats.theo_tt]
        k += 1
    # delete selected waveforms
    for tr in to_del:
        try:
            isn_traces.remove(tr)
        except:
            continue
    #######################################################################################################################
    # BUILD OUTPUT FILE
    if path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}") == 0:
        # write to file
        fout = open(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}", 'w')
        ot = str(datetime.strftime(etab.OriginTime[i], '%d/%m/%Y %H:%M:%S.%f'))
        # write to file
        if nr > 1:
            fout.write(f"{ot} {etab.Latitude[i]:9.4f} {etab.Longitude[i]:9.4f} {etab.Depth[i]:5.1f} {etab.Magnitude[i]:4.2f} "
                       f"{rmhp[exp]:.1f} {taper[exp]:4.1f} {bworder[exp]} {bwminf[exp]:4.1f} {bwmaxf[exp]:4.1f} "
                       f"{sta[exp]:4.1f} {lta[exp]:4.1f} {trigon[exp]:4.1f} {trigoff[exp]:4.1f} {nr:3i} "
                       f"{(rtab.aarr-rtab.harr).mean():9.4f} {(rtab.aarr-rtab.harr).std():9.4f} "
                       f"{(rtab.aarr-rtab.tarr).mean():9.4f} {(rtab.aarr-rtab.tarr).std():9.4f}\n")
        elif nr == 1:
            fout.write(f"{ot} {etab.Latitude[i]:9.4f} {etab.Longitude[i]:9.4f} {etab.Depth[i]:5.1f} {etab.Magnitude[i]:4.2f} "
                       f"{rmhp[exp]:.1f} {taper[exp]:4.1f} {bworder[exp]} {bwminf[exp]:4.1f} {bwmaxf[exp]:4.1f} "
                       f"{sta[exp]:4.1f} {lta[exp]:4.1f} {trigon[exp]:4.1f} {trigoff[exp]:4.1f} {nr:3i} "
                       f"{(rtab.aarr-rtab.harr).mean():9.4f} {np.nan:9.4f} "
                       f"{(rtab.aarr-rtab.tarr).mean():9.4f} {(rtab.aarr-rtab.tarr).std():9.4f}\n")
        elif nr == 0:
            fout.write(f"{ot} {etab.Latitude[i]:9.4f} {etab.Longitude[i]:9.4f} {etab.Depth[i]:5.1f} {etab.Magnitude[i]:4.2f} "
                       f"{rmhp[exp]:.1f} {taper[exp]:4.1f} {bworder[exp]} {bwminf[exp]:4.1f} {bwmaxf[exp]:4.1f} "
                       f"{sta[exp]:4.1f} {lta[exp]:4.1f} {trigon[exp]:4.1f} {trigoff[exp]:4.1f} {nr:3i} "
                       f"{np.nan:9.4f} {np.nan:9.4f} "
                       f"{(rtab.aarr-rtab.tarr).mean():9.4f} {(rtab.aarr-rtab.tarr).std():9.4f}\n")
        for k in range(len(rtab)):
            fout.write(f"{rtab.loc[k, 'net']:-2s} {rtab.loc[k, 'sta']:-6s} {rtab.loc[k, 'loc']:-2s} {rtab.loc[k, 'chn']:-3s} "
                       f"{rtab.loc[k, 'dis']:9.4f} {rtab.loc[k, 'harr']:9.4f} {rtab.loc[k, 'aarr']:9.4f} {rtab.loc[k, 'tarr']:9.4f}\n")
        fout.close()
    #######################################################################################################################
    # BUILD OUTPUT FIGURE
    # only if figure does not already exist
    if path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}") == 0:
        # sort according to distance (descending)
        tr_dist = [tr.stats.distance for tr in isn_traces]
        sorted_ind = np.argsort(tr_dist)[::-1]
        print(f" {len(isn_traces)} waveforms")
        plot_autopick_sec(isn_traces, atab, htab, epar, fpar, sorted_ind, f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}")
    print()

########################################################################################################################
# ifyes = False
# if ifyes:
#     # select station
#     stn = 'MRON'
#     S = isn_traces.select(station=stn)[0]
#     # figure
#     fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, squeeze=True)
#     ax1.grid(which='both', axis='both')
#     ax2.grid(which='both', axis='both')
#     # plot velocity waveform
#     T = pd.date_range(datetime.strptime(str(S.stats.starttime), '%Y-%m-%dT%H:%M:%S.%fZ'),
#                       datetime.strptime(str(S.stats.endtime), '%Y-%m-%dT%H:%M:%S.%fZ'), freq='%fL' % S.stats.delta*1000).to_list()
#     h11, = ax1.plot(T, S.data, color='red', alpha=.7, label='Velocity')
#     # plot STA/LTA waveform
#     stalta = sta_lta(S.data, sta[exp] * S.stats.sampling_rate, lta[exp] * S.stats.sampling_rate)
#     h21, = ax2.plot(T, stalta, color='blue', alpha=.7, label='STA/LTA')
#     # plot thresholds
#     h22, = ax2.plot(ax2.get_xlim(), [trigon[exp], trigon[exp]], color='pink', label='TrigOn')
#     h23, = ax2.plot(ax2.get_xlim(), [trigoff[exp], trigoff[exp]], color='orange', label='TrigOff')
#     # picks from autopicker
#     ii = atab.index[(atab.sta == S.stats.station) & (atab.net == S.stats.network) & (atab.chn == S.stats.channel)].to_list()
#     h12 = []
#     h24 = []
#     for i in range(len(ii)):
#         h12, = ax1.plot([atab.pic[ii[i]], atab.pic[ii[i]]], [-np.amax(S.data), np.amax(S.data)], color='purple', label='Autopick')
#         h24, = ax2.plot([atab.pic[ii[i]], atab.pic[ii[i]]], [0., np.amax(stalta)], color='purple', label='Autopick')
#     # index of max. STA/LTA value
#     ax2.text(T[np.argmax(stalta)], np.amax(stalta), '%.1f' % np.amax(stalta), fontsize=8, ha='right', va='center', color='red')
#     # titles, labels & legends
#     ax2.set_title('%s.%s.%s.%s' % (S.stats.network, S.stats.station, S.stats.location, S.stats.channel))
#     ax1.set_xlim([0., 100.])
#     ax2.set_xlim([0., 100.])
#     ax1.set_ylabel('Velocity', fontsize=15)
#     ax2.set_ylabel('STA/LTA', fontsize=15)
#     ax2.set_xlabel('Time - OT [s]', fontsize=15)
#     ax1.legend(handles=[h11, h12], loc='upper left')
#     ax2.legend(handles=[h21, h22, h23, h24], loc='upper left')
#     # show autopicker parameters
#     microsecond = epar['eori'].microsecond
#     millisecond = int(round(microsecond / 1000.))
#     plt.gcf().text(.5, .98, '%s \u2013 %.2f km \u2013 M%3.1f' % (
#         epar['eori'].strftime('%d/%m/%Y %H:%M:%S.%f').replace('.{:06d}'.format(microsecond), '.{:03d}'.format(millisecond)),
#         epar['edep'], epar['emag']), fontsize=15, fontweight='bold', ha='center', va='center')
#     plt.gcf().text(.5, .95, 'HP: %.2f [s] \u2013 Taper: %.2f [s] \u2013 BP: %i / %.2f / %.2f [Hz]'.
#                    format(fpar['rmhp'], fpar['taper'], fpar['bworder'], fpar['bwminf'], fpar['bwmaxf']),
#                    fontsize=15, fontweight='bold', ha='center', va='center')
#     plt.gcf().text(.5, .92, 'STA/LTA: %.2f / %.2f [s] \u2013 Trigger: %.2f / %.2f'.
#                    format(fpar['sta'], fpar['lta'], fpar['trigon'], fpar['trigoff']), fontsize=15, fontweight='bold', ha='center', va='center')
#     plt.show()
#     exit()


# # write to file simultaneously to making figure
# fout = open('%s/%s_%i.txt' % (wdir, evt, exp), 'w')
# ifmap = True
# ########################################################################################################################
# # FIGURE
# XLim1 = [-1., max(ttab.theo_tt) + 80.]
# XLim2 = [-7., 50.]
# XLim3 = [-2., 2.]
# if ifmap:
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, squeeze=True, gridspec_kw={'width_ratios': [1, 2, 1, 1]})
# else:
#     ax3 = []
#     ax4 = []
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True)
# plt.show(block=False)
# # AXIS 1: VELOCITY WAVEFORMS
# ax1.grid(which='both', axis='both')
# ax1.set_xlim(XLim1)
# # AXIS 2: VELOCITY STA/LTA's
# ax2.grid(which='both', axis='both')
# ax2.set_xlim(XLim2)
# if ifmap:
#     # AXIS 3: RESIDUAL OVER DISTANCE PLOT
#     ax3.grid(which='both', axis='both')
#     ax3.set_xlim(XLim3)
#     # AXIS 4: LOCAL STATIONS & EVENT MAP
#     M = Basemap(projection='cyl', llcrnrlon=igrd[2], llcrnrlat=igrd[0], urcrnrlon=igrd[3], urcrnrlat=igrd[1], ax=ax4)
#     # draw map
#     M.drawmapboundary(fill_color='none')
#     M.fillcontinents(color='0.8', lake_color='white')
#     # show parallels and meridians (labels = [left,right,top,bottom])
#     M.drawparallels(np.arange(20., 40., 1.), labels=[False, True, True, False])
#     M.drawmeridians(np.arange(20., 40., 1.), labels=[True, False, False, True])
# else:
#     M = []
# n = 0
# n1 = 0
# n2 = 0
# nn = 0
# k1 = None
# k2 = None
# lbl = []
# tres = []
# h1 = []
# h2 = []
# h3 = []
# h4 = []
# h5 = []
# h6 = []
# h7 = []
# if ifmap:
#     h8 = []
#     h9 = []
# for i in range(len(isn_traces)):
#     n += 1
#     # sorted indexes
#     ii = sorted_ind[i]
#     # station information
#     lbl.append('%s.%s.%s.%s' % (isn_traces[ii].stats.network, isn_traces[ii].stats.station,
#                                 isn_traces[ii].stats.location, isn_traces[ii].stats.channel))
#     # plot waveform
#     T = isn_traces[ii].times('relative', reftime=UTCDateTime(eori))
#     h1, = ax1.plot(T, isn_traces[ii].data / np.amax(isn_traces[ii].data) + n, color='grey', alpha=.7, label='Velocity')
#     # plot STA/LTA waveform
#     stalta = sta_lta(isn_traces[ii].data, sta[exp] * isn_traces[ii].stats.sampling_rate, lta[exp] * isn_traces[ii].stats.sampling_rate)
#     # h2, = ax2.plot(T - np.timedelta64(int(np.floor(ttheo[ii])), 's') - np.timedelta64(int((ttheo[ii]-np.floor(ttheo[ii]))*1e6), '[ns]'),
#     #                stalta / np.amax(stalta) + n, color='grey', alpha=.7, label='STA/LTA')
#     h2, = ax2.plot(T - ttheo[ii], stalta / np.amax(stalta) + n, color='grey', alpha=.7, label='STA/LTA')
#     # # index of max. STA/LTA value
#     # ax2.text(ax2.get_xlim()[1], n, '{:.1f}' % (np.amax(stalta)), fontsize=8, ha='left', va='center', color='red', clip_on=False)
#     # ax2.plot([T[np.argmax(stalta)] - ttheo[ii], T[np.argmax(stalta)] - ttheo[ii]], [n-1, n+1], color='red')
#     # theoretical travel time
#     if ttheo[ii] != '':
#         h3, = ax1.plot([ttheo[ii], ttheo[ii]], [n - 1, n + 1], color='blue', linestyle='dotted', label='iasp91')
#     # show hand picks
#     k1 = None
#     if not htab[(htab.sta == isn_traces[ii].stats.station)].empty:
#         k1 = htab.index[(htab.sta == isn_traces[ii].stats.station) & (htab.net == isn_traces[ii].stats.network)
#                         & (htab.chn == isn_traces[ii].stats.channel)].to_list()
#         if len(k1) == 1:
#             k1 = k1[0]
#             # print('One hand pick: %s.%s.%s.%s (%.2f s)' % (htab.net[k1], htab.sta[k1], htab.lct[k1], htab.chn[k1], htab.pic[k1]))
#         else:
#             # print('No hand pick: %s' % (stream[ii].stats.station))
#             continue
#         if k1 is not None:
#             n1 += 1
#             # show on both axes
#             h4, = ax1.plot([htab.pic[k1], htab.pic[k1]], [n - 1, n + 1], color='orange', label='Handpick')
#             h5, = ax2.plot([htab.pic[k1] - ttheo[ii], htab.pic[k1] - ttheo[ii]], [n - 1, n + 1], color='orange', label='Handpick')
#             if ifmap:
#                 # station name (MAP)
#                 for j in range(len(isn_inv.networks[0])):
#                     if isn_traces[ii].stats.station == isn_inv.networks[0].stations[j].code:
#                         x, y = M(isn_inv.networks[0].stations[j].longitude, isn_inv.networks[0].stations[j].latitude)
#                         ax4.text(x, y - .05, isn_traces[ii].stats.station, ha='center', va='center', fontsize=8, color='orange', label='Autopick')
#             # pick time
#             t1 = ax2.text(htab.pic[k1] - 2 - ttheo[ii], n + .25, '%.2f s' % (htab.pic[k1]), ha='right', va='center', color='orange', fontsize=8)
#             t1.set_bbox(dict(facecolor='white', alpha=.6, edgecolor='orange'))
#     # show automatic picks
#     k2 = None
#     if not atab[(atab.sta == isn_traces[ii].stats.station)].empty:
#         # indexing
#         P = atab[(atab.sta == isn_traces[ii].stats.station) & (atab.net == isn_traces[ii].stats.network) & (atab.chn == isn_traces[ii].stats.channel)]
#         P = P.assign(tdif=pd.Series([None] * len(P), dtype='float'))
#         if P.empty:
#             # print('No automatic pick: {:s}'.format(stream[ii].stats.station))
#             continue
#         if len(P) > 1:
#             k2 = None
#             # print('Multiple automatic picks: ({:n})' % (len(kk)))
#             P['tdif'] = [abs(x - ttheo[ii]) for x in P.pic.to_list()]
#             for k in P.index:
#                 # print(' %s.%s.%s.%s (%.2f s)' % (atab.net[kk[k]], atab.sta[kk[k]], atab.lct[kk[k]], atab.chn[kk[k]], atab.pic[kk[k]]))
#                 if P.pic[k] > 0 and k == P['tdif'].idxmin():
#                     k2 = k
#                     break
#         else:
#             k2 = P.index[0]
#             # print('One automatic pick: %s.%s.%s.%s (%.2f s)' % (atab.net[k2], atab.sta[k2], atab.lct[k2], atab.chn[k2], atab.pic[k2]))
#         n2 += 1
#         for k in P.index:
#             # show on both axes
#             h6, = ax1.plot([P.pic[k], P.pic[k]], [n - 1, n + 1], color='purple', label='Autopick')
#             ax1.text(ax1.get_xlim()[1], n, P.sta[k], color='purple', fontsize=6, ha='left', va='center')
#             h7, = ax2.plot([P.pic[k] - ttheo[ii], P.pic[k] - ttheo[ii]], [n - 1, n + 1], color='purple', label='Autopick')
#             if ifmap:
#                 # station name (MAP)
#                 for j in range(len(isn_inv.networks[0])):
#                     if isn_traces[ii].stats.station == isn_inv.networks[0].stations[j].code:
#                         x, y = M(isn_inv.networks[0].stations[j].longitude, isn_inv.networks[0].stations[j].latitude)
#                         ax4.text(x, y + .05, isn_traces[ii].stats.station, ha='center', va='center', fontsize=8, color='purple', label='Autopick')
#             if XLim2[0] <= P.pic[k] - ttheo[ii] <= XLim2[1]:
#                 # pick time
#                 t2 = ax2.text(P.pic[k] + 2 - ttheo[ii], n + .25, '%.2f s' % (P.pic[k]), ha='center', va='center', color='purple', fontsize=8)
#                 t2.set_bbox(dict(facecolor='white', alpha=.6, edgecolor='purple'))
#                 # pick SNR
#                 t3 = ax2.text(25., n, '%.2f' % (P.snr[k]), ha='center', va='center', fontsize=8)
#                 t3.set_bbox(dict(facecolor='white', alpha=.6))
#     # residual w.r.t. hand pick
#     if k1 is not None and k2 is not None:
#         # table for residuals (for statistics)
#         tres.append(atab.pic[k2] - htab.pic[k1])
#         # show value
#         t4 = ax2.text(35., n, '%.2f s' % (atab.pic[k2] - htab.pic[k1]), ha='center', va='center', fontsize=8)
#         t4.set_bbox(dict(facecolor='white', alpha=.6))
#         if ifmap:
#             # residual plot
#             ax3.plot(atab.pic[k2] - htab.pic[k1], n, 'o', markersize=7, mfc='grey', mec='none', alpha=.7)
#             # station names
#             ax3.text(atab.pic[k2] - htab.pic[k1] - .1, n, isn_traces[ii].stats.station, ha='right', va='center', fontsize=8)
#         nn += 1
# # statistics on residuals
# if ifmap and nn > 1:
#     ax3.text(.95 * XLim3[0], 1, '<\u0394t> = %.2f \u00B1 %.2f [s]' % (statistics.mean(tres), statistics.stdev(tres)), fontsize=10, fontweight='bold')
# # write to file
# if nn > 1:
#     fout.write('%s %f %f %i %f %f %f %f %f %f %i %f %f\n' % (evt, rmhp[exp], taper[exp], bworder[exp], bwminf[exp], bwmaxf[exp], sta[exp], lta[exp],
#                                                              trigon[exp], trigoff[exp], nn, statistics.mean(tres), statistics.stdev(tres)))
# elif nn == 1:
#     fout.write('%s %f %f %i %f %f %f %f %f %f %i %f %s\n' % (evt, rmhp[exp], taper[exp], bworder[exp], bwminf[exp], bwmaxf[exp], sta[exp], lta[exp],
#                                                              trigon[exp], trigoff[exp], nn, statistics.mean(tres), 'NaN'))
# elif nn == 0:
#     fout.write('%s %f %f %i %f %f %f %f %f %f %i %s %s\n' % (evt, rmhp[exp], taper[exp], bworder[exp], bwminf[exp], bwmaxf[exp], sta[exp], lta[exp],
#                                                              trigon[exp], trigoff[exp], nn, 'NaN', 'NaN'))
# fout.close()
# # axis limits
# ax1.set_ylim([0, n + 1])
# ax2.set_ylim([0, n + 1])
# if ifmap:
#     ax3.set_ylim([0, n + 1])
# # legend
# if not h6 and not h7:
#     ax1.legend(handles=[h1, h3, h4], loc='upper right')
#     ax2.legend(handles=[h2, h5], loc='lower right')
# else:
#     ax1.legend(handles=[h1, h3, h4, h6], loc='upper right')
#     ax2.legend(handles=[h2, h5, h7], loc='lower right')
# ax2.text(25., 1.01 * ax1.get_ylim()[1], 'S/N', ha='center', va='center', fontsize=10)
# ax2.text(35., 1.01 * ax1.get_ylim()[1], '\u0394t (N=%i)' % nn, ha='center', va='center', fontsize=10)
# # station and pick numbers
# ax1.text(XLim1[0] - .025 * XLim1[1], 1.01 * ax1.get_ylim()[1], 'N=%i' % i, ha='right', va='center', fontsize=10)
# ax2.text(5., 1.01 * ax1.get_ylim()[1], 'N=%i' % n2, ha='center', va='center', fontsize=10, color='purple')
# ax2.text(-5., 1.01 * ax1.get_ylim()[1], 'N=%i' % n1, ha='center', va='center', fontsize=10, color='orange')
# # replace numerical tick labels with station names
# ax1.set_yticks(np.arange(1, n + 1, 1))
# ax1.set_yticklabels(lbl, fontsize=6)
# ax2.set_yticks(np.arange(1, n + 1, 1))
# ax2.set_yticklabels([])
# if ifmap:
#     ax3.set_yticklabels([])
# # axis labels
# ax1.set_xlabel('Time - OT [s]', fontsize=20, fontweight='bold')
# ax1.set_ylabel('Station', fontsize=20, fontweight='bold')
# ax2.set_xlabel('Time - Tp(iasp91) [s]', fontsize=20, fontweight='bold')
# if ifmap:
#     ax3.set_xlabel('\u0394t [s]', fontsize=20, fontweight='bold')
#     # MAP
#     # fault lines
#     f = open('%s/../Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz' % wdir, 'r')
#     flts = f.readlines()
#     f.close()
#     X = []
#     Y = []
#     for i in range(len(flts)):  # len(flts)
#         if re.search('NaN', flts[i]):
#             X = []
#             Y = []
#         elif i < len(flts) - 1 and re.search('NaN', flts[i + 1]):
#             f, = ax4.plot(X, Y, '.6', label='Faults')
#         else:
#             line = flts[i].split()
#             x, y = M(float(line[0]), float(line[1]))
#             X.append(x)
#             Y.append(y)
#     # stations
#     hm1 = []
#     hm2 = []
#     for i in range(len(isn_traces)):
#         # sorted index
#         ii = sorted_ind[i]
#         for j in range(len(isn_inv.networks[0])):
#             if isn_traces[ii].stats.station == isn_inv.networks[0].stations[j].code:
#                 x, y = M(isn_inv.networks[0].stations[j].longitude, isn_inv.networks[0].stations[j].latitude)
#                 if isn_traces[ii].stats.network == 'IS':
#                     hm1, = ax4.plot(x, y, 'b^', markersize=5, alpha=.7, mec='none', label=isn_traces[ii].stats.network)
#                 elif isn_traces[ii].stats.network == 'GE':
#                     hm2, = ax4.plot(x, y, 'sg', markersize=7, mfc='none', label=isn_traces[ii].stats.network)
#     # event
#     x, y = M(elon, elat)
#     e, = ax4.plot(x, y, 'r*', markersize=15, markeredgecolor='black', label='Event')
#     if emag:
#         ax4.text(x + .1, y - .1, 'M%3.1f' % emag, ha='left', va='bottom', color='red', fontweight='bold')
#     # legend
#     if not hm1:
#         hm = [hm2, e, f]
#     elif not hm2:
#         hm = [hm1, e, f]
#     else:
#         hm = [hm1, hm2, e, f]
#     ax4.legend(handles=hm, loc='upper left')
# # show autopicker parameters
# microsecond = eori.microsecond
# millisecond = int(round(microsecond / 1000.))
# plt.gcf().text(.5, .98, '%s \u2013 %.2f km' % (
#     eori.strftime('%d/%m/%Y %H:%M:%S.%f').replace('.{:06d}'.format(microsecond), '.{:03d}'.format(millisecond)),
#     edep), fontsize=15, fontweight='bold', ha='center', va='center')
# plt.gcf().text(.5, .95, 'HP: %.2f [s] \u2013 Taper: %.2f [s] \u2013 BP: %i / %.2f / %.2f [Hz]'
#                % (rmhp[exp], taper[exp], bworder[exp], bwminf[exp], bwmaxf[exp]), fontsize=15, fontweight='bold', ha='center', va='center')
# plt.gcf().text(.5, .92, 'STA/LTA: %.2f / %.2f [s] \u2013 Trigger: %.2f / %.2f'
#                % (sta[exp], lta[exp], trigon[exp], trigoff[exp]), fontsize=15, fontweight='bold', ha='center', va='center')
# # maximise figure
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# # adjust plots
# fig.subplots_adjust(left=.07, right=.98, wspace=.1)
# # show figure
# plt.show()

######################################################################################################################
# AUTOPICKER CONFIGURATION
# # autopicker picking configuration
# # if pic == 'AIC':
# #     os.system('mv $SEISCOMP_ROOT/etc/scautopick.aic $SEISCOMP_ROOT/etc/scautopick.cfg')
# # else:
# #     os.system('rm $SEISCOMP_ROOT/etc/scautopick.cfg')
# # autopicker detection configuration
# if exp == 0:
#     cfg = 'config.xml'
#     # check if default autopicker configuration file exists
#     if path.exists('%s/%s' % (wdir, cfg)) != 0 or path.exists('%s/%s' % (wdir, cfg)) == 0:
#         # if path.exists('%s/%s' % (wdir, cfg)) != 0:
#         #     print('Default autopicker configuration file already exists:')
#         #     os.system('ls -lh %s/%s' % (wdir, cfg))
#         #     print()
#         # else:
#         cmd = 'scxmldump -fC -o %s/Autopicker/%s -d postgresql://' % (wdir, cfg)
#         print(cmd)
#         print()
# else:
#     cfg = 'config_%s_%i.xml' % (evt, exp)
#     if path.exists('%s/%s' % (wdir, cfg)) != 0 or path.exists('%s/%s' % (wdir, cfg)) == 0:
#         # if path.exists('%s/%s' % (wdir, cfg)) != 0:
#         #     print('Configuration file for event %s and experiment #{:n} already exists:' % (evt, exp))
#         #     os.system('ls -lh %s/%s' % (wdir, cfg))
#         #     print()
#         # else:
#         # read default autopicker configuration file
#         ETree.register_namespace('', "http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12")
#         tree = ETree.parse('%s/config.bak' % wdir)
#         root = tree.getroot()
#         # loop over stations
#         n = 0
#         for i in range(len(root[0])):
#             if re.search('scautopick', str(root[0][i].attrib)):
#                 s = re.search("'ParameterSet/trunk/Station/(.*?)/scautopick',", str(root[0][i].attrib)).group(1)
#                 # loop over station parameters to modify
#                 for j in range(len(root[0][i])):
#                     if len(root[0][i][j]) != 0:
#                         if root[0][i][j][0].text == 'trigOn':
#                             # threshold/trigger on
#                             root[0][i][j][1].text = '%g' % fpar['trigon']
#                         if root[0][i][j][0].text == 'trigOff':
#                             # threshold/trigger off
#                             root[0][i][j][1].text = '%g' % fpar['trigoff']
#                         if root[0][i][j][0].text == 'detecFilter':
#                             # list of filters
#                             x = root[0][i][j][1].text.split('>>')
#                             # running-mean high-pass filter
#                             s = re.sub(r'\((.*?)\)', r'(%g)' % fpar['rmhp'], x[0])
#                             root[0][i][j][1].text = root[0][i][j][1].text.replace(x[0], s)
#                             # taper
#                             s = re.sub(r'\((.*?)\)', r'(%g)' % fpar['taper'], x[1])
#                             root[0][i][j][1].text = root[0][i][j][1].text.replace(x[1], s)
#                             # BW band-pass filter
#                             s = re.sub(r'\((.*?)\)', r'(%i,%g,%g)' % (fpar['bworder'], fpar['bwminf'], fpar['bwmaxf']),
#                                        x[2])
#                             root[0][i][j][1].text = root[0][i][j][1].text.replace(x[2], s)
#                             # STA/LTA filter
#                             s = re.sub(r'\((.*?)\)', r'(%g,%g)' % (fpar['sta'], fpar['lta']), x[3])
#                             root[0][i][j][1].text = root[0][i][j][1].text.replace(x[3], s)
#                 n += 1
#         # write new configuration file
#         tree.write('%s/%s' % (wdir, cfg), xml_declaration=True, method="xml", encoding="UTF-8")
#         print('Configuration file for experiment #%i written:' % exp)
#         os.system('ls -lh %s/%s' % (wdir, cfg))
#         print()
