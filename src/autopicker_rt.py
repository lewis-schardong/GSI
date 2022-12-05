########################################################################################################################
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
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import xml.etree.ElementTree as ETree
from obspy.core import UTCDateTime
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def read_autopick_xml(xml_path, phase='P', picker=''):
    """
    :param xml_path: path to .XML file containing picks to read
    :param phase: seismic phase of picks to retrieve; defaults to "P"
    :param picker: picker used
    :return: DataFrame containing automatic picks
    """
    # initialise output DataFrame
    pick_tab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'),
                             'loc': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string'),
                             'pick': pd.Series(dtype='datetime64[ms]')})
    # read .xml file from scautopick
    for line_xml in ETree.parse(xml_path).getroot()[0]:
        # search for pick entry
        if re.search('pick', str(line_xml.tag)):
            # check picker is requested one
            if picker and not re.search(picker, str(line_xml.attrib)):
                continue
            # check phase is requested one
            if (line_xml[3].tag == 'phaseHint' and line_xml[3].text != phase) or \
                    (line_xml[4].tag == 'phaseHint' and line_xml[4].text != phase):
                continue
            # find waveform metadata
            str1 = re.search("'publicID': '(.*?)'", str(line_xml.attrib)).group(1)
            str2 = str1.split('.')
            str3 = str2[2].split('-')
            # network code
            if not picker:
                pik_net = str3[1]
            else:
                pik_net = str3[2]
            # location code
            if str2[4] is None:
                pik_loc = ''
            else:
                pik_loc = str2[4]
            # fill DataFrame with channel metadata and picking time
            pick_tab.loc[pick_tab.shape[0]] = [pik_net, str2[3], pik_loc, str2[5],
                                               datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ')]
    return pick_tab


def get_traces_deci(evt_id, ori_time, time_win, networks='IS',
                    channels='(B|H|E)(H|N)Z', db_dir='/net/172.16.46.200/archive/jqdata/archive'):
    """
    :param evt_id: event ID
    :param ori_time: event origin time
    :param time_win: half-length of time window
    :param networks: seismic networks for which to request waveform data; defaults to "IS"
    :param channels: seismic channels for which to request waveform data; defaults to "(B|H|E)(H|N)Z"
    :param db_dir: path to database directory from which to request waveform data; defaults to decimated data
    on /net/172.16.46.200/archive/jqdata/archive
    :return: full path of created miniSEED file
    """
    # define output .mseed file name & path
    mseed = f'{wdir}/{evt_id}.mseed'
    if path.exists(mseed) == 0:
        # define start and end times for data retrieval
        tbeg = str(datetime.strftime(ori_time - time_win, '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(ori_time + time_win, '%Y-%m-%d %H:%M:%S'))
        # retrieve data using SeisComP's 'scart' command
        os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{tbeg}~{tend}" {db_dir} > {mseed}')
        # delete file if empty
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            return ''
        # read created miniSEED file
        stream = read(mseed)
        # ensuring 50-Hz sampling rate for all traces
        stream.resample(50.0, window='hann')
        # merging all traces (e.g. if time window crosses midnight, data from both dates need to be merged)
        stream.merge(fill_value='interpolate')
        # write miniSEED file
        stream.write(mseed)
    else:
        # delete file if existing but empty
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            return ''
    return mseed


def get_catalogue_picks(client, event_id, phase='P'):
    """
    :param client: Obspy FDSN client to retrieve data from
    :param event_id: event ID
    :param phase: seismic phase of picks to retrieve; defaults to "P"
    :return: DataFrame containing catalogue picks
    """
    # initialise output DataFrame
    cat_tab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'),
                            'loc': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string'),
                            'pick': pd.Series(dtype='datetime64[ms]')})
    # retrieve event info.
    event = client.get_events(eventid=event_id, includearrivals=True)[0]
    # loop over picks
    for pick in event.picks:
        # retrieve
        if pick.phase_hint[0] == phase:
            # empty variable if location code is empty
            if pick.waveform_id.location_code is None:
                pik_loc = ''
            else:
                pik_loc = pick.waveform_id.location_code
            # fill DataFrame with channel metadata and picking time
            cat_tab.loc[cat_tab.shape[0]] = [pick.waveform_id.network_code, pick.waveform_id.station_code, pik_loc,
                                             pick.waveform_id.channel_code,
                                             datetime.strptime(str(pick.time), '%Y-%m-%dT%H:%M:%S.%fZ')]
    return cat_tab


def add_event_data(stream, evt_param, ref_mod, sta_inv):
    """
    :param stream: data streamer of waveforms to process
    :param evt_param: dictionary containing event parameters
    :param ref_mod: reference velocity model to use for theoretical travel times
    :param sta_inv: .xml inventory containing all station info.
    :return: data streamer with event info. and DataFrame containing arrivals common to catalogue and autopicker
    """
    # initialise TauP to compute theroetical arrivals
    theory = TauPyModel(model=ref_mod)
    # initialise output DataFrame
    res_tab = pd.DataFrame({'net': pd.Series(dtype='string'), 'stn': pd.Series(dtype='string'),
                            'loc': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string'),
                            'dis': pd.Series(dtype='float64'), 'carr': pd.Series(dtype='datetime64[ms]'),
                            'aarr': pd.Series(dtype='datetime64[ms]'), 'tarr': pd.Series(dtype='datetime64[ms]')})
    # initialise list of traces to delete
    to_del = []
    for tr in stream:
        # SELECTION OF BEST AVAILABLE CHANNEL
        # list all available channels for station
        lst = stream.select(station=tr.stats.station)
        # initialise variable for selected channel
        sc = None
        # in case multiple channels available
        if len(lst) > 1:
            for trace in lst:
                sc = None
                # in case several picks exist, test each acceptable option (listed here with decreasing priority)
                if sc is None and trace.stats.channel == 'HHZ':
                    sc = lst.select(station=trace.stats.station, channel='HHZ')[0]
                if sc is None and trace.stats.channel == 'BHZ':
                    sc = lst.select(station=trace.stats.station, channel='BHZ')[0]
                if sc is None and trace.stats.channel == 'ENZ':
                    sc = lst.select(station=trace.stats.station, channel='ENZ')[0]
            # remove traces for unselected channels (only getting here if >1 channels)
            for item in lst:
                if sc is not None and item != sc:
                    to_del.append(item)
        # EPICENTRAL DISTANCE
        # find station/channel in inventory
        station = sta_inv.select(network=tr.stats.network, station=tr.stats.station,
                                 channel=tr.stats.channel, location=tr.stats.location)
        # remove trace if station/channel not in inventory
        if not station:
            to_del.append(tr)
            continue
        elif len(station) > 1:
            print(f" Multiple matches in station inventory: {tr.stats.station}")
            return None, None
        # calculate event-station distance
        dis = gdist.distance((evt_param['elat'], evt_param['elon']), (station[0].stations[0].channels[0].latitude,
                                                                      station[0].stations[0].channels[0].longitude))
        # add event-station distance to trace header
        tr.stats.distance = dis.m
        # THEORETICAL ARRIVAL
        # compute theoretical travel time for all possible first P arrivals
        theo_tt = theory.get_travel_times(source_depth_in_km=evt_param['edep'],
                                          distance_in_degree=dis.km / (2 * np.pi * rrad / 360),
                                          phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        if len(theo_tt) != 0:
            # add theoretical travel time to trace header
            tr.stats['theo_tt'] = evt_param['eori'] + timedelta(seconds=theo_tt[0].time)
        else:
            # save as NaN in case of no theoretical arrival
            tr.stats['theo_tt'] = np.nan
    # delete selected waveforms
    for tr in to_del:
        # remove trace if not already removed
        try:
            stream.remove(tr)
        except:
            continue
    return stream, res_tab


def plot_autopick_evt_sec(stream, auto_tab, cat_tab, evt_param, filt_param, fig_name=None):
    """
    :param stream: data streamer of waveforms to process
    :param auto_tab: DataFrame containing automatic picks
    :param cat_tab: DataFrame containing catalogue picks
    :param evt_param: dictionary containing event parameters
    :param filt_param: dictionary containing filter parameters
    :param fig_name: figure file name (figure is shown if None)
    :return: data streamer with event info.
    """
    # time difference with theory for best autopick
    tlim = 5.
    # plotting mode
    if fig_name:
        mpl.use('Agg')
    # create figure & axes
    fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200,
                                              gridspec_kw={'width_ratios': [2, 1, 1]})
    if not fig_name:
        plt.show(block=False)
    # AXIS 1: VELOCITY WAVEFORMS
    # show axis grids
    axis1.grid(which='both', axis='both')
    # set x-axis limits
    axis1.set_xlim([evt_param['eori'] - twin, evt_param['eori'] + twin])
    # set date format to x-axis tick labels
    axis1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # AXIS 2: RESIDUAL OVER DISTANCE PLOT
    # show axis grids
    axis2.grid(which='both', axis='both')
    # set x-axis limits
    axis2.set_xlim([-5., 5.])
    # AXIS 3: LOCAL STATIONS & EVENT MAP
    # define map boundaries & resolution
    m = Basemap(projection='cyl', llcrnrlon=mgrd[2] + .5, llcrnrlat=mgrd[0],
                urcrnrlon=mgrd[3] - .5, urcrnrlat=mgrd[1], resolution='i', ax=axis3)
    # draw map
    m.drawmapboundary(fill_color='none')
    # fill continents
    m.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels = [left,right,top,bottom])
    m.drawparallels(np.arange(m.llcrnrlat, m.urcrnrlat + 1, 2.), labels=[True, False, True, False])
    m.drawmeridians(np.arange(m.llcrnrlon, m.urcrnrlon + 1, 2.), labels=[True, False, False, True])
    # faults
    fid = open('/home/lewis/.seiscomp/bna/ActiveFaults/activefaults.bna', 'r')
    flts = fid.readlines()
    fid.close()
    flt = []
    hf = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            hf, = axis3.plot(flt.lon, flt.lat, '.6', label='Faults')
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # quarries
    fid = open('/home/lewis/.seiscomp/bna/Quarries/quarries.bna', 'r')
    flts = fid.readlines()
    fid.close()
    hq = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            hq, = axis3.plot(flt.lon, flt.lat, 's', markersize=5, color='white', label='Quarries', alpha=.1)
        else:
            l_line = flts[iii].split(',')
            if l_line != ['\n']:
                flt.loc[flt.shape[0]] = [float(l_line[1].replace(' ', '').replace('\n', '')),
                                         float(l_line[0].replace(' ', '').replace('\n', ''))]
    # LOOP OVER TRACES
    # intialise counters
    n_trace = 0
    n_redpk = 0
    # initialise residual tables
    tres_tab = []
    theo_tab = []
    # initialise tables for legend
    h1 = []
    h2 = []
    h3 = []
    h4 = []
    # initialise tables for map legend
    hm1 = []
    hm2 = []
    # initialise list of station labels
    stn_lbl = []
    for tr in stream:
        # station label for y-axis
        stn_lbl.append(f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}")
        # counter
        n_trace += 1
        # time vector
        t_vec = np.arange(0, len(tr)) * np.timedelta64(int(tr.stats.delta * 1000), '[ms]')\
            + np.datetime64(str(tr.stats.starttime)[:-1])
        # plot waveform
        h1, = axis1.plot(t_vec, tr.data / tr.max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # plot theoretical travel time
        if tr.stats.theo_tt:
            h2, = axis1.plot([tr.stats.theo_tt, tr.stats.theo_tt], [n_trace - 1, n_trace + 1],
                             color='blue', linestyle='dotted', label=vel_mod)
        # plot station in map
        st = isn_inv.select(network=tr.stats.network, station=tr.stats.station, channel=tr.stats.channel)
        if not st:
            print(f'Missing station: {stn_lbl[n_trace-1]}')
        # use different symbols and colours for different network codes
        hm1, = axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                          '^', mfc='blue', mec='none', markersize=5, alpha=.7, label=tr.stats.network)
        # CATALOGUE PICKS
        # initialise index for pick of interest
        k_cpic = None
        if not cat_tab.empty:
            # make sure station has picks
            if not cat_tab[(cat_tab.sta == tr.stats.station)].empty:
                # identify picked channel
                k_cpic = cat_tab.index[(cat_tab.sta == tr.stats.station) & (cat_tab.net == tr.stats.network)
                                       & (cat_tab.chn == tr.stats.channel)].to_list()
                if not k_cpic:
                    continue
                # make sure only one match exists
                if len(k_cpic) == 1:
                    k_cpic = k_cpic[0]
                else:
                    print(f"More than one pick was found for {stn_lbl[n_trace-1]}")
                    k_cpic = k_cpic[0]
                # plot pick on top of waveform
                h3, = axis1.plot([cat_tab.pick[k_cpic], cat_tab.pick[k_cpic]], [n_trace - 1, n_trace + 1],
                                 color='orange', label='Catalogue')
                # right-hand side marker to quickly know which station has picks
                axis1.plot(1.02, n_trace/(len(stream)+1), 'o', markersize=5, mfc='orange', mec='none',
                           alpha=.7, clip_on=False, transform=axis1.transAxes)
                # highlight station in map
                axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                           's', markersize=7, mec='orange', mfc='none', alpha=.7)
        # AUTOMATIC PICKS
        # initialise index for pick of interest
        k_apic = None
        if not auto_tab.empty:
            # make sure station has picks
            if not auto_tab[(auto_tab.sta == tr.stats.station)].empty:
                # copy rows of DataFrame for channel
                temp_tab = auto_tab[(auto_tab.sta == tr.stats.station) & (auto_tab.net == tr.stats.network)
                                    & (auto_tab.chn == tr.stats.channel)]
                # add entry to DataFrame for travel-time differences
                temp_tab = temp_tab.assign(tdif=pd.Series([None] * len(temp_tab), dtype='float'))
                # case no automatic picks for channel
                if temp_tab.empty:
                    continue
                # choose best pick in case channel has several
                k_apic = None
                # time difference between automatic picks and theoretical arrival
                temp_tab['tdif'] = [abs((xx - tr.stats.theo_tt).total_seconds()) for xx in temp_tab.pick.to_list()]
                # loop over all picks for channel
                for kk in temp_tab.index:
                    # find pick >OT and <10s difference with theoretical arrival
                    if temp_tab.pick[kk] > evt_param['eori'] and kk == temp_tab.tdif.idxmin()\
                            and temp_tab.tdif[kk] < tlim:
                        h4, = axis1.plot([temp_tab.pick[kk], temp_tab.pick[kk]], [n_trace - 1, n_trace + 1],
                                         color='red', label='Automatic')
                        # right-hand side marker to quickly know which station has picks
                        axis1.plot(1.01, n_trace/(len(stream)+1), 'o', markersize=5, mfc='purple', mec='red',
                                   alpha=.7, clip_on=False, transform=axis1.transAxes)
                        # highlight station in map
                        axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                   'o', markersize=7, color='red', mfc='none', alpha=.7)
                        k_apic = kk
                        # counter
                        n_redpk += 1
                    else:
                        h4, = axis1.plot([temp_tab.pick[kk], temp_tab.pick[kk]], [n_trace - 1, n_trace + 1],
                                         color='purple', label='Automatic')
                        # right-hand side marker to quickly know which station has picks
                        axis1.plot(1.01, n_trace/(len(stream)+1), 'o', markersize=5, mfc='purple', mec='none',
                                   alpha=.7, clip_on=False, transform=axis1.transAxes)
                        # highlight station in map
                        axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                   'o', markersize=7, color='purple', mfc='none', alpha=.7)
        # residual w.r.t. catalogue pick (if both exist)
        if k_cpic is not None and k_apic is not None:
            # table for residuals (for statistics)
            tres_tab.append((auto_tab.pick[k_apic] - cat_tab.pick[k_cpic]).total_seconds())
            # residual plot
            axis2.plot((auto_tab.pick[k_apic] - cat_tab.pick[k_cpic]).total_seconds(), n_trace,
                       'o', markersize=5, mfc='orange', mec='none', alpha=.7)
        # residual w.r.t. theoretical pick (if automatic pick exists)
        if k_apic is not None:
            # table for residuals (for statistics)
            theo_tab.append((auto_tab.pick[k_apic] - tr.stats.theo_tt).total_seconds())
            # residual plot
            axis2.plot((auto_tab.pick[k_apic] - tr.stats.theo_tt).total_seconds(), n_trace,
                       'o', markersize=5, mfc='blue', mec='none', alpha=.7)
    # display number of automatic picks <5 s from theoretical arrival
    axis1.text(.5, 1.01, f"N={n_redpk}",
               ha='center', va='center', fontweight='bold', fontsize=8, color='red', transform=axis1.transAxes)
    # display statistics on residuals
    if len(tres_tab) > 1:
        axis2.text(.01, .03, f"N={len(tres_tab)}: {statistics.mean(tres_tab):.2f} \u00B1"
                             f" {statistics.stdev(tres_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='orange', transform=axis2.transAxes)
    if len(theo_tab) > 1:
        axis2.text(.01, .01, f"N={len(theo_tab)}: {statistics.mean(theo_tab):.2f} \u00B1"
                             f" {statistics.stdev(theo_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='blue', transform=axis2.transAxes)
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
    axis1.text(-.01, 1.01, f"N={n_trace}", ha='right', va='center', transform=axis1.transAxes)
    # replace numerical tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace + 1, 1))
    axis1.set_yticklabels(stn_lbl, fontsize=5)
    axis2.set_yticklabels([])
    # axis labels
    axis1.set_xlabel('Time - OT [s]', fontweight='bold')
    axis1.set_ylabel('Station', fontweight='bold')
    axis2.set_xlabel('\u0394t [s]', fontweight='bold')
    # MAP
    # show location of stations without automatic picks
    for nt in isn_inv:
        for st in nt.stations:
            if not stream.select(network=nt.code, station=st.code):
                axis3.plot(st.longitude, st.latitude, 'x', markersize=7, color='black', mfc='none', alpha=.7)
    # plot event location
    he, = axis3.plot(evt_param['elon'], evt_param['elat'], '*',
                     color='orange', markersize=10, markeredgecolor='black', label='Event')
    # map legend
    if not hm1:
        hm = [hm2, he, hf, hq]
    elif not hm2:
        hm = [hm1, he, hf, hq]
    else:
        hm = [hm1, hm2, he, hf, hq]
    axis3.legend(handles=hm, loc='upper left', fontsize=8)
    # AXIS 4: INSET MAP FOR REGIONAL SETTINGS
    axis4 = inset_axes(axis3, '30%', '18%', loc='lower left')
    m = Basemap(projection='cyl', llcrnrlon=rgrd[2], llcrnrlat=rgrd[0],
                urcrnrlon=rgrd[3], urcrnrlat=rgrd[1], resolution='l', ax=axis4)
    # draw map
    m.drawmapboundary(fill_color='white')
    # fill continents
    m.fillcontinents(color='0.8', lake_color='white')
    # highlight area of interest
    axis4.plot([mgrd[2] + .5, mgrd[2] + .5, mgrd[3] - .5, mgrd[3] - .5, mgrd[2] + .5],
               [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]], color='red')
    # plot event location
    axis4.plot(evt_param['elon'], evt_param['elat'], '*', color='red', markersize=10, markeredgecolor='black')
    # display magnitude if exists
    if evt_param['emag'] and not np.isnan(evt_param['emag']):
        if evt_param['elon'] < mgrd[2]+(mgrd[3]-mgrd[2])/2.:
            axis4.text(evt_param['elon']-2., evt_param['elat'], f"M{evt_param['emag']:3.1f}",
                       ha='right', va='center', color='red', clip_on=True, fontsize=8)
        else:
            axis4.text(evt_param['elon']+2., evt_param['elat'], f"M{evt_param['emag']:3.1f}",
                       ha='left', va='center', color='red', clip_on=True, fontsize=8)
    # figure title
    tit1 = f"{evt_param['eori'].strftime('%d/%m/%Y %H:%M:%S')}" \
           f" \u2013 {evt_param['edep']:.2f} km \u2013 M{evt_param['emag']:3.1f}"
    tit2 = f"HP: {filt_param['rmhp']:.2f} [s] \u2013 Taper: {filt_param['taper']:.2f} [s] \u2013 BP: " \
           f"{filt_param['bworder']} / {filt_param['bwminf']:.2f} / {filt_param['bwmaxf']:.2f} [Hz]"
    tit3 = f"STA/LTA: {filt_param['sta']:.2f} / {filt_param['lta']:.2f} [s] " \
           f"\u2013 Trigger: {filt_param['trigon']:.2f} / {filt_param['trigoff']:.2f}"
    fig.suptitle(tit1 + '\n' + tit2 + '\n' + tit3, fontweight='bold')
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
    return tres_tab, theo_tab


########################################################################################################################
# input parameters
ntw = 'IS'
chn = '(B|H|E)(H|N)Z'
pic = 'AIC'
fpar = {'rmhp': 10., 'taper': 30., 'bworder': 4, 'bwminf': 4., 'bwmaxf': 8.,
        'sta': .2, 'lta': 10., 'trigon': 3., 'trigoff': 1.5}

# area of interest
rrad = 6371.
igrd = [29., 34., 34., 36.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 25., 45.]
vel_mod = 'giimod'

# working directory
wdir = '/home/lewis/autopicker-rt'
# data archive directory
adir = '/net/172.16.46.200/archive/jqdata/archive'
# FDSN database
isn_client = Client('http://172.16.46.102:8181/')

# retrieve station inventory
if path.exists(f"{wdir}/inventory.xml") != 0:
    print('ISN inventory file already exists:')
    os.system(f"ls -lh {wdir}/inventory.xml")
    print()
else:
    isn_inv = isn_client.get_stations(network='IS', channel='ENZ, HHZ, BHZ', level='response')
    isn_inv.write(f'{wdir}/inventory.xml', level='response', format='STATIONXML')
# read station inventory
isn_inv = read_inventory(f"{wdir}/inventory.xml", format='STATIONXML')

# list events since 2022-10-26 (activation of autopicker)
evt_lst = isn_client.get_events(starttime=UTCDateTime(datetime.strptime('2022-10-26 00:00:00', '%Y-%m-%d %H:%M:%S')),
                                endtime=UTCDateTime(datetime.now()),
                                includearrivals=True, orderby='time-asc')
# event time window
twin = timedelta(minutes=5)
# loop over listed events
for evt in evt_lst:
    ####################################################################################################################
    # EXTRACT EVENT PARAMETERS
    if not evt.preferred_magnitude():
        epar = {'evid': str(evt.resource_id).replace('smi:org.gfz-potsdam.de/geofon/', ''),
                'elat': evt.preferred_origin().latitude, 'elon': evt.preferred_origin().longitude,
                'edep': evt.preferred_origin().depth/1000., 'emag': np.nan, 'eori':
                    datetime.strptime(str(evt.preferred_origin().time).replace('+00:00', ''), '%Y-%m-%dT%H:%M:%S.%fZ')}
    else:
        epar = {'evid': str(evt.resource_id).replace('smi:org.gfz-potsdam.de/geofon/', ''),
                'elat': evt.preferred_origin().latitude, 'elon': evt.preferred_origin().longitude,
                'edep': evt.preferred_origin().depth / 1000., 'emag': evt.preferred_magnitude().mag, 'eori':
                    datetime.strptime(str(evt.preferred_origin().time).replace('+00:00', ''), '%Y-%m-%dT%H:%M:%S.%fZ')}
    print(f"{epar['evid']}: {datetime.strftime(epar['eori'], '%d/%m/%Y %H:%M:%S.%f')} (M{epar['emag']:3.1f})")
    # skip event without any picks
    if not evt.picks:
        print(' No picks found')
        print()
        continue
    # skip regional and teleseismic events
    if epar['elat'] < mgrd[0] or epar['elat'] > mgrd[1] or epar['elon'] < mgrd[2] or epar['elon'] > mgrd[3]:
        print(' Teleseismic or regional event')
        print()
        continue
    # skip event if figure already exists
    if os.path.exists(f"{wdir}/{epar['evid']}.png") != 0:
        print('Figure already exists:')
        os.system(f"ls -lh {wdir}/{epar['evid']}.png")
        print()
        continue
    ####################################################################################################################
    # RETRIEVE CATALOGUE PICKS
    ctab = pd.DataFrame({'net': pd.Series(dtype='string'), 'sta': pd.Series(dtype='string'),
                         'loc': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string'),
                         'pick': pd.Series(dtype='datetime64[ms]')})
    for pik in evt.picks:
        if pik.phase_hint[0] == 'P':
            if pik.waveform_id.location_code is None:
                loc = ''
            else:
                loc = pik.waveform_id.location_code
            ctab.loc[ctab.shape[0]] = [pik.waveform_id.network_code, pik.waveform_id.station_code,
                                       loc, pik.waveform_id.channel_code,
                                       datetime.strptime(str(pik.time), '%Y-%m-%dT%H:%M:%S.%fZ')]
    print(f' {len(ctab)} catalogue picks')
    ####################################################################################################################
    # RETRIEVE AUTOMATIC PICKS
    if path.exists(f"{wdir}/{epar['evid']}.xml") == 0:
        os.system(f"dump_picks -t '{datetime.strftime(epar['eori']-twin, '%Y-%m-%d %H:%M:%S')}~"
                  f"{datetime.strftime(epar['eori']+twin, '%Y-%m-%d %H:%M:%S')}' > {wdir}/{epar['evid']}.xml")
    else:
        print(' Pick file already exists:')
        os.system(f"ls -lh {wdir}/{epar['evid']}.xml")
    # read pick file
    atab = read_autopick_xml(f"{wdir}/{epar['evid']}.xml", 'P', pic)
    print(f' {len(atab)} automatic picks')
    ####################################################################################################################
    # RETRIEVE WAVEFORM DATA
    if path.exists(f"{wdir}/{epar['evid']}.mseed") == 0:
        mfile = get_traces_deci(epar['evid'], epar['eori'], twin, ntw, chn, adir)
    else:
        print(' Waveform data already exists:')
        os.system(f"ls -lh {wdir}/{epar['evid']}.mseed")
    ####################################################################################################################
    # WAVEFORM DATA PROCESSING
    # read waveform data
    isn_traces = read(f"{wdir}/{epar['evid']}.mseed").merge()
    # add event data to waveforms
    isn_traces, rtab = add_event_data(isn_traces, epar, vel_mod, isn_inv)
    # sort according to newly added distance to event
    isn_traces.sort(['distance'], reverse=True)
    # apply taper to all traces
    isn_traces.taper(max_percentage=.5, type='cosine', max_length=fpar['taper'], side='left')
    # apply high-pass filter to all traces
    isn_traces.filter('highpass', freq=1./fpar['rmhp'])
    # remove trend from all traces
    isn_traces.detrend('spline', order=3, dspline=500)
    # apply Butterworth band-pass filter to all traces
    isn_traces.filter('bandpass', freqmin=fpar['bwminf'], freqmax=fpar['bwmaxf'], corners=fpar['bworder'])
    ####################################################################################################################
    # PLOT FIGURE
    plot_autopick_evt_sec(isn_traces, atab, ctab, epar, fpar, f"{wdir}/{epar['evid']}.png")
    print()
