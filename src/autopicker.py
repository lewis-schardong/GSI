########################################################################################################################
import os
from os import path
import re
import filecmp
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
from scipy import signal
from datetime import datetime, timedelta
import xml.etree.ElementTree as ETree
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def get_traces_deci(evt_id, ori_time, time_win):
    """
    :param evt_id: event ID
    :param ori_time: event origin time
    :param time_win: half-length of time window
    :return: full path of created miniSEED file
    """
    # define file extension based on time window length
    f_ext = ''
    if time_win == timedelta(minutes=5):
        f_ext = '10m'
    elif time_win == timedelta(hours=5):
        f_ext = '10h'
    # load miniSEED data
    mseed = f'{wdir}/{idat}/{evt_id}.{f_ext}.raw.mseed'
    if path.exists(mseed) == 0:
        # import miniSEED file
        tbeg = str(datetime.strftime(ori_time - time_win, '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(ori_time + time_win, '%Y-%m-%d %H:%M:%S'))
        os.system(f'scart -dsE -n "{ntw}" -c "{chn}" -t "{tbeg}~{tend}" {adir} > {mseed}')
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
        # read raw miniSEED file
        stream = read(mseed)
        # ensuring 50-Hz sampling rate for all traces (e.g. if time window crosses midnight and data is missing for one of the two dates, we get non-decimated data as replacement)
        stream.resample(50.0, window='hann')
        # merging all traces (e.g. if time window crosses midnight, data from both dates need to be merged)
        stream.merge(fill_value='interpolate')
        # remove problematic channels
        for trace in stream:
            if trace.stats.station == 'KRPN' or (trace.stats.station == 'EIL' and trace.stats.channel == 'BHZ') or (trace.stats.station == 'GEM' and trace.stats.channel == 'BHZ') \
                    or (trace.stats.station == 'KFSB' and trace.stats.channel == 'HHZ' and trace.stats.location == '22')\
                    or (trace.stats.station == 'HRFI' and trace.stats.channel == 'HHZ' and trace.stats.location == '')\
                    or (trace.stats.channel == 'HHZ' and ('MMA' in trace.stats.station or 'MMB' in trace.stats.station or 'MMC' in trace.stats.station)):
                stream.remove(trace)
        # write miniSEED file
        stream.write(mseed)
    else:
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    return mseed


def process_mseed(mseed_in, sta_inv, filt_param):
    """
    :param mseed_in: full path to miniSEED file to process
    :param sta_inv: station inventory
    :param filt_param: dictionary containing filter parameters
    :return: data streamer with event info.
    """
    # read raw miniSEED file
    stream = read(mseed_in).merge()
    mseed_out = mseed_in.replace('.raw', '')
    if path.exists(mseed_out) != 0:
        print(f' Waveform data already processed: {mseed_out}')
        return mseed_out
    # remove response from all traces
    try:
        stream.remove_response(output='VEL', inventory=sta_inv)
    except:
        for trace in stream:
            print(f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}')
            trace.remove_response(output='VEL', inventory=sta_inv)
            exit()
    # apply taper to all traces
    stream.taper(max_percentage=.5, type='cosine', max_length=filt_param['taper'], side='left')
    # apply high-pass filter to all traces
    stream.filter('highpass', freq=1./filt_param['rmhp'])
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply Butterworth band-pass filter to all traces
    stream.filter('bandpass', freqmin=filt_param['bwminf'], freqmax=filt_param['bwmaxf'], corners=filt_param['bworder'])
    # write miniSEED file
    stream.write(mseed_out)
    return mseed_out


def read_autopick_xml(xml_path, stream, phase='P', picker=''):
    """
    :param xml_path: path to .XML file containing picks to read
    :param stream: data streamer of waveforms to process
    :param phase: seismic phase of interest
    :param picker: picker used
    :return: data streamer containing automatic picks in trace headers and total number of picks found
    """
    npic = 0
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
            trace = stream.select(network=pik_net, station=str2[3], location=pik_loc, channel=str2[5])
            if not trace:
                continue
            else:
                trace = trace[0]
            # initialise pick table if needed
            if not hasattr(trace.stats, 'auto_tt'):
                trace.stats['auto_tt'] = []
            trace.stats.auto_tt.append(datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ'))
            npic += 1
    return stream, npic


def get_catalogue_picks(client, evt_id, stream):
    """
    :param client: Obspy FDSN client to retrieve data from
    :param evt_id: event ID according to the GSI's FDSN database
    :param stream: data streamer of waveforms to process
    :return: data streamer containing catalogue picks in trace headers and total number of picks found
    """
    npic = 0
    # retrieve event data
    evt_lst = client.get_events(eventid=evt_id, includearrivals=True)[0]
    for pik in evt_lst.picks:
        if pik.phase_hint[0] == 'P':
            if pik.waveform_id.location_code is None:
                pik_loc = ''
            else:
                pik_loc = pik.waveform_id.location_code
            trace = stream.select(network=pik.waveform_id.network_code, station=pik.waveform_id.station_code,
                                  location=pik_loc, channel=pik.waveform_id.channel_code)
            if not trace:
                continue
            else:
                trace = trace[0]
            # initialise pick table if needed
            if not hasattr(trace.stats, 'cata_tt'):
                trace.stats['cata_tt'] = []
            trace.stats.cata_tt.append(datetime.strptime(str(pik.time), '%Y-%m-%dT%H:%M:%S.%fZ'))
            npic += 1
    return stream, npic


def add_event_data(stream, evt_dict, ref_mod, sta_inv):
    """
    :param stream: data streamer of waveforms to process
    :param evt_dict: dictionary containing event parameters
    :param ref_mod: reference velocity model to use for theoretical travel times
    :param sta_inv: .xml inventory containing all station info.
    :return: data streamer with event info.
    """
    # test whether event or continuous data needs processing
    if_evt = False
    if stream[0].stats.endtime-stream[0].stats.starttime <= 600.:
        if_evt = True
    # initialising TauP
    theory = TauPyModel(model=ref_mod)
    if if_evt:
        # DataFrame initialisation for common arrivals (hand/auto)
        res_tab = pd.DataFrame({'net': pd.Series(dtype='string'), 'stn': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                                'chn': pd.Series(dtype='string'), 'dis': pd.Series(dtype='float64'), 'carr': pd.Series(dtype='datetime64[ms]'),
                                'aarr': pd.Series(dtype='datetime64[ms]'), 'tarr': pd.Series(dtype='datetime64[ms]')})
    else:
        # empty DataFrame
        res_tab = pd.DataFrame()
    # traces to delete
    to_del = []
    k = 0
    while k < len(stream):
        # selection of best available channel
        lst = stream.select(station=stream[k].stats.station)
        ss = None
        # in case multiple channels available
        if len(lst) > 1:
            for trace in lst:
                ss = None
                # in case several picks exist, test each acceptable option (ranked with decreasing priority)
                if ss is None and trace.stats.channel == 'HHZ':
                    ss = lst.select(station=trace.stats.station, channel='HHZ')
                if ss is None and trace.stats.channel == 'BHZ':
                    ss = lst.select(station=trace.stats.station, channel='BHZ')
                if ss is None and trace.stats.channel == 'ENZ':
                    ss = lst.select(station=trace.stats.station, channel='ENZ')
            # remove traces for unselected channels (only getting here if >1 channels)
            for item in lst:
                if (ss and ss is not None) and item != ss[0]:
                    to_del.append(item)
        # find station in inventory
        station = sta_inv.select(network=stream[k].stats.network, station=stream[k].stats.station,
                                 channel=stream[k].stats.channel, location=stream[k].stats.location)
        # remove traces if station/channel not in inventory
        if not station:
            to_del.append(stream[k])
            k += 1
            continue
        elif len(station) > 1:
            print(f" Multiple matches in station inventory: {stream[k].stats.station}")
            return None, None
        # calculate event-station distance
        dis = gdist.distance((evt_dict['elat'], evt_dict['elon']), (station[0].stations[0].channels[0].latitude, station[0].stations[0].channels[0].longitude))
        stream[k].stats.distance = dis.m
        # compute theoretical travel time
        theo_tt = theory.get_travel_times(source_depth_in_km=evt_dict['edep'], distance_in_degree=dis.km / (2 * np.pi * rrad / 360), phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        # add theoretical travel time to streamer header
        if len(theo_tt) != 0:
            stream[k].stats['theo_tt'] = evt_dict['eori'] + timedelta(seconds=theo_tt[0].time)
        else:
            stream[k].stats['theo_tt'] = np.nan
        if if_evt:
            # selection of picks for statistics and output file
            kh = None
            # catalogue picks
            if hasattr(stream[k].stats, 'cata_tt') and len(stream[k].stats.cata_tt) > 0:
                if len(stream[k].stats.cata_tt) == 1:
                    kh = 0
                elif len(stream[k].stats.cata_tt) > 1:
                    print(' Multiple results in catalogue picks')
                    return None, None
                else:
                    # print(' No result in catalogue picks')
                    k += 1
                    continue
            ka = None
            # automatic picks
            if hasattr(stream[k].stats, 'auto_tt') and len(stream[k].stats.auto_tt) > 0:
                stream[k].stats['tdiff'] = []
                if len(stream[k].stats.auto_tt) > 1:
                    ka = None
                    # find pick closest to theoretical arrival
                    stream[k].stats.tdiff = [abs(xx - (evt_dict['eori'] + timedelta(seconds=theo_tt[0].time))).total_seconds() for xx in stream[k].stats.auto_tt]
                    for jj in range(len(stream[k].stats.auto_tt)):
                        if stream[k].stats.auto_tt[jj] > evt_dict['eori'] and jj == stream[k].stats.tdiff.index(min(stream[k].stats.tdiff)):
                            ka = jj
                            break
                else:
                    ka = 0
            # table for residuals (for statistics)
            if kh is not None and ka is not None:
                res_tab.loc[res_tab.shape[0]] = [stream[k].stats.network, stream[k].stats.station, stream[k].stats.location,
                                                 stream[k].stats.channel, stream[k].stats.distance/1000.,
                                                 stream[k].stats.cata_tt[kh], stream[k].stats.auto_tt[ka], stream[k].stats.theo_tt]
            elif kh is None and ka is not None:
                res_tab.loc[res_tab.shape[0]] = [stream[k].stats.network, stream[k].stats.station, stream[k].stats.location,
                                                 stream[k].stats.channel, stream[k].stats.distance/1000.,
                                                 np.datetime64('NaT'), stream[k].stats.auto_tt[ka], stream[k].stats.theo_tt]
        k += 1
    # delete selected waveforms
    for trace in to_del:
        try:
            stream.remove(trace)
        except:
            continue
    return stream, res_tab


def plot_autopick_evt_sec(stream, auto_tab, hand_tab, evt_param, filt_param, index=None, fig_name=None):
    """
    :param stream: data streamer of waveforms to process
    :param auto_tab: DataFrame containing automatic picks
    :param hand_tab: DataFrame containing catalogue picks
    :param evt_param: dictionary containing event parameters
    :param filt_param: dictionary containing filter parameters
    :param index: indexes to sort waveforms
    :param fig_name: figure file name (figure is shown if None)
    :return: data streamer with event info.
    """
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
    fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200,
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
                    for station in isn_inv.networks[0]:
                        if stream[jjj].stats.station == station.code:
                            axis3.plot(station.longitude, station.latitude, 's', markersize=7, color='orange', mfc='none', alpha=.7, label='Autopick')
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
                if len(temp_tab) > 1:
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
                    for station in isn_inv.networks[0]:
                        if stream[jjj].stats.station == station.code:
                            axis3.plot(station.longitude, station.latitude, 'o', markersize=7, color='purple', mfc='none', alpha=.7)
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
        axis2.text(.95 * xlim2[0], 4, f"N={n_tres}: {np.sqrt(np.mean(np.array(tres_tab)**2)):.2f} s",
                   fontweight='bold', fontsize=8, color='orange')
    if n_theo > 1:
        axis2.text(.95 * xlim2[0], 1, f"N={n_theo}: {np.sqrt(np.mean(np.array(theo_tab)**2)):.2f} s",
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
        for station in isn_inv.networks[0]:
            if stream[jjj].stats.station == station.code:
                if stream[jjj].stats.network == 'IS':
                    hm1, = axis3.plot(station.longitude, station.latitude, 'b^', markersize=5, alpha=.7, mec='none', label=stream[jjj].stats.network)
                elif stream[jjj].stats.network == 'GE':
                    hm2, = axis3.plot(station.longitude, station.latitude, 'cs', markersize=5, alpha=.7, mec='none', label=stream[jjj].stats.network)
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
    axis4.plot([mgrd[2]+.5, mgrd[2]+.5, mgrd[3]-.5, mgrd[3]-.5, mgrd[2]+.5], [mgrd[0], mgrd[1], mgrd[1], mgrd[0], mgrd[0]], color='red')
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
    fig.suptitle(tit1 + '\n' + tit2 + '\n' + tit3, fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.07, right=.98, wspace=.1)
    # show or save figure
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()
    return tres_tab, n_tres, theo_tab, n_theo


def plot_autopick_cont_sec(stream, win_tab, fig_name=None):
    """
    :param stream: data streamer of waveforms to process
    :param win_tab: list of datetimes defining the beginning of 30-s time windows containing >6 picks
    :param fig_name: file name & path of the output figure (figure is displayed if none is provided)
    :return: nothing
    """
    # define axis limits for both axes
    tmin = datetime.strptime(str(min([trace.stats.starttime for trace in stream])), '%Y-%m-%dT%H:%M:%S.%fZ')
    tmax = datetime.strptime(str(max([trace.stats.endtime for trace in stream])), '%Y-%m-%dT%H:%M:%S.%fZ')
    xlim = [tmin, tmax]
    # create figure
    fig, axes = plt.subplots(figsize=(18, 9), dpi=200)
    # set date format to x-axis tick labels
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # show axis grids
    plt.grid(which='both', axis='x')
    # set x-axis limits
    axes.set_xlim(xlim)
    # show catalogue events
    evt_loc = isn_client.get_events(starttime=xlim[0], endtime=xlim[1])
    # explosions counter
    n_exp = 0
    for event in evt_loc:
        # normalised x-axis coordinate (because of problems with datetimes and JSON)
        xpos = (event.preferred_origin().time.datetime-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
        if event.magnitudes:
            # plot marker
            plt.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime], [0, len(stream)+1], color='red', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.01, f"M{event.preferred_magnitude().mag:3.1f}",
                     color='red', fontsize=8, ha='center', va='bottom', clip_on=False, transform=axes.transAxes)
        else:
            n_exp += 1
            # plot marker
            plt.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime], [0, len(stream)+1], color='green', linewidth=2, alpha=.5)
            # display event type
            plt.text(xpos, 1.03, 'EXP', color='green', fontsize=8, ha='center', va='bottom', clip_on=False, transform=axes.transAxes)
    # FDSN client for  EMSC events
    tele_client = Client('EMSC')
    # M>5 teleseismic events catalogue
    evt_tel = []
    try:
        evt_tel = tele_client.get_events(starttime=xlim[0], endtime=xlim[1], minmagnitude=5)
    except:
        print(' No M>5 teleseismic events')
    if evt_tel:
        for event in evt_tel:
            if abs((event.preferred_origin().time.datetime-stream[0].stats.origin_time).total_seconds()) < 60.:
                if len(evt_tel) > 1:
                    e1 = evt_tel.filter(f"time < {datetime.strftime(stream[0].stats.origin_time - timedelta(minutes=1), '%Y-%m-%dT%H:%M')}")
                    e2 = evt_tel.filter(f"time > {datetime.strftime(stream[0].stats.origin_time + timedelta(minutes=1), '%Y-%m-%dT%H:%M')}")
                    for e in e2:
                        e1.append(e)
                    evt_tel = e1
                    del e2
                else:
                    evt_tel = []
                continue
            # normalised x-axis coordinate (because of problems with datetimes and JSON)
            xpos = (event.preferred_origin().time.datetime-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
            # plot marker
            plt.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime], [0, len(stream)+1], color='purple', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.07, f"M{event.preferred_magnitude().mag:3.1f}", color='purple', fontsize=8, ha='center', va='bottom', clip_on=False, transform=axes.transAxes)
    # M>3 regional events catalogue
    evt_reg = []
    try:
        evt_reg = tele_client.get_events(starttime=xlim[0], endtime=xlim[1], minmagnitude=3, minlatitude=23, maxlatitude=43, minlongitude=20, maxlongitude=50)
    except:
        print('No M>3 regional events')
    if evt_reg:
        for event in evt_reg:
            if abs((event.preferred_origin().time.datetime-stream[0].stats.origin_time).total_seconds()) < 60.:
                if len(evt_reg) > 1:
                    e1 = evt_reg.filter(f"time < {datetime.strftime(stream[0].stats.origin_time - timedelta(minutes=1), '%Y-%m-%dT%H:%M')}")
                    e2 = evt_reg.filter(f"time > {datetime.strftime(stream[0].stats.origin_time + timedelta(minutes=1), '%Y-%m-%dT%H:%M')}")
                    for e in e2:
                        e1.append(e)
                    evt_reg = e1
                    del e2
                else:
                    evt_reg = []
                continue
            # normalised x-axis coordinate (because of problems with datetimes and JSON)
            xpos = (event.preferred_origin().time.datetime-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
            # plot marker
            plt.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime], [0, len(stream)+1], color='orange', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.05, f"M{event.preferred_magnitude().mag:3.1f}", color='orange', fontsize=8, ha='center', va='bottom', clip_on=False, transform=axes.transAxes)
    # display total number of stations
    plt.text(-.01, 1., f"Ns={len(stream)}", ha='right', va='bottom', fontsize=8, transform=axes.transAxes)
    # display total number of local events
    plt.text(1.01, 1.01, f"Nl={len(evt_loc) - n_exp}", ha='left', va='bottom', color='red', fontsize=8, transform=axes.transAxes)
    # display total number of blasts
    plt.text(1.01, 1.03, f"Nx={n_exp}", ha='left', va='bottom', color='green', fontsize=8, transform=axes.transAxes)
    # display total number of regional events
    plt.text(1.01, 1.05, f"Nr={len(evt_reg)}", ha='left', va='bottom', color='orange', fontsize=8, transform=axes.transAxes)
    # display total number of teleseismic events
    plt.text(1.01, 1.07, f"Nt={len(evt_tel)}", ha='left', va='bottom', color='purple', fontsize=8, transform=axes.transAxes)
    # list number of autopicks per station
    plst = [len(wf.stats.auto_tt) for wf in isn_traces if hasattr(wf.stats, 'auto_tt')]
    # initialise counter
    n_trace = 0
    # loop over stream channels
    for trace in stream:
        n_trace += 1
        # plot waveforms
        t_vec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]') + np.datetime64(str(trace.stats.starttime)[:-1])
        plt.plot(t_vec, trace.data/trace.max() + n_trace, color='grey', alpha=.7, label='Velocity')
        if hasattr(trace.stats, 'auto_tt'):
            # number of picks per station
            # if len(trace.stats.auto_tt) > (len(evt_loc.filter('magnitude > 3')) + len(evt_reg)):
            if len(trace.stats.auto_tt) > np.nanmean(plst)+2.5*np.nanstd(plst):
                # show number of picks
                plt.text(1.01, n_trace/(len(stream)+1), f"Np={len(trace.stats.auto_tt)}",
                         color='red', fontsize=5, clip_on=False, ha='left', va='center', transform=axes.transAxes)
                # show station/channel name
                plt.text(-.01, n_trace/(len(stream)+1), f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                         color='red', fontsize=5, clip_on=False, ha='right', va='center', transform=axes.transAxes)
            else:
                # show number of picks
                plt.text(1.01, n_trace/(len(stream)+1), f"Np={len(trace.stats.auto_tt)}",
                         fontsize=5, clip_on=False, ha='left', va='center', transform=axes.transAxes)
                # show station/channel name
                plt.text(-.01, n_trace/(len(stream)+1), f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                         fontsize=5, clip_on=False, ha='right', va='center', transform=axes.transAxes)
            for at in trace.stats.auto_tt:
                plt.plot([at, at], [n_trace-.5, n_trace+.5], color='blue', label='Autopick')
        else:
            # show station/channel name
            plt.text(-.01, n_trace/(len(stream)+1), f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                     fontsize=5, clip_on=False, ha='right', va='center', transform=axes.transAxes)
    # 30-s time windows with >6 picks
    for win in win_tab:
        axes.fill_betweenx([0, n_trace+1], win-timedelta(minutes=2.5), win+timedelta(minutes=2.5), color='green', alpha=.5)
    # set y-axis limits
    axes.set_ylim([0, n_trace+1])
    # remove y-axis tick labels
    axes.set_yticklabels([])
    # set x-axis font size
    axes.tick_params(axis='x', which='major', labelsize=10)
    # show or save figure
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()
    return


########################################################################################################################
# input parameters
exp = 3
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
# wdir = f'/home/{user_name}/GoogleDrive/Research/GSI/Autopicker'
# wdir = f'/home/lewis/Documents/Research/Autopicker'
wdir = f'/mnt/c/Users/lewiss/Documents/Research/Autopicker'
print(f'Working directory: {wdir}\n')
mpl.rcParams['savefig.directory'] = f"{wdir}"
# data archive directory
adir = '/net/172.16.46.200/archive/jqdata/archive'
# input data subdirectory
idat = '01-06-2021_01-06-2022_M3'

# FDSN database
isn_client = Client('http://172.16.46.102:8181/')       # jfdsn
# isn_client = Client('http://172.16.46.140:8181/')       # jtfdsn

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

# retrieve station inventory
if path.exists(f"{wdir}/inventory.xml") != 0:
    print('ISN inventory file already exists:')
    os.system(f"ls -lh {wdir}/inventory.xml")
    print()
else:
    isn_inv = isn_client.get_stations(network=ntw, channel='ENZ, HHZ, BHZ', level='response')
    isn_inv.write(f'{wdir}/inventory.xml', level='response', format='STATIONXML')
# read station inventory
isn_inv = read_inventory(f"{wdir}/inventory.xml", format='STATIONXML')

# read M>3.0 events from FDSNws for period June 2021 - June 2022
etab = pd.read_csv(f"{wdir}/{idat}.csv", parse_dates=['OriginTime'])

########################################################################################################################
# TESTS WITH SPECTRA
if_spe = False
if if_spe:
    # initialising TauP
    model = TauPyModel(model=vel_mod)
    twin = 3.
    evt = '20220216032228454'
    eid = 'gsi202202160321'
    elat = 32.59742
    elon = 35.60145
    edep = 5.
    emag = 3.2
    # epic = '20210615230854375'
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
        # if path.exists(f'{wdir}/{idat}/{epic}/{tr1.stats.network}.{stn}.{chn}.png') != 0:
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
        # plt.savefig(f'{wdir}/{idat}/{epic}/{tr1.stats.network}.{stn}.{chn}.png', bbox_inches='tight', dpi='figure')
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
if_res = False
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
            if e != evt or apar[0] != rmhp[j] or apar[1] != taper[j] or apar[2] != bworder[j] or apar[3] != bwminf[j] \
                    or apar[4] != bwmaxf[j] or apar[5] != sta[j] or apar[6] != lta[j] or apar[7] != trigon[j] or apar[8] != trigoff[j]:
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
        # pdif = dict(hdif)
        # ff.suptitle('Comparison with catalogue picks', fontweight='bold')
        pdif = dict(tdif)
        ff.suptitle('Comparison with theoretical picks', fontweight='bold')
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
                ax.plot([ax.get_xlim()[0]-.5, ax.get_xlim()[0]-4.5], [ie+.7, ie+.7], color='red', clip_on=False)
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

########################################################################################################################
# FILE PROCESSING TO BUILD FIGURES
if_proc = False
if if_proc:
    # event experiments
    # twin = timedelta(minutes=5)
    # half a day-long experiments
    twin = timedelta(hours=5)
    # loop over events
    for _, erow in etab.iterrows():
        # define file extension based on time window length
        ext = ''
        if twin == timedelta(minutes=5):
            ext = '10m'
        elif twin == timedelta(hours=5):
            ext = '10h'
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        if evt == '20210617215634852' or evt == '20211027045305932':
            continue
        epar = {'evid': erow.EventID, 'elat': erow.Latitude, 'elon': erow.Longitude, 'edep': erow.Depth,
                'eori': datetime.strptime(str(erow.OriginTime).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), 'emag': erow.Magnitude}
        print(f"{evt} M{epar['emag']:3.1f}")

        #######################################################################################################################
        # RUN AUTOPICKER
        # output .xml file name
        if pic == 'AIC':
            oxml = f"{evt}_{exp}_AIC.{ext}.xml"
        else:
            oxml = f"{evt}_{exp}.{ext}.xml"
        # check the autopicker was run
        if path.exists(f"{wdir}/{idat}/{oxml}") != 0:
            print(f" Experiment #{exp} already ran:")
            os.system(f"ls -lh {wdir}/{idat}/{oxml}")
        else:
            # check the right configuration file is the current one
            if not filecmp.cmp(f'{wdir}/config_autop.xml', f'{wdir}/config_autop_{exp}.xml'):
                os.system(f'cp -p {wdir}/config_autop_{exp}.xml {wdir}/config_autop.xml')
            # autopicker command (using raw .mseed file)
            cmd = f"scautopick --ep --config-db {wdir}/config_autop.xml --inventory-db {wdir}/inventory_autop.xml" \
                  f" --playback -I file://{wdir}/{idat}/{evt}.{ext}.raw.mseed > {wdir}/{idat}/{oxml}"
            print(f" Running {ext} experiment #{exp}:")
            print(' ' + cmd)
            os.system(cmd)

        #######################################################################################################################
        # RETRIEVE WAVEFORM DATA
        if path.exists(f"{wdir}/{idat}/{evt}.{ext}.mseed") == 0:
            # .mseed file name
            mfile = get_traces_deci(evt, epar['eori'], twin)
            # read .mseed file
            isn_traces = read(mfile).merge()
            # add event data to waveforms
            isn_traces, rtab = add_event_data(isn_traces, epar, vel_mod, isn_inv)
            # remove trend from all traces
            isn_traces.detrend('spline', order=3, dspline=500)
            # sort according to newly added distance to event
            isn_traces.sort(['distance'], reverse=True)
            # write new .mseed file for processed data
            isn_traces.write(mfile.replace('.raw', ''))
            print(' Waveform data retrieved and processed:')
            os.system(f"ls -lh {wdir}/{idat}/{evt}.{ext}.mseed")
        else:
            print(' Waveform data already exists:')
            os.system(f"ls -lh {wdir}/{idat}/{evt}.{ext}.mseed")
        print()

########################################################################################################################
# FIGURES
if_plot = True
if if_plot:
    # event experiments
    # twin = timedelta(minutes=5)
    # half a day-long experiments
    twin = timedelta(hours=5)
    # loop over events
    for _, erow in etab.iterrows():
        # define file extension based on time window length
        ext = ''
        if twin == timedelta(minutes=5):
            ext = '10m'
        elif twin == timedelta(hours=5):
            ext = '10h'
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        # if epic != '20210615230854375' and epic != '20220122220957402':
        #     continue
        if evt == '20210617215634852' or evt == '20211027045305932':
            continue
        epar = {'evid': erow.EventID, 'elat': erow.Latitude, 'elon': erow.Longitude, 'edep': erow.Depth,
                'eori': datetime.strptime(str(erow.OriginTime).replace('+00:00', ''), '%Y-%m-%d %H:%M:%S.%f'), 'emag': erow.Magnitude}
        print(f"{evt} M{epar['emag']:3.1f}")
        # check whether figure and output file already exist or not
        if pic == 'AIC':
            oxml = f"{evt}_{exp}_AIC_{ext}.xml"
        else:
            oxml = f"{evt}_{exp}.{ext}.xml"
        # check if figures needs to be plotted
        if (ext == '10m' and path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}") != 0 and path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}") != 0) or\
                (ext == '10h' and path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}") != 0):
            print()
            continue
        #######################################################################################################################
        # RETRIEVE AUTOMATIC PICKS
        # load waveform data to plot (not .raw.mseed)
        isn_traces = read(f"{wdir}/{idat}/{evt}_{ext}.mseed")
        # read output file
        if pic == 'AIC':
            isn_traces, na = read_autopick_xml(f"{wdir}/{idat}/{oxml}", isn_traces, 'P', 'AIC')
        else:
            isn_traces, na = read_autopick_xml(f"{wdir}/{idat}/{oxml}", isn_traces, 'P', '')
        if na < 2:
            print(' Not enough automatic picks')
            print()
            continue
        else:
            print(f" {na} automatic picks")
        # create DataFrame with automatic picks for time window analysis
        atab = pd.DataFrame({'net': pd.Series(dtype='string'), 'stn': pd.Series(dtype='string'), 'loc': pd.Series(dtype='string'),
                             'chn': pd.Series(dtype='string'), 'pick': pd.Series(dtype='datetime64[ms]')})
        for t in isn_traces:
            if hasattr(t.stats, 'auto_tt'):
                for p in t.stats.auto_tt:
                    atab.loc[atab.shape[0]] = [t.stats.network, t.stats.station, t.stats.location, t.stats.channel, p]
        # initialise table to contain 30-s time windows with >6 picks
        wtab = []
        # look for 30-s time windows with >6 picks
        tper = pd.date_range(epar['eori']-twin, epar['eori']+twin, freq='30S').to_list()
        for i in range(len(tper)-1):
            tab = atab.loc[(atab.pick > tper[i]) & (atab.pick <= tper[i+1])]
            if not tab.empty and len(tab) > 5:
                wtab.append(tper[i])

        #######################################################################################################################
        # MORE DATA PROCESSING (FOR PLOTTING ONLY)
        for t in isn_traces:
            # # remove traces with no automatic picks
            # if hasattr(t.stats, 'auto_tt') and len(t.stats.auto_tt) < 1 or (not hasattr(t.stats, 'auto_tt')):
            #     isn_traces.remove(t)
            # remove Meiron array
            if (re.search('MMA', t.stats.station) and t.stats.station != 'MMA0B') or re.search('MMB', t.stats.station) or re.search('MMC', t.stats.station):
                isn_traces.remove(t)
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
        nc = 0
        if ext == '10m':
            isn_traces, nc = get_catalogue_picks(isn_client, epar['evid'], isn_traces)
            if nc < 1:
                print(' No hand picks')
            else:
                print(f" {nc} hand picks")

        #######################################################################################################################
        # BUILD OUTPUT FILE
        if ext == '10m' and path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}") == 0:
            # create table for TT residual data if needed
            _, rtab = add_event_data(isn_traces, epar, vel_mod, isn_inv)
            # write to file
            fout = open(f"{wdir}/{idat}/{oxml.replace('.xml', '.txt')}", 'w')
            # write to file
            if nc > 1:
                dt_cata = [(row.aarr-row.carr).total_seconds() for _, row in rtab.iterrows()]
                dt_theo = [(row.aarr-row.tarr).total_seconds() for _, row in rtab.iterrows()]
                fout.write(f"{epar['eori']} {epar['elat']:9.4f} {epar['elon']:9.4f} {epar['edep']:5.1f} {epar['emag']:4.2f} "
                           f"{fpar['rmhp']:.1f} {fpar['taper']:4.1f} {fpar['bworder']:1d} {fpar['bwminf']:4.1f} {fpar['bwmaxf']:4.1f} "
                           f"{fpar['sta']:4.1f} {fpar['lta']:4.1f} {fpar['trigon']:4.1f} {fpar['trigoff']:4.1f} {nc:3d} "
                           f"{np.nanmean(dt_cata):9.4f} {np.nanstd(dt_cata):9.4f} {np.nanmean(dt_theo):9.4f} {np.nanstd(dt_theo):9.4f}\n")
            elif nc == 1:
                dt_cata = [(row.aarr-row.carr).total_seconds() for _, row in rtab.iterrows()]
                dt_theo = [(row.aarr-row.tarr).total_seconds() for _, row in rtab.iterrows()]
                fout.write(f"{epar['eori']} {epar['elat']:9.4f} {epar['elon']:9.4f} {epar['edep']:5.1f} {epar['emag']:4.2f} "
                           f"{fpar['rmhp']:.1f} {fpar['taper']:4.1f} {fpar['bworder']:1d} {fpar['bwminf']:4.1f} {fpar['bwmaxf']:4.1f} "
                           f"{fpar['sta']:4.1f} {fpar['lta']:4.1f} {fpar['trigon']:4.1f} {fpar['trigoff']:4.1f} {nc:3d} "
                           f"{np.nanmean(dt_cata):9.4f} {np.nan:9.4f} {np.nanmean(dt_theo):9.4f} {np.nanstd(dt_theo):9.4f}\n")
            elif nc == 0:
                dt_theo = [(row.aarr-row.tarr).total_seconds() for _, row in rtab.iterrows()]
                fout.write(f"{epar['eori']} {epar['elat']:9.4f} {epar['elon']:9.4f} {epar['edep']:5.1f} {epar['emag']:4.2f} "
                           f"{fpar['rmhp']:.1f} {fpar['taper']:4.1f} {fpar['bworder']:1d} {fpar['bwminf']:4.1f} {fpar['bwmaxf']:4.1f} "
                           f"{fpar['sta']:4.1f} {fpar['lta']:4.1f} {fpar['trigon']:4.1f} {fpar['trigoff']:4.1f} {nc:3d} {np.nan:9.4f} {np.nan:9.4f} "
                           f"{np.nanmean(dt_theo):9.4f} {np.nanstd(dt_theo):9.4f}\n")
            for _, row in rtab.iterrows():
                fout.write(f"{row.net:2s} {row.stn:6s} {row.loc:2s} {row.chn:3s} {row.dis:9.4f} {datetime.strftime(row.carr, '%Y-%m-%d %H:%M:%S.%f'):s} "
                           f"{datetime.strftime(row.aarr, '%Y-%m-%d %H:%M:%S.%f'):s} {datetime.strftime(row.tarr, '%Y-%m-%d %H:%M:%S.%f'):s}\n")
            fout.close()
            rtab = None

        #######################################################################################################################
        # BUILD OUTPUT FIGURE
        # add event origin time to trace headers
        for tr in isn_traces:
            tr.stats['origin_time'] = epar['eori']
        # only if figure does not already exist
        if path.exists(f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}") == 0:
            print(f' {len(isn_traces)} waveforms to plot')
            plot_autopick_cont_sec(isn_traces, wtab, f"{wdir}/{idat}/{oxml.replace('.xml', '.png')}")
        isn_traces = None
        print()
