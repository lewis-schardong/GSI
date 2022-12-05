########################################################################################################################
import os
from os import path
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
import statistics
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
import xml.etree.ElementTree as ETree
from obspy import read, read_inventory, read_events
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
    # load miniSEED data
    mseed = f'{wdir}/{evt_id}.raw.mseed'
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
        # ensuring 50-Hz sampling rate for all traces (e.g. if time window crosses midnight and data is missing
        # for one of the two dates, we get non-decimated data as replacement)
        stream.resample(50.0, window='hann')
        # merging all traces (e.g. if time window crosses midnight, data from both dates need to be merged)
        stream.merge(fill_value='interpolate')
        # write miniSEED file
        stream.write(mseed)
    else:
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    return mseed


def process_mseed(mseed_in, sta_inv):
    """
    :param mseed_in: full path to miniSEED file to process
    :param sta_inv: station inventory
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
    stream.taper(max_percentage=.5, type='cosine', max_length=10., side='left')
    # apply high-pass filter to all traces
    stream.filter('highpass', freq=1./5.)
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply Butterworth band-pass filter to all traces
    stream.filter('bandpass', freqmin=4., freqmax=8., corners=4)
    # write miniSEED file
    stream.write(mseed_out)
    return mseed_out


def read_autopick_xml(xml_path, stream, phase='P'):
    """
    :param xml_path: path to .XML file containing picks to read
    :param stream: data streamer of waveforms to process
    :param phase: seismic phase of interest
    :return: data streamer containing automatic picks in trace headers and total number of picks found
    """
    npic = 0
    # read resulting xml file
    for line_xml in ETree.parse(xml_path).getroot()[0]:
        if re.search('pick', str(line_xml.tag)):
            # check picker
            if not re.search('AIC', str(line_xml.attrib)):
                continue
            # check phase
            if (line_xml[3].tag == 'phaseHint' and line_xml[3].text != phase) or\
                    (line_xml[4].tag == 'phaseHint' and line_xml[4].text != phase):
                continue
            # station metadata
            str1 = re.search("'publicID': '(.*?)'", str(line_xml.attrib)).group(1)
            str2 = str1.split('.')
            str3 = str2[2].split('-')
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
    event = client.get_events(eventid=evt_id, includearrivals=True)[0]
    for pik in event.picks:
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


def add_event_data(stream, evt_param, ref_mod, sta_inv):
    """
    :param stream: data streamer of waveforms to process
    :param evt_param: dictionary containing event parameters
    :param ref_mod: reference velocity model to use for theoretical travel times
    :param sta_inv: .xml inventory containing all station info.
    :return: data streamer with event info.
    """
    # initialising TauP
    theory = TauPyModel(model=ref_mod)
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
            return None
        # calculate event-station distance
        dis = gdist.distance((evt_param['elat'], evt_param['elon']), (station[0].stations[0].channels[0].latitude,
                                                                      station[0].stations[0].channels[0].longitude))
        stream[k].stats.distance = dis.m
        # compute theoretical travel time
        theo_tt = theory.get_travel_times(source_depth_in_km=evt_param['edep'],
                                          distance_in_degree=dis.km / (2 * np.pi * rrad / 360),
                                          phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        # add theoretical travel time to streamer header
        if len(theo_tt) != 0:
            stream[k].stats['theo_tt'] = evt_param['eori'] + timedelta(seconds=theo_tt[0].time)
        else:
            stream[k].stats['theo_tt'] = np.nan
        k += 1
    # delete selected waveforms
    for trace in to_del:
        try:
            stream.remove(trace)
        except:
            continue
    return stream


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
    fig = plt.figure(figsize=(9, 5), dpi=200)
    # set x-axis limits
    plt.gca().set_xlim(xlim)
    # show catalogue events
    evt_lst = isn_client.get_events(starttime=xlim[0], endtime=xlim[1])
    # explosions counter
    n_exp = 0
    for event in evt_lst:
        # origin time
        tori = datetime.strptime(str(event.preferred_origin().time), '%Y-%m-%dT%H:%M:%S.%fZ')
        # normalised x-axis coordinate (because of problems with datetimes and JSON)
        xpos = (tori-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
        if event.magnitudes:
            # plot marker
            plt.plot([tori, tori], [0, len(stream)+1], color='red', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.01, f"M{event.preferred_magnitude().mag:3.1f}",
                     color='red', fontsize=10, ha='center', va='bottom',
                     clip_on=False, transform=plt.gca().transAxes)
        else:
            n_exp += 1
            # plot marker
            plt.plot([tori, tori], [0, len(stream)+1], color='green', linewidth=2, alpha=.5)
            # display event type
            plt.text(xpos, 1.03, 'EXP', color='green', fontsize=10, ha='center', va='bottom',
                     clip_on=False, transform=plt.gca().transAxes)
    # show EMSC events
    tele_client = Client('EMSC')
    # M>5 teleseismic events catalogue
    evt_lst1 = []
    try:
        evt_lst1 = tele_client.get_events(starttime=xlim[0], endtime=xlim[1], minmagnitude=5)
    except:
        print(' No M>5 teleseismic events')
    if evt_lst1:
        for event in evt_lst1:
            # origin time
            tori = datetime.strptime(str(event.preferred_origin().time), '%Y-%m-%dT%H:%M:%S.%fZ')
            if abs((tori-stream[0].stats.origin_time).total_seconds()) < 60.:
                continue
            # normalised x-axis coordinate (because of problems with datetimes and JSON)
            xpos = (tori-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
            # plot marker
            plt.plot([tori, tori], [0, len(stream)+1], color='purple', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.07, f"M{event.preferred_magnitude().mag:3.1f}",
                     color='purple', fontsize=10, ha='center', va='bottom',
                     clip_on=False, transform=plt.gca().transAxes)
    # M>3 regional events catalogue
    evt_lst2 = []
    try:
        evt_lst2 = tele_client.get_events(starttime=xlim[0], endtime=xlim[1], minmagnitude=3,
                                          minlatitude=23, maxlatitude=43, minlongitude=20, maxlongitude=50)
    except:
        print('No M>3 regional events')
    if evt_lst2:
        for event in evt_lst2:
            # origin time
            tori = datetime.strptime(str(event.preferred_origin().time), '%Y-%m-%dT%H:%M:%S.%fZ')
            if abs((tori-stream[0].stats.origin_time).total_seconds()) < 60.:
                continue
            # normalised x-axis coordinate (because of problems with datetimes and JSON)
            xpos = (tori-xlim[0]).total_seconds() / (xlim[1]-xlim[0]).total_seconds()
            # plot marker
            plt.plot([tori, tori], [0, len(stream)+1], color='orange', linewidth=2, alpha=.5)
            # display event type & magnitude
            plt.text(xpos, 1.05, f"M{event.preferred_magnitude().mag:3.1f}",
                     color='orange', fontsize=10, ha='center', va='bottom',
                     clip_on=False, transform=plt.gca().transAxes)
    # display total number of stations
    plt.text(-.01, 1., f"Ns={len(stream)}", ha='right', va='bottom',
             fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
    # display total number of local events
    plt.text(1.01, 1.01, f"Nl={len(evt_lst)-n_exp}", ha='left', va='bottom', color='red',
             fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
    # display total number of blasts
    plt.text(1.01, 1.03, f"Nx={n_exp}", ha='left', va='bottom', color='green', fontsize=10,
             fontweight='bold', transform=plt.gca().transAxes)
    # display total number of regional events
    plt.text(1.01, 1.05, f"Nr={len(evt_lst2)}", ha='left', va='bottom', color='orange',
             fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
    # display total number of teleseismic events
    plt.text(1.01, 1.07, f"Nt={len(evt_lst1)}", ha='left', va='bottom', color='purple',
             fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
    # initialise counter
    n_trace = 0
    # loop over stream channels
    for trace in stream:
        n_trace += 1
        # plot waveforms
        t_vec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]')\
            + np.datetime64(str(trace.stats.starttime)[:-1])
        plt.plot(t_vec, trace.data/trace.max() + n_trace, color='grey', alpha=.7, label='Velocity')
        if hasattr(trace.stats, 'auto_tt'):
            # number of picks per station
            if len(trace.stats.auto_tt) > 10:
                # show number of picks
                plt.text(1.01, n_trace/(len(stream)+1), f"Np={len(trace.stats.auto_tt)}",
                         color='red', fontsize=10, clip_on=False, ha='left', va='center', transform=plt.gca().transAxes)
                # show station/channel name
                plt.text(-.01, n_trace/(len(stream)+1),
                         f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                         color='red', fontsize=10, clip_on=False,
                         ha='right', va='center', transform=plt.gca().transAxes)
            else:
                # show number of picks
                plt.text(1.01, n_trace/(len(stream)+1), f"Np={len(trace.stats.auto_tt)}",
                         color='blue', fontsize=10, clip_on=False,
                         ha='left', va='center', transform=plt.gca().transAxes)
                # show station/channel name
                plt.text(-.01, n_trace/(len(stream)+1),
                         f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                         color='blue', fontsize=10, clip_on=False,
                         ha='right', va='center', transform=plt.gca().transAxes)
            for at in trace.stats.auto_tt:
                plt.plot([at, at], [n_trace-.5, n_trace+.5], color='blue', label='Autopick')
    # 30-s time windows with >6 picks
    for win in win_tab:
        plt.gca().fill_betweenx([0, n_trace+1], win-timedelta(minutes=2.5),
                                win+timedelta(minutes=2.5), color='red', alpha=.5)
    # set y-axis limits
    plt.gca().set_ylim([0, n_trace+1])
    # remove y-axis tick labels
    plt.gca().set_yticklabels([])
    # set x-axis font size
    plt.gca().tick_params(axis='x', which='major', labelsize=10)
    # save figure
    fid = open(fig_name, 'w')
    mpld3.save_html(fig, fid)
    print(f" Figure saved: {fig_name}")
    plt.close()
    fid.close()
    return


def plot_autopick_evt_sec(stream, evt_param, fig_name=None):
    """
    :param stream: data streamer of waveforms to process
    :param evt_param: dictionary containing event parameters
    :param fig_name: figure file name (figure is shown if None)
    :return: data streamer with event info.
    """
    # event info. from playback associator
    event = read_events(f"{wdir}/{oxml.replace('_picks', '_events')}")
    print(f' {len(event)} event(s) were found')
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
    fid = open('/home/sysop/.seiscomp/bna/ActiveFaults/activefaults.bna', 'r')
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
    fid = open('/home/sysop/.seiscomp/bna/Quarries/quarries.bna', 'r')
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
    for trace in stream:
        # station label for y-axis
        stn_lbl.append(f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}")
        # counter
        n_trace += 1
        # time vector
        t_vec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]')\
            + np.datetime64(str(trace.stats.starttime)[:-1])
        # plot waveform
        h1, = axis1.plot(t_vec, trace.data / trace.max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # plot theoretical travel time
        if hasattr(trace.stats, 'theo_tt') and trace.stats.theo_tt:
            h2, = axis1.plot([trace.stats.theo_tt, trace.stats.theo_tt], [n_trace - 1, n_trace + 1],
                             color='blue', linestyle='dotted', label=vel_mod)
        # plot station in map
        st = isn_inv.select(network=trace.stats.network, station=trace.stats.station, channel=trace.stats.channel)
        if not st:
            print(f'Missing station: {stn_lbl[n_trace-1]}')
        # use different symbols and colours for different network codes
        hm1, = axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                          '^', mfc='blue', mec='none', markersize=5, alpha=.7, label=trace.stats.network)
        # AUTOMATIC PICKS
        # initialise index for pick of interest
        k_apic = None
        # make sure station has picks
        if hasattr(trace.stats, 'auto_tt') and trace.stats.auto_tt:
            # time difference between automatic picks and theoretical arrival
            tdif = [abs((xx - trace.stats.theo_tt).total_seconds()) for xx in trace.stats.auto_tt]
            # initialise counter
            n_pic = 0
            # loop over all picks for channel
            for pick in trace.stats.auto_tt:
                h4, = axis1.plot([pick, pick], [n_trace - 1, n_trace + 1], color='purple', label='Automatic')
                # right-hand side marker to quickly know which station has picks
                axis1.plot(1.01, n_trace/(len(stream)+1), 'o', markersize=5, mfc='purple', mec='none',
                           alpha=.7, clip_on=False, transform=axis1.transAxes)
                # highlight station in map
                axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                           'o', markersize=7, color='purple', mfc='none', alpha=.7)
                # find pick >OT and <[tlim]s difference with theoretical arrival
                if pick > evt_param['eori'] and n_pic == tdif.index(min(tdif)) and tdif[n_pic] < tlim:
                    k_apic = n_pic
                # counter
                n_pic += 1
        # CATALOGUE PICKS
        # make sure station has catalogue picks
        if hasattr(trace.stats, 'cata_tt') and trace.stats.cata_tt:
            # plot pick on top of waveform
            h3, = axis1.plot([trace.stats.cata_tt[0], trace.stats.cata_tt[0]], [n_trace - 1, n_trace + 1],
                             color='orange', label='Catalogue')
            # right-hand side marker to quickly know which station has picks
            axis1.plot(1.02, n_trace/(len(stream)+1), 'o', markersize=5, mfc='orange', mec='none',
                       alpha=.7, clip_on=False, transform=axis1.transAxes)
            # highlight station in map
            axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                       's', markersize=7, mec='orange', mfc='none', alpha=.7)
        # residual w.r.t. catalogue pick (if both exist)
        if hasattr(trace.stats, 'cata_tt') and k_apic is not None:
            # table for residuals (for statistics)
            tres_tab.append((trace.stats.auto_tt[k_apic] - trace.stats.cata_tt[0]).total_seconds())
            # residual plot
            axis2.plot((trace.stats.auto_tt[k_apic] - trace.stats.cata_tt[0]).total_seconds(),
                       n_trace, 'o', markersize=5, mfc='orange', mec='none', alpha=.7)
        # residual w.r.t. theoretical pick (if automatic pick exists)
        if k_apic is not None:
            # table for residuals (for statistics)
            theo_tab.append((trace.stats.auto_tt[k_apic] - trace.stats.theo_tt).total_seconds())
            # residual plot
            axis2.plot((trace.stats.auto_tt[k_apic] - trace.stats.theo_tt).total_seconds(), n_trace, 'o', markersize=5, mfc='blue', mec='none', alpha=.7)
    # display statistics on residuals
    if len(tres_tab) > 1:
        axis2.text(.01, .03, f"N={len(tres_tab)}: {statistics.mean(tres_tab):.2f} \u00B1"
                             f" {statistics.stdev(tres_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='orange', transform=axis2.transAxes)
    if len(theo_tab) > 1:
        axis2.text(.01, .01, f"N={len(theo_tab)}: {statistics.mean(theo_tab):.2f} \u00B1"
                             f" {statistics.stdev(theo_tab):.2f} [s]",
                   fontweight='bold', fontsize=8, color='blue', transform=axis2.transAxes)
    axis2.text(.01, .05, f"N={event[0].preferred_origin().quality.used_phase_count}:"
                         f" {event[0].preferred_origin().quality.standard_error:.2f} [s]",
               fontweight='bold', fontsize=8, color='purple', transform=axis2.transAxes)
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
    tit1 = f"{evt_param['eori'].strftime('%d/%m/%Y %H:%M:%S.%f')} \u2013 " \
           f"{event[0].preferred_origin().time.strftime('%d/%m/%Y %H:%M:%S.%f')} " \
           f"({event[0].preferred_origin().time_errors.uncertainty:.2f} s)"
    tit2 = f"[{evt_param['elat']:.2f},{evt_param['elon']:.2f}] \u2013 [{event[0].preferred_origin().latitude:.2f}," \
           f"{event[0].preferred_origin().longitude:.2f}] ([{event[0].preferred_origin().latitude_errors.uncertainty:.2f}," \
           f"{event[0].preferred_origin().longitude_errors.uncertainty:.2f}] km)"
    tit3 = f"M{evt_param['emag']:.2f} \u2013 M{event[0].preferred_magnitude().mag:.2f}"
    tit4 = f"{evt_param['edep']:.2f} \u2013 {event[0].preferred_origin().depth/1000.:.2f} ({event[0].preferred_origin().depth_errors.uncertainty/1000.:.2f} km)"
    # # figure title
    # tit1 = f"{evt_param['eori'].strftime('%d/%m/%Y %H:%M:%S')}" \
    #        f" \u2013 {evt_param['edep']:.2f} km \u2013 M{evt_param['emag']:3.1f}"
    # tit2 = f"HP: {filt_param['rmhp']:.2f} [s] \u2013 Taper: {filt_param['taper']:.2f} [s] \u2013 BP: " \
    #        f"{filt_param['bworder']} / {filt_param['bwminf']:.2f} / {filt_param['bwmaxf']:.2f} [Hz]"
    # tit3 = f"STA/LTA: {filt_param['sta']:.2f} / {filt_param['lta']:.2f} [s] " \
    #        f"\u2013 Trigger: {filt_param['trigon']:.2f} / {filt_param['trigoff']:.2f}"
    fig.suptitle(tit1 + '\n' + tit2 + '\n' + tit3 + '\n' + tit4, fontweight='bold')
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
    return
    # return tres_tab, theo_tab


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
# wdir = f'/home/lewis/autopicker-pb'
wdir = f'/mnt/c/Users/lewiss/Documents/Research/Autopicker/autopicker-pb'
print(f'Working directory: {wdir}\n')
mpl.rcParams['savefig.directory'] = f"{wdir}"
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
    isn_inv = isn_client.get_stations(network=ntw, channel='ENZ, HHZ, BHZ', level='response')
    isn_inv.write(f'{wdir}/inventory.xml', level='response', format='STATIONXML')
# read station inventory
isn_inv = read_inventory(f"{wdir}/inventory.xml", format='STATIONXML')

# read M>3.0 events from FDSNws for period June 2021 - June 2022
etab = pd.read_csv(f"{wdir}/01-06-2021_01-06-2022_M3.csv", parse_dates=['OriginTime'])
# # half a day-long experiments
# twin = timedelta(hours=5)
# event experiments
twin = timedelta(minutes=5)

########################################################################################################################
# FILE PROCESSING TO BUILD FIGURES
if_proc = False
if if_proc:
    # loop over events
    for _, erow in etab.iterrows():
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                                  '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        if evt == '20210617215634852' or evt == '20211027045305932':
            continue
        epar = {'evid': erow.EventID, 'elat': erow.Latitude, 'elon': erow.Longitude, 'edep': erow.Depth,
                'eori': datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                          '%Y-%m-%d %H:%M:%S.%f'), 'emag': erow.Magnitude}
        print(f"{evt} M{epar['emag']:3.1f}")

        ################################################################################################################
        # RETRIEVE WAVEFORM DATA
        if path.exists(f"{wdir}/{evt}.mseed") == 0:
            # .mseed file name
            mfile = get_traces_deci(evt, epar['eori'], twin)
            # read .mseed file
            isn_traces = read(mfile).merge()
            # remove trend from all traces
            isn_traces.detrend('spline', order=3, dspline=500)
            # write new .mseed file for processed data
            isn_traces.write(mfile.replace('.raw', ''))
            print(' Waveform data retrieved and processed:')
            os.system(f"ls -lh {wdir}/{evt}.mseed")
        else:
            print(' Waveform data already exists:')
            os.system(f"ls -lh {wdir}/{evt}.mseed")

        ################################################################################################################
        # RUN AUTOPICKER
        # output .xml file name
        oxml = f"{evt}_picks.xml"
        # check the autopicker was run
        if path.exists(f"{wdir}/{oxml.replace('picks', 'events')}") != 0:
            print(f" Playback already ran for {evt}:")
            os.system(f"ls -lh {wdir}/{oxml.replace('picks', 'events')}")
        else:
            # autopicker command (using raw .mseed file)
            print(f" Running playback for {evt}")
            os.system(f'{wdir}/playback.sh {evt}')
        print()

########################################################################################################################
# FIGURES
if_plot = True
if if_plot:
    # loop over events
    for _, erow in etab.iterrows():
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                                  '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        if evt == '20210617215634852' or evt == '20211027045305932':
            continue
        epar = {'evid': erow.EventID, 'elat': erow.Latitude, 'elon': erow.Longitude, 'edep': erow.Depth,
                'eori': datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                          '%Y-%m-%d %H:%M:%S.%f'), 'emag': erow.Magnitude}
        print(f"{evt} M{epar['emag']:3.1f}")
        # check whether figure and output file already exist or not
        oxml = f"{evt}_picks.xml"
        if path.exists(f"{wdir}/{oxml.replace('_picks.xml', '.png')}") != 0:
            print()
            continue

        ################################################################################################################
        # RETRIEVE AUTOMATIC PICKS
        # load waveform data to plot (not .raw.mseed)
        isn_traces = read(f"{wdir}/{evt}.mseed")
        # add event data to waveforms
        isn_traces = add_event_data(isn_traces, epar, vel_mod, isn_inv)
        # sort according to newly added distance to event
        isn_traces.sort(['distance'], reverse=True)
        # read output file
        isn_traces, na = read_autopick_xml(f"{wdir}/{oxml}", isn_traces, 'P')
        if na < 2:
            print(' Not enough automatic picks')
            print()
            continue
        else:
            print(f" {na} automatic picks")
        # # create DataFrame with automatic picks for time window analysis
        # atab = pd.DataFrame({'net': pd.Series(dtype='string'), 'stn': pd.Series(dtype='string'),
        #                      'loc': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string'),
        #                      'pick': pd.Series(dtype='datetime64[ms]')})
        # for t in isn_traces:
        #     if hasattr(t.stats, 'auto_tt'):
        #         for p in t.stats.auto_tt:
        #             atab.loc[atab.shape[0]] = [t.stats.network, t.stats.station, t.stats.location, t.stats.channel, p]
        # # initialise table to contain 30-s time windows with >6 picks
        # wtab = []
        # # look for 30-s time windows with >6 picks
        # tper = pd.date_range(epar['eori']-twin, epar['eori']+twin, freq='30S').to_list()
        # for i in range(len(tper)-1):
        #     tab = atab.loc[(atab.pick > tper[i]) & (atab.pick <= tper[i+1])]
        #     if not tab.empty and len(tab) > 5:
        #         wtab.append(tper[i])

        ################################################################################################################
        # RETRIEVE CATALOGUE PICKS
        isn_traces, nc = get_catalogue_picks(isn_client, epar['evid'], isn_traces)
        print(f" {nc} catalogue picks")

        ################################################################################################################
        # MORE DATA PROCESSING (FOR PLOTTING ONLY)
        # remove traces with <2 automatic picks (assuming first pick is for event)
        for t in isn_traces:
            if hasattr(t.stats, 'auto_tt') and len(t.stats.auto_tt) < 2 or (not hasattr(t.stats, 'auto_tt')):
                isn_traces.remove(t)
        # apply taper to all traces
        isn_traces.taper(max_percentage=.5, type='cosine', max_length=fpar['rmhp'], side='left')
        # apply high-pass filter to all traces
        isn_traces.filter('highpass', freq=1./fpar['taper'])
        # apply Butterworth band-pass filter to all traces
        isn_traces.filter('bandpass', freqmin=fpar['bwminf'], freqmax=fpar['bwmaxf'], corners=fpar['bworder'])
        # # downsample data (to avoid memory issues when plotting)
        # isn_traces.resample(1.0, window='hann')

        ################################################################################################################
        # BUILD OUTPUT FIGURE
        # # add event origin time to trace headers
        # for tr in isn_traces:
        #     tr.stats['origin_time'] = epar['eori']
        # only if figure does not already exist
        if path.exists(f"{wdir}/{oxml.replace('_picks.xml', '.png')}") == 0:
            print(f' {len(isn_traces)} waveforms to plot')
            # plot_autopick_cont_sec(isn_traces, wtab, f"{wdir}/{oxml.replace('_picks.xml', '.html')}")
            plot_autopick_evt_sec(isn_traces, epar, f"{wdir}/{oxml.replace('_picks.xml', '.png')}")
        isn_traces = None
        print()
        exit()
    exit()
