########################################################################################################################
import os
from os import path
import glob
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
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


def read_autopick_xml(xml_path, stream):
    """
    :param xml_path: path to .XML file containing picks to read
    :param stream: data streamer of waveforms to process
    :return: data streamer containing automatic picks in trace headers and total number of picks found
    """
    # initialise counter
    npic = 0
    # loop over .xml file lines
    for line_xml in ETree.parse(xml_path).getroot()[0]:
        # search pick sections
        if re.search('pick', str(line_xml.tag)):
            if len(line_xml) == 8 and line_xml[4].text == 'S':
                channel = line_xml[1].attrib['channelCode'].replace('HHN', 'HHZ').replace('HHE', 'HHZ')\
                    .replace('ENN', 'ENZ').replace('ENE', 'ENZ').replace('BHN', 'BHZ').replace('BHE', 'BHZ')
            else:
                channel = line_xml[1].attrib['channelCode']
            # search for trace in streamer corresponding to pick
            trace = stream.select(network=line_xml[1].attrib['networkCode'],
                                  station=line_xml[1].attrib['stationCode'],
                                  channel=channel)
            # ignore pick if trace not in streamer
            if not trace:
                continue
            else:
                trace = trace[0]
            if len(line_xml) == 6 and line_xml[3].text == 'P':
                # initialise pick table if needed
                if not hasattr(trace.stats, 'auto_ttp'):
                    trace.stats['auto_ttp'] = []
                trace.stats.auto_ttp.append(datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ'))
            if len(line_xml) == 8 and line_xml[4].text == 'S':
                # initialise pick table if needed
                if not hasattr(trace.stats, 'auto_tts'):
                    trace.stats['auto_tts'] = []
                trace.stats.auto_tts.append(datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ'))
            npic += 1
    return stream, npic


def get_catalogue_picks(client, evt_id, stream):
    """
    :param client: Obspy FDSN client to retrieve data from
    :param evt_id: event ID according to the GSI's FDSN database
    :param stream: data streamer of waveforms to process
    :return: data streamer containing catalogue picks in trace headers and total number of picks found
    """
    # initialise counter
    n_pick = 0
    # retrieve event data
    try:
        event = client.get_events(eventid=evt_id, includearrivals=True)
    except:
        print(f'Event ID missing from catalogue: {evt_id}')
        return stream, 0
    # select first and only (?) event
    if event:
        event = event[0]
    else:
        return stream, 0
    # loop over traces
    for pick in event.picks:
        # select for trace in streamer corresponding to pick metadata
        trace = stream.select(network=pick.waveform_id.network_code, station=pick.waveform_id.station_code,
                              channel=pick.waveform_id.channel_code)
        # ignore pick if trace not in streamer
        if not trace:
            continue
        else:
            trace = trace[0]
        if pick.phase_hint == 'P':
            # initialise pick table if needed
            if not hasattr(trace.stats, 'cata_ttp'):
                trace.stats['cata_ttp'] = []
            trace.stats.cata_ttp.append(pick.time.datetime)
        if pick.phase_hint == 'S':
            # initialise pick table if needed
            if not hasattr(trace.stats, 'cata_tts'):
                trace.stats['cata_tts'] = []
            trace.stats.cata_tts.append(pick.time.datetime)
        n_pick += 1
    return stream, n_pick


def add_event_data(stream, event_cata, ref_mod, sta_inv):
    """
    :param stream: data streamer of waveforms to process
    :param event_cata: catalogue event entry
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
        dis = gdist.distance((event_cata.preferred_origin().latitude, event_cata.preferred_origin().longitude),
                             (station[0].stations[0].channels[0].latitude,
                              station[0].stations[0].channels[0].longitude))
        stream[k].stats.distance = dis.m
        # compute theoretical travel time
        theo_ttp = theory.get_travel_times(source_depth_in_km=event_cata.preferred_origin().depth/1000.,
                                           distance_in_degree=dis.km / (2 * np.pi * rrad / 360),
                                           phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        # add theoretical travel time to streamer header
        if len(theo_ttp) != 0:
            stream[k].stats['theo_ttp'] = [event_cata.preferred_origin().time.datetime +
                                           timedelta(seconds=theo_ttp[0].time)]
        else:
            stream[k].stats['theo_ttp'] = [np.nan]
        theo_tts = theory.get_travel_times(source_depth_in_km=event_cata.preferred_origin().depth/1000.,
                                           distance_in_degree=dis.km / (2 * np.pi * rrad / 360),
                                           phase_list=['s', 'S', 'Sg', 'Sn', 'Sdiff'])
        # add theoretical travel time to streamer header
        if len(theo_tts) != 0:
            stream[k].stats['theo_tts'] = [event_cata.preferred_origin().time.datetime +
                                           timedelta(seconds=theo_tts[0].time)]
        else:
            stream[k].stats['theo_tts'] = [np.nan]
        k += 1
    # delete selected waveforms
    for trace in to_del:
        try:
            stream.remove(trace)
        except:
            continue
    return stream


def plot_autopick_evt_sec(stream, event_cata, fig_name=None):
    """
    :param stream: data streamer of waveforms to process
    :param event_cata: catalogue event entry
    :param fig_name: figure file name (figure is shown if None)
    :return: data streamer with event info.
    """
    # event info. from playback associator
    events = read_events(f"{wdir}/{oxml.replace('_picks', '_events')}")
    print(f' {len(events)} event(s) found')
    event_auto = []
    if events:
        if len(events) > 1:
            tdif = [abs(event_cata.preferred_origin().time.datetime -
                        e.preferred_origin().time.datetime).total_seconds() for e in events]
            event_auto = events[tdif.index(min(tdif))]
            print(f"  Min. time difference: {min(tdif)} s")
        else:
            event_auto = events[0]
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
    # axis1.set_xlim([event_cata.preferred_origin().time.datetime - twin,
    #                 event_cata.preferred_origin().time.datetime + twin])
    axis1.set_xlim([event_cata.preferred_origin().time.datetime - timedelta(minutes=0.5),
                    event_cata.preferred_origin().time.datetime + timedelta(minutes=2.5)])
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
    file_id = open(f"/home/{os.environ['LOGNAME']}/.seiscomp/bna/ActiveFaults/activefaults.bna", 'r')
    flts = file_id.readlines()
    file_id.close()
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
    file_id = open(f"/home/{os.environ['LOGNAME']}/.seiscomp/bna/Quarries/quarries.bna", 'r')
    # fid = open('/home/sysop/.seiscomp/bna/Quarries/quarries.bna', 'r')
    flts = file_id.readlines()
    file_id.close()
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
    pres_tab = []
    sres_tab = []
    ptheo_tab = []
    stheo_tab = []
    # initialise tables for legend
    hw = []
    hthp = []
    hcp = []
    hcs = []
    hta = []
    hap = []
    has = []
    # initialise tables for map legend
    hs = []
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
        hw, = axis1.plot(t_vec, trace.data / trace.max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # THEORY P
        if hasattr(trace.stats, 'theo_ttp') and trace.stats.theo_ttp:
            hthp, = axis1.plot([trace.stats.theo_ttp[0], trace.stats.theo_ttp[0]], [n_trace - 1, n_trace + 1],
                               color='blue', linestyle='dotted', label=vel_mod + ' P/S')
        # THEORY S
        if hasattr(trace.stats, 'theo_tts') and trace.stats.theo_tts:
            axis1.plot([trace.stats.theo_tts[0], trace.stats.theo_tts[0]], [n_trace - 1, n_trace + 1],
                       color='cyan', linestyle='dotted')
        # plot station in map
        st = isn_inv.select(network=trace.stats.network, station=trace.stats.station, channel=trace.stats.channel)
        if not st:
            print(f'Missing station: {stn_lbl[n_trace-1]}')
        hs, = axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                         '^', mfc='blue', mec='none', markersize=5, alpha=.7, label=trace.stats.network)
        # AUTOMATIC P
        # initialise index for pick of interest
        k_apicp = None
        # make sure station has automatic picks
        if hasattr(trace.stats, 'auto_ttp') and trace.stats.auto_ttp:
            # time difference between automatic picks and theoretical arrival
            tdif = [abs((xx - trace.stats.theo_ttp[0]).total_seconds()) for xx in trace.stats.auto_ttp]
            # initialise counter
            n_pic = 0
            # loop over all picks for channel
            for pick in trace.stats.auto_ttp:
                pid = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"
                fnd = False
                # check if pick listed as arrival (i.e. used in event location)
                if events:
                    for a in event_auto.preferred_origin().arrivals:
                        if re.search(pid, str(a.pick_id)):
                            xid = str(event_auto.preferred_origin().resource_id)\
                                .replace('smi:org.gfz-potsdam.de/geofon/', '')
                            rid = str(a.resource_id).replace(f"_{xid}", '')
                            for p in event_auto.picks:
                                if rid == p.resource_id and pick == p.time.datetime:
                                    fnd = True
                                    break
                            if fnd:
                                break
                if fnd:
                    # show picks selected for location
                    hta, = axis1.plot([pick, pick], [n_trace - 1, n_trace + 1], color='red', label='Arrival P/S')
                    # right-hand side marker to quickly know which station has picks
                    axis1.plot(1.01, n_trace/(len(stream)+1), 'o', markersize=5, mfc='none', mec='red',
                               alpha=.7, clip_on=False, transform=axis1.transAxes)
                else:
                    # show picks NOT selected for location
                    hap, = axis1.plot([pick, pick], [n_trace - 1, n_trace + 1], color='purple', label='Automatic P')
                # right-hand side marker to quickly know which station has picks
                axis1.plot(1.01, n_trace/(len(stream)+1), 'o', markersize=5, mfc='purple', mec='none',
                           alpha=.7, clip_on=False, transform=axis1.transAxes)
                # highlight station in map
                axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                           'o', markersize=7, mfc='none', mec='purple', alpha=.7)
                # find pick >OT and <[tlim]s difference with theoretical arrival
                if pick > event_cata.preferred_origin().time.datetime and \
                        n_pic == tdif.index(min(tdif)) and tdif[n_pic] < tlim:
                    k_apicp = n_pic
                # counter
                n_pic += 1
        # AUTOMATIC S
        # initialise index for pick of interest
        k_apics = None
        # make sure station has automatic picks
        if hasattr(trace.stats, 'auto_tts') and trace.stats.auto_tts:
            # time difference between automatic picks and theoretical arrival
            tdif = [abs((xx - trace.stats.theo_tts[0]).total_seconds()) for xx in trace.stats.auto_tts]
            # initialise counter
            n_pic = 0
            # loop over all picks for channel
            for pick in trace.stats.auto_tts:
                pid = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"
                fnd = False
                # check if pick listed as arrival (i.e. used in event location)
                if events:
                    for a in event_auto.preferred_origin().arrivals:
                        if re.search(pid, str(a.pick_id)):
                            xid = str(event_auto.preferred_origin().resource_id)\
                                .replace('smi:org.gfz-potsdam.de/geofon/', '')
                            rid = str(a.resource_id).replace(f"_{xid}", '')
                            for p in event_auto.picks:
                                if rid == p.resource_id and pick == p.time.datetime:
                                    fnd = True
                                    break
                            if fnd:
                                break
                if fnd:
                    # show picks selected for location
                    axis1.plot([pick, pick], [n_trace - 1, n_trace + 1], color='red')
                    # right-hand side marker to quickly know which station has picks
                    axis1.plot(1.02, n_trace/(len(stream)+1), 'o', markersize=5, mfc='none', mec='red',
                               alpha=.7, clip_on=False, transform=axis1.transAxes)
                else:
                    # show picks NOT selected for location
                    has, = axis1.plot([pick, pick], [n_trace - 1, n_trace + 1], color='magenta', label='Automatic S')
                # right-hand side marker to quickly know which station has picks
                axis1.plot(1.02, n_trace/(len(stream)+1), 'o', markersize=5, mfc='magenta', mec='none',
                           alpha=.7, clip_on=False, transform=axis1.transAxes)
                # highlight station in map
                axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                           'd', markersize=7, color='magenta', mfc='none', alpha=.7)
                # find pick >OT and <[tlim]s difference with theoretical arrival
                if pick > event_cata.preferred_origin().time.datetime and \
                        n_pic == tdif.index(min(tdif)) and tdif[n_pic] < tlim:
                    k_apics = n_pic
                # counter
                n_pic += 1
        # CATALOGUE P
        # make sure station has catalogue P picks
        if hasattr(trace.stats, 'cata_ttp') and trace.stats.cata_ttp:
            # plot pick on top of waveform
            hcp, = axis1.plot([trace.stats.cata_ttp[0], trace.stats.cata_ttp[0]], [n_trace - 1, n_trace + 1],
                              color='orange', label='Catalogue P')
            # right-hand side marker to quickly know which station has picks
            axis1.plot(1.03, n_trace/(len(stream)+1), 'o', markersize=5, mfc='orange', mec='none',
                       alpha=.7, clip_on=False, transform=axis1.transAxes)
            # highlight station in map
            axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                       's', markersize=7, mec='orange', mfc='none', alpha=.7)
        # CATALOGUE S
        # make sure station has catalogue S picks
        if hasattr(trace.stats, 'cata_tts') and trace.stats.cata_tts:
            # plot pick on top of waveform
            hcs, = axis1.plot([trace.stats.cata_tts[0], trace.stats.cata_tts[0]], [n_trace - 1, n_trace + 1],
                              color='yellow', label='Catalogue S')
            # right-hand side marker to quickly know which station has picks
            axis1.plot(1.04, n_trace/(len(stream)+1), 'o', markersize=5, mfc='yellow', mec='none',
                       alpha=.7, clip_on=False, transform=axis1.transAxes)
            # highlight station in map
            axis3.plot(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                       '^', markersize=7, mec='yellow', mfc='none', alpha=.7)
        # RESIDUALS
        # residual w.r.t. catalogue pick (if both exist)
        if hasattr(trace.stats, 'cata_ttp') and k_apicp is not None:
            # table for residuals (for statistics)
            pres_tab.append((trace.stats.auto_ttp[k_apicp] - trace.stats.cata_ttp[0]).total_seconds())
            # residual plot
            axis2.plot((trace.stats.auto_ttp[k_apicp] - trace.stats.cata_ttp[0]).total_seconds(),
                       n_trace, 'o', markersize=5, mfc='orange', mec='none', alpha=.7)
        # residual w.r.t. theoretical pick (if automatic pick exists)
        if k_apicp is not None:
            # table for residuals (for statistics)
            ptheo_tab.append((trace.stats.auto_ttp[k_apicp] - trace.stats.theo_ttp[0]).total_seconds())
            # residual plot
            axis2.plot((trace.stats.auto_ttp[k_apicp] - trace.stats.theo_ttp[0]).total_seconds(), n_trace,
                       'o', markersize=5, mfc='blue', mec='none', alpha=.7)
        # residual w.r.t. catalogue pick (if both exist)
        if hasattr(trace.stats, 'cata_tts') and k_apics is not None:
            # table for residuals (for statistics)
            sres_tab.append((trace.stats.auto_tts[k_apics] - trace.stats.cata_tts[0]).total_seconds())
            # residual plot
            axis2.plot((trace.stats.auto_tts[k_apics] - trace.stats.cata_tts[0]).total_seconds(),
                       n_trace, '^', markersize=5, mfc='orange', mec='none', alpha=.7)
        # residual w.r.t. theoretical pick (if automatic pick exists)
        if k_apics is not None:
            # table for residuals (for statistics)
            stheo_tab.append((trace.stats.auto_tts[k_apics] - trace.stats.theo_tts[0]).total_seconds())
            # residual plot
            print((trace.stats.auto_tts[k_apics] - trace.stats.theo_tts[0]).total_seconds())
            axis2.plot((trace.stats.auto_tts[k_apics] - trace.stats.theo_tts[0]).total_seconds(), n_trace,
                       '^', markersize=5, mfc='blue', mec='none', alpha=.7)
    # display catalogue residuals RMS
    if len(pres_tab) > 1:
        axis2.text(.01, .03, f"Np={len(pres_tab)}: {np.sqrt(np.mean(np.array(pres_tab)**2)):.2f} s",
                   fontweight='bold', fontsize=8, color='orange', transform=axis2.transAxes)
    # display theoretical residuals RMS
    if len(ptheo_tab) > 1:
        axis2.text(.01, .01, f"Np={len(ptheo_tab)}: {np.sqrt(np.mean(np.array(ptheo_tab)**2)):.2f} s",
                   fontweight='bold', fontsize=8, color='blue', transform=axis2.transAxes)
    if len(sres_tab) > 1:
        axis2.text(.01, .09, f"Ns={len(sres_tab)}: {np.sqrt(np.mean(np.array(sres_tab)**2)):.2f} s",
                   fontweight='bold', fontsize=8, color='yellow', transform=axis2.transAxes)
    # display theoretical residuals RMS
    if len(stheo_tab) > 1:
        axis2.text(.01, .07, f"Ns={len(stheo_tab)}: {np.sqrt(np.mean(np.array(stheo_tab)**2)):.2f} s",
                   fontweight='bold', fontsize=8, color='cyan', transform=axis2.transAxes)
    # display catalogue origin time
    hco = []
    if event_cata:
        hco, = axis1.plot([event_cata.preferred_origin().time.datetime, event_cata.preferred_origin().time.datetime],
                          [0, n_trace+1], color='lime', label='Catalogue OT')
    # display autoloc origin times
    hao = []
    if events:
        for e in events:
            hao, = axis1.plot([e.preferred_origin().time.datetime,
                               e.preferred_origin().time.datetime], [0, n_trace+1],
                              color='green', label='Autoloc OT')
        # display autoloc residuals RMS
        axis2.text(.01, .05, f"N={event_auto.preferred_origin().quality.used_phase_count}:"
                             f" {event_auto.preferred_origin().quality.standard_error:.2f} s",
                   fontweight='bold', fontsize=8, color='purple', transform=axis2.transAxes)
        # display number of Autoloc events
        axis1.text(event_cata.preferred_origin().time.datetime, n_trace+1,
                   f"N={len(events)}", color='green', ha='center', va='bottom')
    # axis limits
    axis1.set_ylim([0, n_trace + 1])
    axis2.set_ylim([0, n_trace + 1])
    # legend
    if events:
        if event_cata:
            if has:
                hh = [hw, hco, hao, hthp, hcp, hcs, hap, has, hta]
            else:
                hh = [hw, hco, hao, hthp, hcp, hcs, hap, hta]
        else:
            if has:
                hh = [hw, hao, hthp, hcp, hcs, hap, has]
            else:
                hh = [hw, hao, hthp, hcp, hcs, hap]
    else:
        if event_cata:
            if has:
                hh = [hw, hco, hthp, hcp, hcs, hap, has]
            else:
                hh = [hw, hco, hthp, hcp, hcs, hap]
        else:
            if has:
                hh = [hw, hthp, hcp, hcs, hap, has]
            else:
                hh = [hw, hthp, hcp, hcs, hap]
    axis1.legend(handles=hh, loc='lower left', fontsize=8)
    # station and pick numbers
    axis1.text(-.01, 1.01, f"N={n_trace}", ha='right', va='center', transform=axis1.transAxes)
    # replace numerical tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace + 1, 1))
    axis1.set_yticklabels(stn_lbl, fontsize=5)
    axis2.set_yticklabels([])
    # axis labels
    axis1.set_xlabel('Time [s]', fontweight='bold')
    axis1.set_ylabel('Station', fontweight='bold')
    axis2.set_xlabel('\u0394t [s]', fontweight='bold')
    # MAP
    # show location of stations without automatic picks
    for nt in isn_inv:
        for st in nt.stations:
            if not stream.select(network=nt.code, station=st.code):
                axis3.plot(st.longitude, st.latitude, 'x', mec='black', mfc='none', markersize=7, alpha=.7)
    # plot catalogue event location
    hc, = axis3.plot(event_cata.preferred_origin().longitude, event_cata.preferred_origin().latitude, '*',
                     mfc='orange', mec='black', markersize=10, label='Catalogue', alpha=.7)
    ell_unc = mpl.patches.Ellipse((event_cata.preferred_origin().longitude, event_cata.preferred_origin().latitude),
                                  width=event_cata.preferred_origin().longitude_errors.uncertainty,
                                  height=event_cata.preferred_origin().latitude_errors.uncertainty,
                                  color='orange', alpha=.1)
    axis3.add_patch(ell_unc)
    ha = []
    if events:
        # plot autoloc events location
        for e in events:
            if len(events) > 1 and e == event_auto:
                # highlight event closest to catalogue event
                axis3.plot(e.preferred_origin().longitude, e.preferred_origin().latitude, '*',
                           mfc='red', mec='black', markersize=10, alpha=.7)
            else:
                # show autoloc events
                ha, = axis3.plot(e.preferred_origin().longitude, e.preferred_origin().latitude, '*',
                                 mfc='green', mec='black', markersize=10, label='Autoloc', alpha=.7)
            # build ellipce object
            ell_unc = mpl.patches.Ellipse((e.preferred_origin().longitude, e.preferred_origin().latitude),
                                          width=e.preferred_origin().longitude_errors.uncertainty/(2.*np.pi*rrad/360.),
                                          height=e.preferred_origin().latitude_errors.uncertainty /
                                          (2.*np.pi*rrad/360.), color='green', alpha=.1)
            # plot error ellipse
            axis3.add_patch(ell_unc)
        # map legend
        hm = [hs, hc, ha, hf, hq]
    else:
        # map legend
        hm = [hs, hc, hf, hq]
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
    # EVENTS
    # plot catalogue event location
    axis4.plot(event_cata.preferred_origin().longitude, event_cata.preferred_origin().latitude, '*',
               mfc='orange', mec='black', markersize=10, alpha=.7)
    # display catalogue magnitude
    if event_cata.preferred_magnitude().mag and not np.isnan(event_cata.preferred_magnitude().mag):
        axis4.text(event_cata.preferred_origin().longitude-2., event_cata.preferred_origin().latitude,
                   f"M{event_cata.preferred_magnitude().mag:3.1f}",
                   ha='right', va='center', color='orange', clip_on=True, fontsize=8)
    if events:
        # plot autoloc event locations and magnitudes
        for e in events:
            axis4.plot(e.preferred_origin().longitude, e.preferred_origin().latitude, '*',
                       mfc='green', mec='black', markersize=10, alpha=.7)
            if e.preferred_magnitude() and e.preferred_magnitude().mag:
                axis4.text(e.preferred_origin().longitude+2., e.preferred_origin().latitude,
                           f"M{e.preferred_magnitude().mag:3.1f}",
                           ha='left', va='center', color='green', clip_on=True, fontsize=8)
        # figure title
        tit1 = f"{datetime.strftime(event_cata.preferred_origin().time.datetime, '%d/%m/%Y %H:%M:%S')} \u2013 " \
               f"{datetime.strftime(event_auto.preferred_origin().time.datetime, '%d/%m/%Y %H:%M:%S')} " \
               f"({event_auto.preferred_origin().time_errors.uncertainty:.2f} s)"
        tit2 = f"[{event_cata.preferred_origin().latitude:.2f},{event_cata.preferred_origin().longitude:.2f}] \u2013" \
               f" [{event_auto.preferred_origin().latitude:.2f},{event_auto.preferred_origin().longitude:.2f}]" \
               f" ([{event_auto.preferred_origin().latitude_errors.uncertainty:.2f}," \
               f"{event_auto.preferred_origin().longitude_errors.uncertainty:.2f}] km)"
        cmag = np.nan
        if event_cata.preferred_magnitude() and event_cata.preferred_magnitude().mag:
            cmag = event_cata.preferred_magnitude().mag
        amag = np.nan
        if event_auto.preferred_magnitude() and event_auto.preferred_magnitude().mag:
            amag = event_auto.preferred_magnitude().mag
        tit3 = f"M{cmag:.2f} \u2013 M{amag:.2f}"
        tit4 = f"{event_cata.preferred_origin().depth/1000.:.2f} \u2013" \
               f" {event_auto.preferred_origin().depth/1000.:.2f}" \
               f" ({event_auto.preferred_origin().depth_errors.uncertainty/1000.:.2f} km)"
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
chn = '(B|H|E)(H|N)(Z|N|E)'
fpar = {'rmhp': 5., 'taper': 10., 'bworder': 4, 'bwminf': 4., 'bwmaxf': 8.,
        'sta': .2, 'lta': 10., 'trigon': 3., 'trigoff': 1.5}

# area of interest
rrad = 6371.
igrd = [29., 34., 34., 36.]
mgrd = [29., 34., 33., 37.]
rgrd = [23., 43., 25., 45.]
vel_mod = 'giimod'

# working directory
if os.environ['LOGNAME'] == 'sysop':
    wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/autopicker-pb'
else:
    wdir = '/home/lewis/autopicker-pb'
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
# DATA PROCESSING
if_proc = True
if if_proc:
    # loop over events
    for _, erow in etab.iterrows():
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                                  '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        # if evt == '20210617215634852' or evt == '20211027045305932':
        #     continue
        if evt != '20210831091847474':
            continue
        evt_cata = isn_client.get_events(eventid=erow.EventID)[0]
        print(f"{evt} M{evt_cata.preferred_magnitude().mag:3.1f}")

        ################################################################################################################
        # RETRIEVE WAVEFORM DATA
        if path.exists(f"{wdir}/{evt}.mseed") == 0:
            # .mseed file name
            mfile = get_traces_deci(evt, evt_cata.preferred_origin().time.datetime, twin)
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
            print()
            continue
        else:
            # autopicker command (using raw .mseed file)
            print(f" Running playback for {evt}")
            os.system(f'{wdir}/playback.sh {evt}')
        print()

########################################################################################################################
# FIGURE BUILDING
if_plot = True
if if_plot:
    # loop over events
    for _, erow in etab.iterrows():
        # read event parameters
        evt = datetime.strftime(datetime.strptime(str(erow.OriginTime).replace('+00:00', ''),
                                                  '%Y-%m-%d %H:%M:%S.%f'), '%Y%m%d%H%M%S%f')[:-3]
        # if evt == '20210617215634852' or evt == '20211027045305932':
        #     continue
        if evt != '20210831091847474':
            continue
        evt_cata = isn_client.get_events(eventid=erow.EventID)[0]
        print(f"{evt} M{evt_cata.preferred_magnitude().mag:3.1f}")
        # check whether figure already exists or not
        oxml = f"{evt}_picks.xml"
        if path.exists(f"{wdir}/{oxml.replace('_picks.xml', '.png')}") != 0:
            print()
            continue

        ################################################################################################################
        # RETRIEVE AUTOMATIC PICKS
        # load waveform data to plot (not .raw.mseed)
        isn_traces = read(f"{wdir}/{evt}.mseed")
        # add event data to waveforms
        isn_traces = add_event_data(isn_traces, evt_cata, vel_mod, isn_inv)
        # sort according to newly added distance to event
        isn_traces.sort(['distance'], reverse=True)
        # check if playback was ran
        if path.exists(f"{wdir}/{oxml}") == 0 or os.path.getsize(f"{wdir}/{oxml}") == 0:
            print(f' Playback not ran for {evt}')
            print()
            continue
        # read output file
        isn_traces, na = read_autopick_xml(f"{wdir}/{oxml}", isn_traces)
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
        # tper = pd.date_range(evt_cata.preferred_origin().time.datetime-twin,
        #                      evt_cata.preferred_origin().time.datetime+twin, freq='30S').to_list()
        # for i in range(len(tper)-1):
        #     tab = atab.loc[(atab.pick > tper[i]) & (atab.pick <= tper[i+1])]
        #     if not tab.empty and len(tab) > 5:
        #         wtab.append(tper[i])

        ################################################################################################################
        # RETRIEVE CATALOGUE PICKS
        isn_traces, nc = get_catalogue_picks(isn_client, str(evt_cata.resource_id.id).
                                             replace('smi:org.gfz-potsdam.de/geofon/', ''), isn_traces)
        print(f" {nc} catalogue picks")

        ################################################################################################################
        # MORE DATA PROCESSING (FOR PLOTTING ONLY)
        # # remove traces with <2 automatic picks (assuming first pick is for event)
        # for t in isn_traces:
        #     if hasattr(t.stats, 'auto_tt') and len(t.stats.auto_tt) < 2 or (not hasattr(t.stats, 'auto_tt')):
        #         isn_traces.remove(t)
        # removing all Meiron stations apart from MMA0B
        for tr in isn_traces:
            if re.search('MMA', tr.stats.station) or re.search('MMB', tr.stats.station) or \
                    re.search('MMC', tr.stats.station):
                isn_traces.remove(tr)
        # check scautopick bindings to remove unlisted stations and/or channels
        flst = glob.glob('/home/sysop/seiscomp/etc/key/station_IS_*')
        for file in flst:
            if os.path.getsize(file) == 0:
                wf_sel = isn_traces.select(network=file.split('_')[1], station=file.split('_')[2])
                for wf in wf_sel:
                    isn_traces.remove(wf)
            else:
                fid = open(file, 'r')
                lines = fid.readlines()
                fid.close()
                wf_sel = isn_traces.select(network=file.split('_')[1], station=file.split('_')[2])
                for wf in wf_sel:
                    if wf.stats.channel != lines[1].split(':')[1].replace('\n', '').split('_')[1]+'Z':
                        isn_traces.remove(wf)
        # apply taper to all traces
        isn_traces.taper(max_percentage=.5, type='cosine', max_length=fpar['taper'], side='left')
        # apply high-pass filter to all traces
        isn_traces.filter('highpass', freq=1./fpar['rmhp'])
        # apply band-pass filter to all traces
        isn_traces.filter('bandpass', freqmin=fpar['bwminf'], freqmax=fpar['bwmaxf'], corners=fpar['bworder'])
        # # downsample data (to avoid memory issues when plotting)
        # isn_traces.resample(1.0, window='hann')

        ################################################################################################################
        # BUILD OUTPUT FIGURE
        # # add event origin time to trace headers
        # for tr in isn_traces:
        #     tr.stats['origin_time'] = evt_cata.preferred_origin().time.datetime
        # only if figure does not already exist
        if path.exists(f"{wdir}/{oxml.replace('_picks.xml', '.png')}") == 0:
            print(f' {len(isn_traces)} waveforms to plot')
            # plot_autopick_cont_sec(isn_traces, wtab, f"{wdir}/{oxml.replace('_picks.xml', '.html')}")
            plot_autopick_evt_sec(isn_traces, evt_cata, f"{wdir}/{oxml.replace('_picks.xml', '.png')}")
        isn_traces = None
        print()
