import os
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
from GiiPhaseDB import GiiPhaseDB, read_zone_all
from shapely.geometry import Point


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    # creating log file
    log = logging.getLogger(logger_name)
    # messaging level in log file
    log.setLevel(level)
    # message format
    format_string = "%(asctime)s.%(msecs)03d \u2013 %(levelname)s \u2013 %(funcName)s: %(message)s"
    # apply format
    log_format = logging.Formatter(format_string, '%Y-%m-%d %H:%M:%S')
    # creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    log.addHandler(file_handler)
    return log


def get_zone_id(sdb, lat, lon):
    point = Point(lat, lon)
    for zone_id in ["NNL", "NNR", "NNT"]:
        zones = read_zone_all(sdb, zone_id)
        for zk in zones.keys():
            for zp in zones[zk].poligons.keys():
                polygon = zones[zk].poligons[zp]
                if polygon.contains(point):
                    return zone_id
    return None


def get_waveforms_deci(mpath, ori_time, time_win, networks, channels, db_dir):
    """
    :param mpath: .mseed file path
    :param ori_time: event origin time
    :param time_win: half-length of time window
    :param networks: seismic networks for which to request waveform data
    :param channels: seismic channels for which to request waveform data
    :param db_dir: path to database directory from which to request waveform data
    :return: nothing
    """
    if os.path.exists(mpath) == 0:
        # define start and end times for data retrieval
        tbeg = str(datetime.strftime(ori_time - time_win, '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(ori_time + time_win, '%Y-%m-%d %H:%M:%S'))
        # retrieve data using SeisComP's 'scart' command
        os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{tbeg}~{tend}" {db_dir} > {mpath}')
        # delete file if empty
        if os.path.getsize(mpath) == 0:
            os.remove(mpath)
        # read created miniSEED file
        stream = read(mpath)
        # remove problematic AMZNI (20 Hz) trace
        for trace in stream:
            if trace.stats.station == 'AMZNI' and trace.stats.channel == 'BHZ' and trace.stats.sampling_rate == 20.:
                stream.remove(trace)
        # merge traces from different networks/channels separately (different sampling rates)
        s1 = stream.select(network='IS', channel='BHZ').merge(fill_value='interpolate')
        s2 = stream.select(network='IS', channel='HHZ').merge(fill_value='interpolate')
        s3 = stream.select(network='IS', channel='ENZ').merge(fill_value='interpolate')
        s4 = stream.select(network='IS', channel='SHZ').merge(fill_value='interpolate')
        s5 = stream.select(network='GE', channel='BHZ').merge(fill_value='interpolate')
        s6 = stream.select(network='GE', channel='HHZ').merge(fill_value='interpolate')
        s7 = stream.select(network='GE', channel='ENZ').merge(fill_value='interpolate')
        s8 = stream.select(network='GE', channel='SHZ').merge(fill_value='interpolate')
        stream = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
        # write miniSEED file
        stream.write(mpath)
    else:
        # delete file if existing but empty
        if os.path.getsize(mpath) == 0:
            os.remove(mpath)


def process_waveforms(mpath, filt_param, evt_xml, ref_mod, sta_inv, rm_acc=False):
    """
    :param mpath: data streamer containing waveforms to process
    :param filt_param: dictionary containing filter parameters for waveforms
    :param evt_xml: xml container with all event data and picks
    :param ref_mod: reference velocity model to use for theoretical travel times
    :param sta_inv: .xml inventory containing all station info.
    :param rm_acc: remove accelerometer data from stream (default is False)
    :return: data streamer containing processed waveforms
    """
    # Earth radius
    earth_rad = 6371.
    # initialise TauP to compute theroetical arrivals
    theory = TauPyModel(model=ref_mod)
    # read waveforms
    stream = read(mpath)
    # initialise list of traces to delete
    to_del = []
    for trace in stream:
        # trace channel string
        cha = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}'
        # ONLY KEEP SHZ DATA FOR IS.PRNI
        if 'SHZ' in cha:
            if 'PRNI' not in cha and 'IS' not in cha:
                to_del.append(trace)
                continue
        # REMOVE ACCELEROMETER DATA WITHOUT PICKS
        if rm_acc and 'ENZ' in cha:
            # initialise logical variable if pick found
            ff = False
            # finding channel index in event picks
            pind = [index for index, pick in evt_xml.picks if pick.waveform_id.id == cha]
            if pind:
                ppic = evt_xml.picks[pind[0]]
                # finding channel index in event arrivals
                aind = [index for index, arrival in enumerate(evt_xml.preferred_origin().arrivals)
                        if arrival.pick_id.id == ppic.resource_id.id]
                if aind:
                    ff = True
            # add to to-delete list if pick not found
            if not ff:
                to_del.append(trace)
                continue
        # SPECIAL CASE OF ISP (two BHZ channels on two locations --> keep GE.ISP.00.BHZ)
        if cha == 'GE.ISP.10.BHZ':
            to_del.append(trace)
            continue
        # REMOVE DUPLICATE VELOCITY DATA
        if 'BHZ' in cha and \
                stream.select(station=trace.stats.station, network=trace.stats.network, channel='HHZ') != [] or \
                'HHZ' in cha and \
                stream.select(station=trace.stats.station, network=trace.stats.network, channel='BHZ') != []:
            # initialise logical variables
            f1 = False
            f2 = False
            # select waveforms for same station, all channels
            lst = stream.select(network=trace.stats.network, station=trace.stats.station, channel='[BH]HZ')
            # loop over waveforms found
            for tr in lst:
                ch = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}'
                # finding channel index in event picks
                pind = [index for index, pick in enumerate(evt_xml.picks) if pick.waveform_id.id == ch]
                if pind:
                    ppic = evt_xml.picks[pind[0]]
                    # finding channel index in event arrivals
                    aind = [index for index, arrival in enumerate(evt_xml.preferred_origin().arrivals)
                            if arrival.pick_id.id == ppic.resource_id.id]
                    if 'BHZ' in ch and aind:
                        f1 = True
                    if 'HHZ' in ch and aind:
                        f2 = True
            # choose BHZ over HHZ if neither one of the channels was picked
            if not f1 and not f2:
                st = stream.select(network=trace.stats.network, station=trace.stats.station, channel='HHZ')
                for s in st:
                    to_del.append(s)
            # choose the picked channel if only one was
            if f1 and not f2:
                st = stream.select(network=trace.stats.network, station=trace.stats.station, channel='HHZ')
                for s in st:
                    to_del.append(s)
            if not f1 and f2:
                st = stream.select(network=trace.stats.network, station=trace.stats.station, channel='BHZ')
                for s in st:
                    to_del.append(s)
        # EPICENTRAL DISTANCE
        # find station/channel in inventory
        station = sta_inv.select(network=trace.stats.network, station=trace.stats.station,
                                 channel=trace.stats.channel, location=trace.stats.location)
        # remove trace if station/channel not in inventory
        if not station:
            to_del.append(trace)
            continue
        # calculate event-station distance
        dis = gdist.distance((evt_xml.preferred_origin().latitude,
                              evt_xml.preferred_origin().longitude),
                             (station[0].stations[0].channels[0].latitude,
                              station[0].stations[0].channels[0].longitude))
        # add event-station distance to trace header
        trace.stats.distance = dis.m
        # add station location
        trace.stats['sta_lat'] = station[0].stations[0].channels[0].latitude
        trace.stats['sta_lon'] = station[0].stations[0].channels[0].longitude
        # THEORETICAL ARRIVALS
        # compute theoretical travel time for all possible first P arrivals
        theo_tp = theory.get_travel_times(source_depth_in_km=evt_xml.preferred_origin().depth / 1000.,
                                          distance_in_degree=dis.km / (2. * np.pi * earth_rad / 360.),
                                          phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        theo_ts = theory.get_travel_times(source_depth_in_km=evt_xml.preferred_origin().depth / 1000.,
                                          distance_in_degree=dis.km / (2. * np.pi * earth_rad / 360.),
                                          phase_list=['s', 'S', 'Sg', 'Sn', 'Sdiff'])
        # add theoretical travel time to trace header
        if len(theo_tp) != 0:
            trace.stats['theo_tp'] = evt_xml.preferred_origin().time.datetime + timedelta(seconds=theo_tp[0].time)
        else:
            # save as NaT in case of no theoretical arrival
            trace.stats['theo_tp'] = np.datetime64('NaT')
        if len(theo_ts) != 0:
            trace.stats['theo_ts'] = evt_xml.preferred_origin().time.datetime + timedelta(seconds=theo_ts[0].time)
        else:
            # save as NaT in case of no theoretical arrival
            trace.stats['theo_ts'] = np.datetime64('NaT')
    # delete selected waveforms (that were not already removed)
    for trace in to_del:
        try:
            stream.remove(trace)
        except:
            continue
    # apply taper to all traces
    stream.taper(max_percentage=.5, type='cosine', max_length=filt_param['taper'], side='left')
    # apply high-pass filter to all traces
    stream.filter('highpass', freq=1. / filt_param['rmhp'])
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply Butterworth band-pass filter to all traces
    stream.filter('bandpass', freqmin=filt_param['bwminf'], freqmax=filt_param['bwmaxf'], corners=filt_param['bworder'])
    return stream


def read_autopick_xml(year, month, stream, work_dir):
    """
    :param year: year to list automatic false events for
    :param month: year to list automatic false events for
    :param stream: stream name in the format NET.STA.LOC.CHA
    :param work_dir: working directory
    :return: DataFrame containing all automatic picks
    """
    # create time period strings
    if year == 2022 and month == 12:
        tper = '2022-12-27 10:00:00~2023-01-01 00:00:00'
    elif year == 2023:
        tper = f'{year}-{month:02d}-01 00:00:00~{year}-{month+1:02d}-01 00:00:00'
    else:
        tper = ''
    # create .xml file if missing
    netsta = f"{stream.split('.')[0]}.{stream.split('.')[1]}"
    if os.path.exists(f'{work_dir}/{year}-{month:02d}/picks/{netsta}.xml') == 0:
        os.system(f"dump_picks -t '{tper}' -n --net-sta {netsta} -o {work_dir}/{year}-{month:02d}/picks/{netsta}.xml")
    # initialise picks table
    pick_tab = pd.DataFrame({'Stream': pd.Series(dtype='string'), 'Phase': pd.Series(dtype='string'),
                             'PickTime': pd.Series(dtype='datetime64[ms]')})
    # loop over .xml file lines
    for line_xml in ETree.parse(f'{work_dir}/{year}-{month:02d}/picks/{netsta}.xml').getroot()[0]:
        # search pick sections
        if re.search('pick', str(line_xml.tag)):
            # locationCode can be empty
            loc = (line_xml[1].attrib['locationCode'] if (len(line_xml[1].attrib) == 4) else '')
            # fill picks table
            pick_tab.loc[pick_tab.shape[0]] = [f"{line_xml[1].attrib['networkCode']}."
                                               f"{line_xml[1].attrib['stationCode']}.{loc}."
                                               f"{line_xml[1].attrib['channelCode']}", line_xml[3].text,
                                               datetime.strptime(line_xml[0][0].text, '%Y-%m-%dT%H:%M:%S.%fZ')]
    return pick_tab


def write_picks_table(detec_type, phase, year, month, work_dir):
    """
    :param detec_type: detection type to visualise; one-character string: T/F/C (True/False/Catalogue)
    :param phase: seismic phase of interest (P/S)
    :param year: year to list automatic false events for
    :param month: year to list automatic false events for
    :param work_dir: working directory
    :return: nothing
    """
    # FDSN client (if needed)
    isn_client = None
    if detec_type == 'C':
        isn_client = Client('http://172.16.46.102:8181/')
    # read monthly summary
    summary_tab = pd.read_csv(f'{work_dir}/{year}-{month:02d}/summary.csv')
    # select desired event detection type
    if detec_type == 'C':
        summary_tab = summary_tab[(summary_tab.Detection == 'T') |
                                  (summary_tab.Detection == 'M')].reset_index(drop=True)
    else:
        summary_tab = summary_tab[summary_tab.Detection == detec_type].reset_index(drop=True)
    # output table
    output_tab = pd.DataFrame({'EventID': pd.Series(dtype='string'), 'Stream': pd.Series(dtype='string'),
                               'PickTime': pd.Series(dtype='datetime64[ms]'), 'Amplitude': pd.Series(dtype='float64'),
                               'SNR': pd.Series(dtype='float64')})
    # loop over selected events
    for _, event in summary_tab.iterrows():
        evt_xml = []
        evid = ''
        # case True detections
        if detec_type == 'T':
            # read event file
            evt_xml = read_events(f'{work_dir}/{year}-{month:02d}/{event.AutoID}.xml')
            # event ID
            evid = event.AutoID
        # case False detections
        elif detec_type == 'F':
            # read False event file
            evt_xml = read_events(f'{work_dir}/{year}-{month:02d}/false/{event.AutoID}.xml')
            # event ID
            evid = event.AutoID
        # case Catalogue detections
        elif isn_client and detec_type == 'C':
            # retrieve event data from FDSN
            evt_xml = isn_client.get_events(eventid=event.CataID, includearrivals=True)
            # event ID
            evid = event.CataID
        if evt_xml:
            # loop over arrivals
            for arrival in evt_xml.events[0].preferred_origin().arrivals:
                # select phase of interest only
                if arrival.phase != phase:
                    continue
                # reinitialise variables
                pstr = ''
                parr = np.datetime64('NaT')
                pamp = np.nan
                psnr = np.nan
                # loop over picks
                for pick in evt_xml.events[0].picks:
                    # match arrival and pick IDs
                    if arrival.pick_id.id == pick.resource_id.id:
                        pstr = pick.waveform_id.id
                        parr = pick.time.datetime
                # loop over amplitudes
                for amplitude in evt_xml.events[0].amplitudes:
                    # match arrival and pick IDs
                    if arrival.pick_id.id == amplitude.pick_id.id:
                        pamp = amplitude.generic_amplitude
                        psnr = amplitude.snr
                # add pick to table
                output_tab.loc[output_tab.shape[0]] = [evid, pstr, parr, pamp, psnr]
        else:
            if detec_type == 'T' or detec_type == 'F':
                print(f"Event not found: {event.AutoID}")
            elif detec_type == 'C':
                print(f"Event not found: {event.CataID}")
    # sort according to event OT
    output_tab.set_index('EventID').sort_index()
    # write picks table
    output_tab.to_csv(f'{work_dir}/{year}-{month:02d}/picks-{phase}-{detec_type}.csv', index=False)
    return


def prepare_tables(stream, cat_evt, auto_list, work_dir):
    """
    :param stream: data streamer of waveforms to later plot along with picks
    :param cat_evt: xml container for catalogue event and picks
    :param auto_list: list of automatic event IDs
    :param work_dir: working directory
    :return: picks_table, evt_tab
    """
    # waveforms time window (same as 'scolv')
    tmin = min([tr.stats.theo_tp for tr in stream]) - timedelta(minutes=1)
    tmax = max([tr.stats.theo_tp for tr in stream]) + timedelta(minutes=2)
    # create output events table for all catalogue and automatic events
    evt_tab = pd.DataFrame({'EventID': pd.Series(dtype='string'), 'DateTime': pd.Series(dtype='datetime64[ms]'),
                            'Latitude': pd.Series(dtype='float64'), 'Longitude': pd.Series(dtype='float64'),
                            'Depth': pd.Series(dtype='float64'), 'Mag': pd.Series(dtype='float64'),
                            'Type': pd.Series(dtype='string'), 'Author': pd.Series(dtype='string')})
    # create output picks table for all catalogue and automatic events
    picks_tab = pd.DataFrame({'Network': pd.Series(dtype='string'), 'Station': pd.Series(dtype='string'),
                              'Location': pd.Series(dtype='string'), 'Channel': pd.Series(dtype='string'),
                              'EventID': pd.Series(dtype='string'), 'Phase': pd.Series(dtype='string'),
                              'Pick': pd.Series(dtype='datetime64[ms]'), 'Residual': pd.Series(dtype='float64')})
    # CATALOGUE EVENT
    # add catalogue event data to events table
    evt_tab.loc[evt_tab.shape[0]] = [cat_evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', ''),
                                     cat_evt.preferred_origin().time.datetime, cat_evt.preferred_origin().latitude,
                                     cat_evt.preferred_origin().longitude, cat_evt.preferred_origin().depth/1000.,
                                     cat_evt.preferred_magnitude().mag if cat_evt.preferred_magnitude() else np.nan,
                                     cat_evt.event_type, cat_evt.creation_info.author]
    # CATALOGUE PICKS
    # loop over arrivals
    for arrival in cat_evt.preferred_origin().arrivals:
        # extracting index of channel from catalogue arrivals
        pind = [index for index, pick in enumerate(cat_evt.picks)
                if arrival.pick_id.id == pick.resource_id.id]
        # checking pick index
        if len(pind) > 1:
            print(f"Multiple catalogue pick matches for "
                  f"{cat_evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')}")
            exit()
        else:
            pind = pind[0]
        # extracting arrival from catalogue arrivals
        pick = cat_evt.picks[pind]
        # add row to output picks table
        picks_tab.loc[picks_tab.shape[0]] =\
            [pick.waveform_id.network_code, pick.waveform_id.station_code, pick.waveform_id.location_code,
             pick.waveform_id.channel_code, cat_evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', ''),
             pick.phase_hint, pick.time.datetime, arrival.time_residual]
    # AUTOMATIC EVENT & PICKS
    # loop over automatic event IDs
    for auto_evt in auto_list:
        # checking event .xml file exists
        xml_path = f"{work_dir}/{datetime.strftime(cat_evt.preferred_origin().time.datetime, '%Y-%m')}" \
                   f"/{auto_evt}.xml"
        # checking false/ directory if not in monthly directory
        if os.path.exists(xml_path) == 0:
            xml_path = f"{work_dir}/{datetime.strftime(cat_evt.preferred_origin().time.datetime, '%Y-%m')}" \
                       f"/false/{auto_evt}.xml"
        # retrieving event .xml file if not found anywhere
        if os.path.exists(xml_path) == 0:
            os.system(f"scxmldump -E {auto_evt} -PAMFf -o {xml_path}")
        # reading event .xml file
        auto_xml = read_events(xml_path).events[0]
        # listing event picks
        lpic = [p.time.datetime for p in auto_xml.picks]
        # skipping if no picks within waveform time window
        if not lpic or (not (tmin < min(lpic) < tmax) and not (tmin < max(lpic) < tmax)):
            continue
        # add automatic event data to events table
        evt_tab.loc[evt_tab.shape[0]] = [
            auto_xml.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', ''),
            auto_xml.preferred_origin().time.datetime, auto_xml.preferred_origin().latitude,
            auto_xml.preferred_origin().longitude, auto_xml.preferred_origin().depth/1000.,
            auto_xml.preferred_magnitude().mag if auto_xml.preferred_magnitude() else np.nan,
            auto_xml.event_type, auto_xml.creation_info.author]
        # loop over arrivals
        for arrival in auto_xml.preferred_origin().arrivals:
            pind = [index for index, pick in enumerate(auto_xml.picks)
                    if arrival.pick_id.id == pick.resource_id.id]
            # loop over picks (in the not-so-rare possibility of several arrivals per channel)
            for ind in pind:
                pick = auto_xml.picks[ind]
                # checking pick is within waveform time window and ignore KO stations and non-vertical channels
                if tmin < pick.time.datetime < tmax \
                        and pick.waveform_id.network_code != 'KO' \
                        and 'Z' in pick.waveform_id.channel_code:
                    # add row to output picks table
                    picks_tab.loc[picks_tab.shape[0]] =\
                        [pick.waveform_id.network_code, pick.waveform_id.station_code,
                         pick.waveform_id.location_code, pick.waveform_id.channel_code,
                         auto_evt, pick.phase_hint, pick.time.datetime, np.nan]
    return picks_tab, evt_tab


def plot_cata_evt_sec(stream, evt_tab, picks_tab, sta_inv, log_file, fig_name):
    """
    :param stream: data streamer of waveforms to process
    :param evt_tab: DataFrame containing all catalogue and automatic event data
    :param picks_tab: DataFrame containing all catalogue and automatic picks data
    :param sta_inv: .xml inventory containing all station info.
    :param log_file: log file to use
    :param fig_name: figure file name
    :return: nothing
    """
    # axis limits for residual plot
    rlim = 5.
    # geographic boundaries
    local_map = [29., 34., 33.5, 36.5]
    regional_map = [23., 43., 25., 45.]
    # plotting mode
    mpl.use('Agg')
    # select catalogue event from events table
    cevt = evt_tab[evt_tab.EventID.str.contains('gsi')].reset_index(drop=True).iloc[0]
    # select automatic events from events table
    aevt = evt_tab[evt_tab.EventID.str.contains('lew')].reset_index(drop=True)
    # create colour and marker tables for events
    tcol = list(mpl.colors.TABLEAU_COLORS.keys())
    evt_colour = [c.replace('tab:', '') for c in tcol[0:len(aevt)]]
    if len(evt_colour) > 3:
        evt_colour[evt_colour.index('red')] = 'black'
    marker = ['s', '^', 'D', 'P', '8', 'p', '<', 'h', 'X', '>']
    evt_marker = marker[0:len(aevt)]
    # waveforms time window (following SeisComP's time window definition)
    tmin = min([tr.stats.theo_tp for tr in stream]) - timedelta(minutes=1)
    tmax = max([tr.stats.theo_tp for tr in stream]) + timedelta(minutes=2)
    # create figure & axes
    fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200,
                                              gridspec_kw={'width_ratios': [2, 1, 1]})
    # AXIS 1: VELOCITY WAVEFORMS
    # show axis grids
    axis1.grid(which='both', axis='both')
    # set x-axis limits
    axis1.set_xlim([tmin, tmax])
    # set date format to x-axis tick labels
    axis1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # AXIS 2: RESIDUAL OVER DISTANCE PLOT
    # show axis grids
    axis2.grid(which='both', axis='both')
    # set x-axis limits
    axis2.set_xlim([-rlim, rlim])
    # AXIS 3: LOCAL STATIONS & EVENT MAP
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=local_map[2]+.2, llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3], urcrnrlat=local_map[1], resolution='i', ax=axis3)
    # draw map
    m1.drawmapboundary(fill_color='none')
    # fill continents
    m1.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels=[left,right,top,bottom])
    m1.drawparallels(np.arange(np.floor(m1.llcrnrlat), np.floor(m1.urcrnrlat) + 1, 1.),
                     labels=[True, False, True, False])
    m1.drawmeridians(np.arange(np.floor(m1.llcrnrlon), np.floor(m1.urcrnrlon) + 1, 1.),
                     labels=[True, False, False, True])
    # faults
    flts_id = open('/home/sysop/.seiscomp/spatial/vector/ActiveFaults/activefaults.bna', 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    hf = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            hf, = axis3.plot(flt.lon, flt.lat, '.6', linewidth=.5, label='Faults', zorder=1)
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # quarries
    quar_id = open('/home/sysop/.seiscomp/spatial/vector/Quarries/quarries.bna', 'r')
    quar = quar_id.readlines()
    quar_id.close()
    hq = []
    for iii in range(len(quar)):
        if not re.search('"', quar[iii]):
            l_line = quar[iii].replace('\n', '').split(',')
            if l_line[0] != '' and l_line[1] != '':
                hq = axis3.scatter(float(l_line[0]), float(l_line[1]), s=9, c='blue', marker='s',
                                   edgecolors='black', label='Quarries', alpha=.1, zorder=2)
    # AXIS 4: INSET MAP FOR REGIONAL SETTINGS
    axis4 = inset_axes(axis3, '30%', '18%', loc='lower left')
    m2 = Basemap(projection='cyl', llcrnrlon=regional_map[2], llcrnrlat=regional_map[0],
                 urcrnrlon=regional_map[3], urcrnrlat=regional_map[1], resolution='l', ax=axis4)
    # draw map
    m2.drawmapboundary(fill_color='white')
    # fill continents
    m2.fillcontinents(color='0.8', lake_color='white')
    # plot GE station locations in regional map
    for station in sta_inv.networks[0].stations:
        if not (local_map[0] < station.latitude < local_map[1] and local_map[2] < station.longitude < local_map[3]):
            axis4.scatter(station.longitude, station.latitude,
                          s=25, color='cyan', marker='^', edgecolors='black', alpha=.2, zorder=5)
            axis4.text(station.longitude + .5, station.latitude + .1, station.code,
                       c='black', ha='left', va='bottom', fontsize=5, alpha=.2, clip_on=True)
    # highlight area of interest
    axis4.plot([local_map[2], local_map[2], local_map[3], local_map[3], local_map[2]],
               [local_map[0], local_map[1], local_map[1], local_map[0], local_map[0]], color='saddlebrown')
    # plot catalogue event location
    axis4.scatter(cevt.Longitude, cevt.Latitude, s=100, c='red', marker='*', edgecolors='black', zorder=100)
    # LOOP OVER TRACES
    # intialise counters
    n_trace = 0
    # initialise handle tables for legend
    hw = []
    ha = []
    ht = []
    hc = []
    hro = []
    hrc = []
    hrt = []
    hn1 = []
    hn2 = []
    # loop over traces
    for trace in stream:
        # counter
        n_trace += 1
        # time vector
        tvec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]') + \
            np.datetime64(str(trace.stats.starttime)[:-1])
        # define time window around theoretical S arrival for normalisation
        i1 = 0
        i2 = len(trace.data)
        if trace.stats.theo_ts + timedelta(seconds=10) <= max(tvec):
            i1 = np.argmin(np.abs(tvec - np.datetime64(trace.stats.theo_ts - timedelta(seconds=5))))
            i2 = np.argmin(np.abs(tvec - np.datetime64(trace.stats.theo_ts + timedelta(seconds=10))))
        if i2 - i1 < 200:
            i1 = 0
            i2 = len(trace.data)
        # use S-arrival max for normalisation
        wmax = 2 * max(trace.data[i1:i2])
        # use global max for normalisation if abnormal signals
        if max(trace.data / wmax) > 10.:
            wmax = max(trace.data)
        # channel string
        cha = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}'
        # plot waveform
        hw, = axis1.plot(tvec, trace.data / wmax + n_trace, color='grey', alpha=.7, label='Velocity')
        # plot theoretical travel time
        if trace.stats.theo_tp:
            ht, = axis1.plot([trace.stats.theo_tp, trace.stats.theo_tp], [n_trace - 1, n_trace + 1],
                             color='lime', alpha=.7, label=f'{vel_mod}')
        if trace.stats.theo_ts:
            axis1.plot([trace.stats.theo_ts, trace.stats.theo_ts], [n_trace - 1, n_trace + 1],
                       color='lime', alpha=.7, linestyle='dotted')
        # plot station in map
        st = sta_inv.select(network=trace.stats.network, station=trace.stats.station, channel=trace.stats.channel)
        if not st:
            log_file.warning(f'Missing station: {cha}')
        # use different symbols and colours for different network codes
        if trace.stats.network == 'IS':
            hn1 = axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude, s=25,
                                c='magenta', marker='^', edgecolors='black', alpha=.2,
                                label=trace.stats.network, zorder=3)
        elif trace.stats.network == 'GE':
            hn2 = axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude, s=25,
                                c='cyan', marker='^', edgecolors='black', alpha=.2, label=trace.stats.network, zorder=3)
        # display channel name
        htx = axis1.text(axis1.get_xlim()[0]-(axis1.get_xlim()[1]-axis1.get_xlim()[0])/150., n_trace, cha,
                         ha='right', va='center', fontsize=5)
        # CATALOGUE PICKS
        # initialise variables
        fcp = False
        fcs = False
        pc = pd.DataFrame()
        sc = pd.DataFrame()
        if not picks_tab[picks_tab.EventID.str.contains('gsi')].empty:
            # P picks
            pcat = picks_tab[picks_tab.EventID.str.contains('gsi') & (picks_tab.Network == trace.stats.network) &
                             (picks_tab.Station == trace.stats.station) & (picks_tab.Channel == trace.stats.channel) &
                             ((picks_tab.Phase == 'P') | (picks_tab.Phase == 'Pg') | (picks_tab.Phase == 'Pn'))]
            if not pcat.empty:
                pc = pcat.iloc[0]
                # plot catalogue pick
                hc, = axis1.plot([pc.Pick, pc.Pick], [n_trace - 1, n_trace + 1],
                                 color='red', alpha=.7, label='GSI', zorder=100)
                # plot catalogue residual
                hro = axis2.scatter(pc.Residual, n_trace, s=25, c='none',
                                    edgecolors='red', alpha=.7, label='GSI-Theo', zorder=100)
                fcp = True
            # S picks
            scat = picks_tab[picks_tab.EventID.str.contains('gsi') & (picks_tab.Network == trace.stats.network) &
                             (picks_tab.Station == trace.stats.station) & (picks_tab.Channel == trace.stats.channel) &
                             ((picks_tab.Phase == 'S') | (picks_tab.Phase == 'Sg') | (picks_tab.Phase == 'Sn'))]
            if not scat.empty:
                sc = scat.iloc[0]
                # plot catalogue pick
                axis1.plot([sc.Pick, sc.Pick], [n_trace - 1, n_trace + 1],
                           color='red', alpha=.7, linestyle='dotted', zorder=100)
                # plot catalogue residual
                axis2.scatter(sc.Residual, n_trace, s=25, c='none',
                              edgecolors='red', alpha=.7, linestyle='dotted', zorder=100)
                fcs = True
            # case any pick found
            if fcp or fcs:
                # change channel color to red
                htx.set(color='red')
                # highlight station in map
                if trace.stats.network == 'IS':
                    axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                  s=25, c='magenta', marker='^', edgecolors='red', zorder=4)
                elif trace.stats.network == 'GE':
                    axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                  s=25, c='cyan', marker='^', edgecolors='red', zorder=4)
                # display station name in map
                axis3.text(
                    st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude+.07,
                    st.networks[0].stations[0].code, color='red', ha='center', va='top', fontsize=5, clip_on=True)
                # connect station to event
                axis3.plot([cevt.Longitude, st.networks[0].stations[0].longitude],
                           [cevt.Latitude, st.networks[0].stations[0].latitude],
                           color='red', alpha=.2, zorder=2)
                # connect station to event in regional map if event is not local
                if cevt.Longitude < local_map[2] or cevt.Longitude > local_map[3] or \
                        cevt.Latitude < local_map[0] or cevt.Latitude > local_map[1]:
                    axis4.plot([cevt.Longitude, st.networks[0].stations[0].longitude],
                               [cevt.Latitude, st.networks[0].stations[0].latitude], color='red', alpha=.2, zorder=2)
        # AUTOMATIC PICKS
        # loop over automatic events
        for k, event in aevt.iterrows():
            # initialise boolean variables
            fap = False
            fas = False
            pa = pd.DataFrame()
            sa = pd.DataFrame()
            # finding event colour and marker
            ecol = evt_colour[k]
            emrk = evt_marker[k]
            # P picks
            paut = picks_tab[(picks_tab.EventID == event.EventID) & (picks_tab.Network == trace.stats.network) &
                             (picks_tab.Station == trace.stats.station) & (picks_tab.Channel == trace.stats.channel) &
                             ((picks_tab.Phase == 'P') | (picks_tab.Phase == 'Pg') | (picks_tab.Phase == 'Pn'))]
            if not paut.empty:
                pa = paut.iloc[0]
                ha1, = axis1.plot([pa.Pick, pa.Pick], [n_trace - 1, n_trace + 1],
                                  color=ecol, label=event.EventID, zorder=200, alpha=.7)
                if event.EventID not in [plt.getp(hh, 'label') for hh in ha]:
                    ha.append(ha1)
                fap = True
            # S picks
            saut = picks_tab[(picks_tab.EventID == event.EventID) & (picks_tab.Network == trace.stats.network) &
                             (picks_tab.Station == trace.stats.station) & (picks_tab.Channel == trace.stats.channel) &
                             ((picks_tab.Phase == 'S') | (picks_tab.Phase == 'Sg') | (picks_tab.Phase == 'Sn'))]
            if not saut.empty:
                sa = saut.iloc[0]
                axis1.plot([sa.Pick, sa.Pick], [n_trace - 1, n_trace + 1],
                           color=tcol[len(tcol) - k - 1], zorder=200, alpha=.7)
                fas = True
            # case any pick found
            if fap or fas:
                # right-hand side marker to quickly know which station has picks
                axis1.scatter(1.008+k*.01, n_trace / (len(stream) + 1), s=25, c='none', marker=emrk, edgecolors=ecol,
                              alpha=.7, clip_on=False, transform=axis1.transAxes)
            # plot residual w.r.t. catalogue pick (if catalogue pick exists)
            if fcp and fap:
                if abs((pa.Pick - pc.Pick).total_seconds()) < rlim:
                    hr = axis2.scatter((pa.Pick - pc.Pick).total_seconds(), n_trace, s=25, c='red',
                                       marker=emrk, edgecolors=ecol, alpha=.7, label=f'Auto-GSI ({event.EventID})')
                    # check other data point for that event not already in legend
                    if f'Auto-GSI ({event.EventID})' not in [plt.getp(hh, 'label') for hh in hrc]:
                        hrc.append(hr)
            if fcs and fas:
                if abs((sa.Pick - sc.Pick).total_seconds()) < rlim:
                    axis2.scatter((sa.Pick - sc.Pick).total_seconds(), n_trace, s=25, c='red',
                                  marker=emrk, edgecolors=ecol, alpha=.7, linestyle='dotted')
            # plot residual w.r.t. theoretical pick
            if fap:
                if abs((pa.Pick - trace.stats.theo_tp).total_seconds()) < rlim:
                    hr = axis2.scatter(
                        (pa.Pick - trace.stats.theo_tp).total_seconds(), n_trace, s=25, c='lime',
                        marker=emrk, edgecolors=ecol, alpha=.7, label=f'Auto-Theo ({event.EventID})', zorder=50)
                    # check other data point for that event not already in legend
                    if f'Auto-Theo ({event.EventID})' not in [plt.getp(hh, 'label') for hh in hrt]:
                        hrt.append(hr)
            if fas:
                if abs((sa.Pick - trace.stats.theo_ts).total_seconds()) < rlim:
                    axis2.scatter((sa.Pick - trace.stats.theo_ts).total_seconds(), n_trace, s=25, c='lime',
                                  marker=emrk, edgecolors=ecol, alpha=.7, linestyle='dotted', zorder=50)
    # axis limits
    axis1.set_ylim([0, n_trace + 1])
    axis2.set_ylim([0, n_trace + 1])
    # highlight catalogue event origin time
    axis1.plot([cevt.DateTime, cevt.DateTime], [0, n_trace + 1], color='red', alpha=.7)
    # station and pick numbers
    axis1.text(-.01, 1.01, f"N={n_trace}", ha='right', va='center', transform=axis1.transAxes)
    # replace y-axis tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace+1, 1))
    axis1.set_yticklabels([])
    axis1.set_ylim([0, n_trace+1])
    axis2.set_yticklabels([])
    # axis labels
    axis1.set_xlabel('Time', fontweight='bold')
    axis2.set_xlabel('\u0394t [s]', fontweight='bold')
    # EVENTS
    # plot catalogue event location
    hce = axis3.scatter(cevt.Longitude, cevt.Latitude,
                        s=100, c='red', marker='*', edgecolors='black', zorder=100, label='GSI')
    # plot automatic events time & location
    hae = []
    # loop over automatic events
    for k, event in aevt.iterrows():
        # assign event colour
        ecol = evt_colour[k]
        # plot event time marker in waveform view
        ha2, = axis1.plot([event.DateTime, event.DateTime], [0, n_trace + 1], ecol, alpha=.7, label=event.EventID)
        if event.EventID not in [plt.getp(hh, 'label') for hh in ha]:
            ha.append(ha2)
        # display automatic event ID in waveform view
        if tmin < event.DateTime < tmax:
            axis1.text(event.DateTime, n_trace + 3, f"{event.EventID}",
                       color=ecol, ha='center', va='center', fontsize=8, clip_on=False)
        # plot automatic event in local map
        h = axis3.scatter(event.Longitude, event.Latitude,
                          s=100, c=ecol, marker='*', edgecolors='black', label=event.EventID, zorder=10)
        hae.append(h)
        # plot automatic event in regional map
        if regional_map[2] < event.Longitude < regional_map[3] and regional_map[0] < event.Latitude < regional_map[1]:
            axis4.scatter(event.Longitude, event.Latitude, s=100, c=ecol, marker='*', edgecolors='black', zorder=10)
        # highlight recording stations in maps
        for _, pick in picks_tab[picks_tab.EventID.str.contains('lew')].iterrows():
            # match pick ID with stations in inventory
            st = sta_inv.select(network=pick.Network, station=pick.Station,
                                location=pick.Location, channel=pick.Channel)
            # ignore station if not found in inventory
            if not st:
                print(f'Missing station in inventory: {pick.Network}.{pick.Station}.{pick.Location}.{pick.Channel}')
                continue
            # plotting station
            if pick.Network == 'IS':
                axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                              s=25, c='magenta', marker='^', edgecolors=ecol, zorder=5)
            elif pick.Network == 'GE':
                axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                              s=25, c='cyan', marker='s', edgecolors=ecol, zorder=5)
            # connect station to event
            axis3.plot([event.Longitude, st.networks[0].stations[0].longitude],
                       [event.Latitude, st.networks[0].stations[0].latitude], color=ecol, alpha=.2, zorder=2)
            # connect station to event in regional map if event is not local
            if event.Longitude < local_map[2] or event.Longitude > local_map[3] or \
                    event.Latitude < local_map[0] or event.Latitude > local_map[1]:
                axis4.plot([event.Longitude, st.networks[0].stations[0].longitude],
                           [event.Latitude, st.networks[0].stations[0].latitude], color=ecol, alpha=.2, zorder=2)
    # waveforms legend
    hh = [hw, ht]
    if hc:
        hh.append(hc)
    if ha:
        hh = hh + ha
    axis1.legend(handles=hh, loc='lower left', fontsize=8).set_zorder(120)
    # map legend
    if hn2:
        hm = [hn1, hn2, hce]
    else:
        hm = [hn1, hce]
    if hae:
        hm = hm + hae
    hm.append(hf)
    hm.append(hq)
    axis3.legend(handles=hm, loc='lower right', fontsize=8).set_zorder(120)
    # residual plot legend
    h = []
    if hro:
        h.append(hro)
    if hrt:
        h = h + hrt
    if hrc:
        h = h + hrc
    if hro or hrt or hrc:
        axis2.legend(handles=h, loc='lower right', fontsize=8).set_zorder(120)
    # figure title
    tit = f"{datetime.strftime(cevt.DateTime, '%d/%m/%Y %H:%M:%S')} \u2013 {cevt.Depth:.2f} km \u2013 M{cevt.Mag:3.1f}"
    fig.suptitle(f"{tit}\n{cevt.Type} ({cevt.Author})", fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.01, right=.99, top=.92, wspace=.07)
    # show or save figure
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()
    return


def plot_auto_evt_sec(stream, evt_xml, time_win, fig_name):
    """
    :param stream: data streamer of waveforms to process
    :param evt_xml: xml container with all event data and picks
    :param time_win: half-length of time window as a datetime.timedelta() object
    :param fig_name: figure full file path
    :return: nothing
    """
    # geographic boundaries
    local_map = [29., 34., 33.5, 36.5]
    regional_map = [23., 43., 25., 45.]
    # retrieve station inventory
    sta_inv = Client('http://172.16.46.102:8181/').get_stations(
        network='IS,GE', channel='ENZ,HHZ,BHZ,SHZ', level='response')
    # regional events from EMSC's FDSN
    try:
        regi_emsc = Client('EMSC').get_events(
            starttime=UTCDateTime(evt_xml.events[0].preferred_origin().time.datetime - time_win),
            endtime=UTCDateTime(evt_xml.events[0].preferred_origin().time.datetime + time_win),
            minlatitude=regional_map[0], maxlatitude=regional_map[1], minlongitude=regional_map[2],
            maxlongitude=regional_map[3], minmagnitude=3.)
    except:
        regi_emsc = []
    # teleseismic events from EMSC's FDSN
    try:
        tele_emsc = Client('EMSC').get_events(
            starttime=UTCDateTime(evt_xml.events[0].preferred_origin().time.datetime - time_win),
            endtime=UTCDateTime(evt_xml.events[0].preferred_origin().time.datetime + time_win),
            minmagnitude=6.)
    except:
        tele_emsc = []
    # plotting mode
    mpl.use('Agg')
    # waveforms time window (min Tp and max Ts)
    tmin = min([tr.stats.theo_tp for tr in stream]) - timedelta(minutes=2)
    tmax = max([tr.stats.theo_tp for tr in stream]) + timedelta(minutes=2)
    # create figure & axes
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200,
                                       gridspec_kw={'width_ratios': [2, 1]})
    # AXIS 1: VELOCITY WAVEFORMS
    # show axis grids
    axis1.grid(which='both', axis='both')
    # set x-axis limits
    axis1.set_xlim([tmin, tmax])
    # set date format to x-axis tick labels
    axis1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # AXIS 2: LOCAL STATIONS & EVENT MAP
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=local_map[2], llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3], urcrnrlat=local_map[1], resolution='i', ax=axis2)
    # draw map
    m1.drawmapboundary(fill_color='none')
    # fill continents
    m1.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels=[left,right,top,bottom])
    m1.drawparallels(np.arange(m1.llcrnrlat, m1.urcrnrlat + 1, 2.), labels=[True, False, True, False])
    m1.drawmeridians(np.arange(m1.llcrnrlon, m1.urcrnrlon + 1, 2.), labels=[True, False, False, True])
    # faults
    flts_id = open('/home/sysop/.seiscomp/spatial/vector/ActiveFaults/activefaults.bna', 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    hf = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            hf, = axis2.plot(flt.lon, flt.lat, '.6', linewidth=.5, label='Faults', zorder=1)
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # quarries
    quar_id = open('/home/sysop/.seiscomp/spatial/vector/Quarries/quarries.bna', 'r')
    quar = quar_id.readlines()
    quar_id.close()
    hq = []
    for iii in range(len(quar)):
        if not re.search('"', quar[iii]):
            l_line = quar[iii].replace('\n', '').split(',')
            if l_line[0] != '' and l_line[1] != '':
                hq = axis2.scatter(float(l_line[0]), float(l_line[1]), s=25, c='blue', marker='s',
                                   edgecolors='black', label='Quarries', alpha=.1, zorder=2)
    # AXIS 3: INSET MAP FOR REGIONAL SETTINGS
    axis3 = inset_axes(axis2, '30%', '18%', loc='lower left')
    m2 = Basemap(projection='cyl', llcrnrlon=regional_map[2], llcrnrlat=regional_map[0],
                 urcrnrlon=regional_map[3], urcrnrlat=regional_map[1], resolution='l', ax=axis3)
    # draw map
    m2.drawmapboundary(fill_color='white')
    # fill continents
    m2.fillcontinents(color='0.8', lake_color='white')
    # plot GE station locations
    for station in sta_inv.networks[0].stations:
        if not (local_map[0] < station.latitude < local_map[1] and local_map[2] < station.longitude < local_map[3]):
            axis3.scatter(station.longitude, station.latitude,
                          s=25, color='cyan', marker='^', edgecolors='black', alpha=.2, zorder=5)
            axis3.text(station.longitude + .5, station.latitude + .1, station.code,
                       c='black', ha='left', va='bottom', fontsize=5, alpha=.2, clip_on=True)
    # highlight area of interest
    axis3.plot([local_map[2], local_map[2], local_map[3], local_map[3], local_map[2]],
               [local_map[0], local_map[1], local_map[1], local_map[0], local_map[0]], color='saddlebrown')
    # plot trigger location
    hce = axis2.scatter(evt_xml.events[0].preferred_origin().longitude, evt_xml.events[0].preferred_origin().latitude,
                        s=100, c='red', marker='*', edgecolors='black', label='Trigger', zorder=20)
    axis3.scatter(evt_xml.events[0].preferred_origin().longitude, evt_xml.events[0].preferred_origin().latitude,
                  s=100, c='red', marker='*', edgecolors='black', zorder=20)
    # plot EMSC events location
    her = []
    for event in regi_emsc:
        if tmin < event.preferred_origin().time.datetime < tmax:
            her = axis2.scatter(event.preferred_origin().longitude, event.preferred_origin().latitude, s=100,
                                c='orange', marker='*', edgecolors='black', alpha=.7, label='EMSC', zorder=10)
            axis3.scatter(event.preferred_origin().longitude, event.preferred_origin().latitude,
                          s=100, c='orange', marker='*', edgecolors='black', alpha=.7, zorder=10)
    # LOOP OVER TRACES
    # intialise counters
    n_trace = 0
    # initialise tables for waveform legend
    hw = []
    htp = []
    hts = []
    hap = []
    has = []
    herp = []
    hers = []
    hetp = []
    hets = []
    # initialise tables for map legend
    hn1 = []
    hn2 = []
    # initialise list of station labels
    stn_lbl = []
    for trace in stream:
        # station label for y-axis
        stn_lbl.append(f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}")
        # counter
        n_trace += 1
        # time vector
        tvec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]') + \
            np.datetime64(str(trace.stats.starttime)[:-1])
        # look for indexes closest to tmin and tmax for normalisation
        i1 = np.argmin(np.abs(tvec - np.datetime64(tmin)))
        i2 = np.argmin(np.abs(tvec - np.datetime64(tmax)))
        if i2 - i1 < 200:
            i1 = 0
            i2 = len(trace.data)
        # maximum amplitude for normalisation
        wmax = 2. * max(trace.data[i1:i2])
        # plot waveform
        hw, = axis1.plot(tvec, trace.data / wmax + n_trace, color='grey', alpha=.7, label='Velocity')
        # plot theoretical travel time
        if trace.stats.theo_tp:
            htp, = axis1.plot([trace.stats.theo_tp, trace.stats.theo_tp], [n_trace - 1, n_trace + 1],
                              color='lime', linestyle='dotted', label='Theoretical P pick')
        if trace.stats.theo_ts:
            hts, = axis1.plot([trace.stats.theo_ts, trace.stats.theo_ts], [n_trace - 1, n_trace + 1],
                              color='cyan', linestyle='dotted', label='Theoretical S pick')
        # plot station in map
        st = sta_inv.select(network=trace.stats.network, station=trace.stats.station, channel=trace.stats.channel)
        # use different symbols and colours for different network codes
        if trace.stats.network == 'IS':
            hn1 = axis2.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                s=25, c='magenta', marker='^', edgecolors='black', alpha=.2,
                                label=trace.stats.network, zorder=3)
            axis2.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude + 0.06,
                       st.networks[0].stations[0].code, color='black', ha='center', va='center',
                       fontsize=5, alpha=.2, clip_on=True)
        elif trace.stats.network == 'GE':
            hn2 = axis2.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude, s=25,
                                c='cyan', marker='^', edgecolors='black', alpha=.2, label=trace.stats.network, zorder=3)
            axis2.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude - 0.07,
                       st.networks[0].stations[0].code, color='black', ha='center', va='center',
                       fontsize=5, alpha=.2, clip_on=True)
        # TRIGGER PICKS
        for pick in evt_xml.events[0].picks:
            if pick.waveform_id.network_code == trace.stats.network \
                    and pick.waveform_id.station_code == trace.stats.station \
                    and pick.waveform_id.channel_code == trace.stats.channel:
                for arrival in evt_xml.events[0].preferred_origin().arrivals:
                    if arrival.pick_id.id == pick.resource_id.id:
                        # show picks
                        if arrival.phase == 'P':
                            hap, = axis1.plot([pick.time.datetime, pick.time.datetime], [n_trace - 1, n_trace + 1],
                                              color='purple', label=f'P pick')
                        elif arrival.phase == 'S':
                            has, = axis1.plot([pick.time.datetime, pick.time.datetime], [n_trace - 1, n_trace + 1],
                                              color='magenta', label=f'S pick')
                        # highlight station in map
                        if trace.stats.network == 'IS':
                            axis2.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                          s=25, c='magenta', marker='^', edgecolors='red', zorder=5)
                            axis2.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude + 0.06,
                                       st.networks[0].stations[0].code, color='red', ha='center', va='center',
                                       fontsize=5, clip_on=True)
                        elif trace.stats.network == 'GE':
                            axis2.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                          s=25, c='cyan', marker='^', edgecolors='red', zorder=5)
                            axis2.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude - 0.07,
                                       st.networks[0].stations[0].code, color='red', ha='center', va='center',
                                       fontsize=5, clip_on=True)
                        # connect station to event
                        axis2.plot([evt_xml.events[0].preferred_origin().longitude,
                                    st.networks[0].stations[0].longitude],
                                   [evt_xml.events[0].preferred_origin().latitude, st.networks[0].stations[0].latitude],
                                   color='red', alpha=.2, zorder=2)
    # axis limits
    axis1.set_ylim([0, n_trace + 1])
    # legend
    hh = [hw, htp, hts]
    if hap:
        hh.append(hap)
    if has:
        hh.append(has)
    if herp:
        hh.append(herp)
    if hers:
        hh.append(hers)
    if hetp:
        hh.append(hetp)
    if hets:
        hh.append(hets)
    axis1.legend(handles=hh, loc='upper right', fontsize=8)
    # show EMSC events origin time
    for event in regi_emsc:
        axis1.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime],
                   [0, n_trace + 1], color='orange', alpha=.7)
        if tmin < event.preferred_origin().time.datetime < tmax:
            axis1.text(event.preferred_origin().time.datetime, n_trace + 1.1,
                       f'{event.event_descriptions[0].text} M{event.preferred_magnitude().mag}',
                       color='orange', ha='center', va='bottom', fontsize=8)
    for event in tele_emsc:
        axis1.plot([event.preferred_origin().time.datetime, event.preferred_origin().time.datetime],
                   [0, n_trace + 1], color='green', alpha=.7)
        if tmin < event.preferred_origin().time.datetime < tmax:
            axis1.text(event.preferred_origin().time.datetime, n_trace + 1.1,
                       f'{event.event_descriptions[0].text} M{event.preferred_magnitude().mag}',
                       color='green', ha='center', va='bottom', fontsize=8)
    # show trigger origin time
    axis1.plot([evt_xml.events[0].preferred_origin().time.datetime, evt_xml.events[0].preferred_origin().time.datetime],
               [0, n_trace + 1], color='red', alpha=.7)
    # station and pick numbers
    axis1.text(-.01, 1.01, f"N={n_trace}", ha='right', va='center', transform=axis1.transAxes)
    # replace numerical tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace + 1, 1))
    axis1.set_yticklabels(stn_lbl, fontsize=5)
    # axis labels
    axis1.set_xlabel('Time', fontweight='bold')
    axis1.set_ylabel('Station', fontweight='bold')
    # map legend
    if hn2:
        hm = [hn1, hn2, hce]
    else:
        hm = [hn1, hce]
    if her:
        hm.append(her)
    hm.append(hf)
    hm.append(hq)
    axis2.legend(handles=hm, loc='upper left', fontsize=8)
    # look for magnitude
    if not evt_xml.events[0].preferred_magnitude():
        emag = np.nan
    else:
        emag = evt_xml.events[0].preferred_magnitude().mag
    # title string
    tit = f"{evt_xml.events[0].preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S')} \u2013 " \
          f"{evt_xml.events[0].preferred_origin().depth / 1000.:.2f} km \u2013 " \
          f"M{emag:3.1f}\n{len([a.phase for a in evt_xml.events[0].preferred_origin().arrivals if a.phase == 'P'])}" \
          f" P + {len([a.phase for a in evt_xml.events[0].preferred_origin().arrivals if a.phase == 'S'])} S " \
          f"({evt_xml.events[0].creation_info.author})"
    # display figure title
    fig.suptitle(tit, fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust subplots
    fig.subplots_adjust(left=.07, right=.97, top=.92, wspace=0)
    # save figure
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    plt.close()


def list_auto_events(year, month, day1, work_dir):
    """
    :param year: year to list automatic events for
    :param month: year to list automatic events for
    :param day1: starting day to list automatic events for
    :param work_dir: working directory
    :return: nothing
    """
    # FDSN client
    isn_client = Client('http://172.16.46.102:8181/')
    # read table of checked events
    full_tab = pd.read_csv(f'{work_dir}/autopicker_rt_checks_Dec22_Feb23.csv')
    # output DataFrame
    output_tab = pd.DataFrame({'DateTime': pd.Series(dtype='datetime64[ms]'),
                               'AutoID': pd.Series(dtype='string'), 'Detection': pd.Series(dtype='string'),
                               'AutoM': pd.Series(dtype='float64'), 'AutoT': pd.Series(dtype='datetime64[ms]'),
                               'AutoLat': pd.Series(dtype='float64'), 'AutoLon': pd.Series(dtype='float64'),
                               'AutoDep': pd.Series(dtype='float64'), 'AutoNumPicks': pd.Series(dtype='int'),
                               'CataID': pd.Series(dtype='string'), 'CataM': pd.Series(dtype='float64'),
                               'CataT': pd.Series(dtype='datetime64[ms]'), 'CataLat': pd.Series(dtype='float64'),
                               'CataLon': pd.Series(dtype='float64'), 'CataDep': pd.Series(dtype='float64'),
                               'CataNumPicks': pd.Series(dtype='int'), 'CataCat': pd.Series(dtype='string')})
    # month to loop over
    for day in range(day1, monthrange(year, month)[1] + 1):
        # transform EventIDs to dates
        dser = pd.concat([pd.to_datetime(full_tab.EventID[
                                             full_tab.EventID.str.contains('\\.')].str.slice(start=3, stop=-3),
                                         format='%Y%m%d%H%M'),
                          pd.to_datetime(full_tab.EventID[
                                             ~full_tab.EventID.str.contains('\\.')].str.slice(start=3),
                                         format='%Y%m%d%H%M')]).sort_values(ignore_index=True)
        # starting and ending times (one day)
        if year == 2022 and month == 12 and day == day1:
            beg = datetime.strptime(f'{year}-{month:02d}-{day:02d} 10:00:00', '%Y-%m-%d %H:%M:%S')
        else:
            beg = datetime.strptime(f'{year}-{month:02d}-{day:02d} 00:00:00', '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(f'{year}-{month:02d}-{day:02d} 00:00:00', '%Y-%m-%d %H:%M:%S') + timedelta(days=1)
        print(datetime.strftime(beg, '%d/%m/%Y'))
        if len(dser) != len(full_tab):
            print('Check DateTime extraction from catalogue event IDs')
            print(pd.DataFrame({'dser': dser[(dser >= beg) & (dser < end)],
                                'full_tab': full_tab[(dser >= beg) & (dser < end)].EventID}))
            exit()
        # all catalogue events for that day
        day_cat = full_tab[(dser >= beg) & (dser < end)].reset_index(drop=True)
        print(f" {len(day_cat)} daily catalogue events")
        # list automatic events
        os.system(f'scevtls --begin "{beg}" --end "{end}" > {wdir}/{year}-{month:02d}/list.tmp')
        with open(f'{wdir}/{year}-{month:02d}/list.tmp') as fid:
            day_auto = fid.read().splitlines()
        os.remove(f'{work_dir}/{year}-{month:02d}/list.tmp')
        print(f' {len(day_auto)} daily automatic events')
        # catalogue events to be processed (Detection is '?')
        proc_cat = day_cat[day_cat.Detection == '?'].reset_index(drop=True)
        # catalogue events needing more checks (Category contains '?')
        check_cat = day_cat[day_cat.Category.str.contains('\\?', na=False)].reset_index(drop=True)
        # catalogue events to consider (not to be processed or checked and not missed automatic)
        good_cat = day_cat[(day_cat.Detection != '?') & ~day_cat.Category.str.contains('\\?', na=False) &
                           ~day_cat.EventID.str.contains('not', case=True, na=False)].reset_index(drop=True)
        # number of missed catalogue events (AutoID empty)
        miss_cat = good_cat[pd.isna(good_cat.AutoID)].reset_index(drop=True)
        # number of true automatic events (AutoID not empty)
        true_auto = good_cat[~pd.isna(good_cat.AutoID)].reset_index(drop=True)
        # number of missed automatic events (EventID contains 'not')
        miss_auto = day_cat[day_cat.EventID.str.contains('not', case=True, na=False)].reset_index(drop=True)
        if len(day_cat) != (len(proc_cat) + len(check_cat) + len(miss_cat) + len(true_auto) + len(miss_auto)):
            print(' Problem with event categories:')
            print(f"  {len(proc_cat)} catalogue events to process")
            print(f"  {len(check_cat)} catalogue events to check")
            print(f"  {len(miss_cat)} missed catalogue events")
            print(f"  {len(true_auto)} true automatic events")
            print(f"  {len(miss_auto)} missed automatic events")
            exit()
        print(f' {len(true_auto)} true automatic detections')
        # flatten out list of missed automatic events
        not_false = [item for sublist in [evt.split(',') for evt in miss_auto.AutoID.to_list()] for item in sublist]
        print(f' {len(not_false)} missed automatic detections')
        # list non-false automatic events to remove from list
        auto_false = day_auto[:]
        to_del = []
        for auto in auto_false:
            if auto in not_false or auto in list(np.unique(np.array(sorted([
                    item for sublist in [evt.split(',') for evt in true_auto.AutoID.to_list()]
                    for item in sublist])))) or auto in proc_cat.AutoID.to_list() or auto in check_cat.AutoID.to_list():
                to_del.append(auto)
        # remove non-false events
        for auto in to_del:
            auto_false.remove(auto)
        print(f' {len(auto_false)} false automatic detections')
        # all automatic events from the .csv file (True detections + True detections to check +
        # unprocessed events with probable automat match + not-False detections)
        list1 = true_auto.AutoID.to_list() + check_cat[check_cat.Detection == 'T'].AutoID.to_list() + \
            proc_cat[~pd.isna(proc_cat.AutoID)].AutoID.to_list() + not_false
        # flattening, sorting and removing of duplicates in list
        list1 = list(np.unique(np.array(sorted([
            item for sublist in [evt.split(',') for evt in list1] for item in sublist]))))
        # all automatic events from the .csv file also found on 'jdev' event list
        list2 = list(np.unique(np.array(sorted(
            list(set(pd.Series(day_auto)).intersection(set([
                item for sublist in [evt.split(',') for evt in
                                     day_cat.AutoID[~pd.isna(day_cat.AutoID)]] for item in sublist])))))))
        # check all AutoIDs were typed in correctly (i.e. if not all are found)
        if list1 != list2:
            print(' !! Missing true events in list (typo??) !!')
            print(list1)
            print(list2)
            exit()
        # check False directory exists
        if os.path.exists(f'{work_dir}/{year}-{month:02d}/false') == 0:
            os.mkdir(f'{work_dir}/{year}-{month:02d}/false')
        # loop over True detections
        for _, true in true_auto.iterrows():
            # systematically taking first automatic event if several
            if ',' in true.AutoID:
                true.AutoID = true.AutoID.split(',')[0]
            # read event .xml file
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/{true.AutoID}.xml') == 0:
                os.system(f'scxmldump -E {true.AutoID} -PAMFf -o {work_dir}/{year}-{month:02d}/false/{true.AutoID}.xml')
            auto = read_events(f'{work_dir}/{year}-{month:02d}/{true.AutoID}.xml')
            # check preferred origin exists
            if not auto.events[0].preferred_origin():
                print(f'{true.AutoID}: No preferred origin')
                continue
            aori = auto.events[0].preferred_origin()
            # check preferred magnitude exists
            if not auto.events[0].preferred_magnitude():
                amag = np.nan
            else:
                amag = auto.events[0].preferred_magnitude().mag
            # retrieve catalogue event parameters
            cata = isn_client.get_events(eventid=true.EventID, includearrivals=True)
            # 'not existing' and 'not locatable' events
            if cata.events[0].event_type == 'not existing' or cata.events[0].event_type == 'not relocatable':
                print(f'{true.EventID}: Ignored')
                continue
            # check preferred origin exists
            if not cata.events[0].preferred_origin():
                print(f'{true.EventID}: No preferred origin')
                continue
            cori = cata.events[0].preferred_origin()
            # add True or Missed detection to output DataFrame
            output_tab.loc[output_tab.shape[0]] = [cori.time.datetime, true.AutoID, true.Detection, amag,
                                                   aori.time.datetime, aori.latitude, aori.longitude,
                                                   aori.depth / 1000., len(aori.arrivals), true.EventID, true.Magnitude,
                                                   cori.time.datetime, cori.latitude, cori.longitude,
                                                   cori.depth / 1000., len(cori.arrivals), true.Category]
        # loop over Missed detections
        for _, missed in miss_cat.iterrows():
            # retrieve catalogue event parameters
            cata = isn_client.get_events(eventid=missed.EventID, includearrivals=True)
            # check preferred origin exists
            if not cata.events[0].preferred_origin():
                print(f'{missed.EventID}: No preferred origin')
                continue
            cori = cata.events[0].preferred_origin()
            # add Missed detection to output DataFrame
            output_tab.loc[output_tab.shape[0]] = [cori.time.datetime, '', 'M', np.nan, np.datetime64('NaT'), np.nan,
                                                   np.nan, np.nan, np.nan, missed.EventID, missed.Magnitude,
                                                   cori.time.datetime, cori.latitude, cori.longitude,
                                                   cori.depth / 1000., len(cori.arrivals), missed.Category]
        # loop over False detections
        for event in auto_false:
            # read event .xml file
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/false/{event}.xml') == 0:
                if os.path.exists(f'{work_dir}/{year}-{month:02d}/{event}.xml') != 0:
                    os.rename(f'{work_dir}/{year}-{month:02d}/{event}.xml',
                              f'{work_dir}/{year}-{month:02d}/false/{event}.xml')
                else:
                    os.system(f'scxmldump -E {event} -PAMFf -o {work_dir}/{year}-{month:02d}/false/{event}.xml')
            auto = read_events(f'{work_dir}/{year}-{month:02d}/false/{event}.xml')
            # check preferred origin exists
            if not auto.events[0].preferred_origin():
                print(f'{event}: No preferred origin')
                continue
            aori = auto.events[0].preferred_origin()
            # check preferred magnitude exists
            if not auto.events[0].preferred_magnitude():
                amag = np.nan
            else:
                amag = auto.events[0].preferred_magnitude().mag
            # add True or Missed detection to output DataFrame
            output_tab.loc[output_tab.shape[0]] = [aori.time.datetime, event, 'F', amag, aori.time.datetime,
                                                   aori.latitude, aori.longitude, aori.depth / 1000.,
                                                   len(aori.arrivals),
                                                   '', np.nan, np.datetime64('NaT'), np.nan, np.nan, np.nan, np.nan, '']
        # loop over non-False detections (automatic True event detections without catalogue match)
        for event in not_false:
            # read event .xml file
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/false/{event}.xml') == 0:
                if os.path.exists(f'{work_dir}/{year}-{month:02d}/{event}.xml') != 0:
                    os.rename(f'{work_dir}/{year}-{month:02d}/{event}.xml',
                              f'{work_dir}/{year}-{month:02d}/false/{event}.xml')
                else:
                    os.system(f'scxmldump -E {event} -PAMFf -o {work_dir}/{year}-{month:02d}/false/{event}.xml')
            auto = read_events(f'{work_dir}/{year}-{month:02d}/false/{event}.xml')
            # check preferred origin exists
            if not auto.events[0].preferred_origin():
                print(f'{event}: No preferred origin')
                continue
            aori = auto.events[0].preferred_origin()
            # check preferred magnitude exists
            if not auto.events[0].preferred_magnitude():
                amag = np.nan
            else:
                amag = auto.events[0].preferred_magnitude().mag
            # add True or Missed detection to output DataFrame
            output_tab.loc[output_tab.shape[0]] = [aori.time.datetime, event, 'N', amag, aori.time.datetime,
                                                   aori.latitude, aori.longitude, aori.depth / 1000.,
                                                   len(aori.arrivals), '', np.nan, np.datetime64('NaT'),
                                                   np.nan, np.nan, np.nan, np.nan, '']
    # sort according to time
    output_tab = output_tab.set_index('DateTime').sort_index()
    # write output DataFrame
    output_tab.to_csv(f'{work_dir}/{year}-{month:02d}/summary.csv', float_format='%.4f')


def check_missed_events(year, month, day1, work_dir):
    """
    :param year: year to list automatic false events for
    :param month: year to list automatic false events for
    :param day1: starting day to list automatic false events for
    :param work_dir: working directory
    :return: nothing
    """
    # FDSN client
    isn_client = Client('http://172.16.46.102:8181/')
    seisdb = GiiPhaseDB("172.16.46.102")
    seisdb.openDB()
    # retrieve station inventory
    isn_inv = isn_client.get_stations(network='IS,GE', channel='ENZ,HHZ,BHZ,SHZ', level='response')
    # logging file
    if os.path.exists(f"{work_dir}/{year}-{month:02d}/missed_{year}-{month:02d}.log") != 0:
        os.remove(f"{work_dir}/{year}-{month:02d}/missed_{year}-{month:02d}.log")
    logger = my_custom_logger(f"{work_dir}/{year}-{month:02d}/missed_{year}-{month:02d}.log", level=logging.DEBUG)
    # loop over each day because in February too much events for FDSN server
    for day in range(day1, monthrange(year, month)[1] + 1):
        # catalogue events
        try:
            if year == 2022 and month == 12 and day == day1:
                beg = datetime.strptime(f'{year}-{month:02d}-{day:02d} 10:00:00', '%Y-%m-%d %H:%M:%S')
            else:
                beg = datetime.strptime(f'{year}-{month:02d}-{day:02d} 00:00:00', '%Y-%m-%d %H:%M:%S')
            evt_lst = isn_client.get_events(starttime=UTCDateTime(beg), endtime=UTCDateTime(datetime.strptime(
                    f'{year}-{month:02d}-{day:02d} 00:00:00', '%Y-%m-%d %H:%M:%S')
                    + timedelta(days=1)), includearrivals=True, orderby='time-asc')
        except:
            logger.warning(f"No events on FDSN for {day:02d}/{month:02d}/{year}")
            continue
        # loop over listed events
        for evt in evt_lst:
            # EXTRACT CATALOGUE EVENT DATA
            # event ID
            evid = evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')
            print(evid)
            # # to investigate a specific event
            # if evid != 'gsi20230613011551':
            #     continue
            # skipping non-"gsi" catalogue events
            if not re.search('gsi', evid):
                logger.info(f'{evid}: Ignored (event ID)')
                continue
            # skipping "not existing" catalogue events
            # if not evt.event_type or evt.event_type is None or evt.event_type == 'not existing' \
            #         or evt.event_type == 'not relocatable' or evt.event_type == 'other event':
            if evt.event_type == 'not existing' or evt.event_type == 'not relocatable' \
                    or evt.event_type == 'other event':
                logger.info(f"{evid}: Ignored ({evt.event_type if evt.event_type else 'event type'})")
                continue
            # skipping catalogue events witout preferred origin
            if not evt.preferred_origin():
                logger.warning(f'{evid}: Ignored (no origin)')
                continue
            # case depth >800 km
            if evt.preferred_origin().depth > 800000.:
                logger.warning(f'{evid}: event depth >800 km (result divided by 1000)')
                evt.preferred_origin().depth = evt.preferred_origin().depth/1000.
            # skipping catalogue events without picks
            if not evt.picks:
                logger.warning(f'{evid}: Ignored (no picks)')
                continue
            # determine event zone (local/regional/teleseismic)
            zevt = get_zone_id(seisdb.db, evt.preferred_origin().latitude, evt.preferred_origin().longitude)
            # skip teleseismic catalogue events
            if zevt is None or zevt == 'NNT':
                logger.info(f'{evid}: Ignored (teleseismic)')
                continue
            logger.info(f"{evid}:"
                        f" {datetime.strftime(evt.preferred_origin().time.datetime, '%d/%m/%Y %H:%M:%S.%f')} "
                        f"(M{(evt.preferred_magnitude().mag if evt.preferred_magnitude() else np.nan):3.1f} "
                        f"{evt.event_type})")
            # skip catalogue event if figure already exists
            if os.path.exists(f"{work_dir}/{year}-{month:02d}/{evid}.png") != 0:
                continue
            # SEARCH AUTOMATIC EVENTS DATA
            # listing automatic events on the local system within time window +/- [ewin] minutes around catalogue OT
            os.system(f"scevtls "
                      f"--begin '{datetime.strftime(evt.preferred_origin().time.datetime - ewin, '%Y-%m-%d %H:%M:%S')}'"
                      f" --end '{datetime.strftime(evt.preferred_origin().time.datetime + ewin, '%Y-%m-%d %H:%M:%S')}'"
                      f" |grep lew202 > {work_dir}/{year}-{month:02d}/list.tmp")
            # reading automatic events list
            with open(f'{work_dir}/{year}-{month:02d}/list.tmp') as f:
                levt = f.read().splitlines()
            # deleting temporary file
            os.remove(f'{work_dir}/{year}-{month:02d}/list.tmp')
            if levt and levt[0] == '':
                levt = []
            # PREPARING WAVEFORMS
            mfile = f"{work_dir}/{year}-{month:02d}/{evid}.mseed"
            if os.path.exists(mfile) == 0:
                # longer time window for triggers > 1,200 km from center of ISN
                if gdist.distance((31.5, 35.), (evt.preferred_origin().latitude,
                                                evt.preferred_origin().longitude)).km > 1200.:
                    win = 2. * wwin
                else:
                    win = wwin
                get_waveforms_deci(mfile, evt.preferred_origin().time.datetime, win, ntw, chn, adir)
            # filtering waveforms and adding theoretical arrivals, event-station distance, ...
            isn_traces = process_waveforms(mfile, fpar, evt, vel_mod, isn_inv)
            # sort according to event-station distance
            isn_traces.sort(['distance'], reverse=True)
            # PREPARING PICKS TABLE
            # calling function to prepare events and picks tables
            ptab, etab = prepare_tables(isn_traces, evt, levt, work_dir)
            # ptab.to_csv(f'{wdir}/ptab.csv', float_format='%.4f', index=False)
            # PLOT FIGURE
            plot_cata_evt_sec(isn_traces, etab, ptab, isn_inv, logger, f"{work_dir}/{year}-{month:02d}/{evid}.png")
    seisdb.closeDB()


def check_false_events(year, month, day1, work_dir):
    """
    :param year: year to list automatic false events for
    :param month: year to list automatic false events for
    :param day1: starting day to list automatic false events for
    :param work_dir: working directory
    :return: nothing
    """
    # retrieve station inventory
    isn_inv = Client('http://172.16.46.102:8181/').get_stations(
        network='IS,GE', channel='ENZ,HHZ,BHZ,SHZ', level='response')
    # read false automatic events table
    stab = pd.read_csv(f'{work_dir}/{year}-{month:02d}/summary.csv')
    # loop over each day because in February too much events for FDSN server
    for day in range(day1, monthrange(year, month)[1] + 1):
        xtab = stab[(stab.Detection != 'T') &
                    (pd.to_datetime(stab.AutoT) >= datetime.strptime(
                        f'{year}/{month:02d}/{day:02d} 00:00:00', '%Y/%m/%d %H:%M:%S')) &
                    (pd.to_datetime(stab.AutoT) < (datetime.strptime(
                        f'{year}/{month:02d}/{day:02d} 00:00:00', '%Y/%m/%d %H:%M:%S')) + timedelta(days=1))]
        for _, frow in xtab.iterrows():
            # read .xml event file
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.xml') == 0:
                if os.path.exists(f'{work_dir}/{year}-{month:02d}/{frow.AutoID}.xml') != 0:
                    os.rename(f'{work_dir}/{year}-{month:02d}/{frow.AutoID}.xml',
                              f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.xml')
                else:
                    os.system(f'scxmldump -E {frow.AutoID} -PAMFf '
                              f'-o {work_dir}/{year}-{month:02d}/false/{frow.AutoID}.xml')
            axml = read_events(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.xml')
            # in case of undefined magnitude
            if not axml.events[0].preferred_magnitude():
                mag = np.nan
            else:
                mag = axml.events[0].preferred_magnitude().mag
            # display event info
            print(f"{axml.events[0].resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')}: "
                  f"{datetime.strftime(axml.events[0].preferred_origin().time.datetime, '%d/%m/%Y %H:%M:%S.%f')} "
                  f"(M{mag:3.1f})")
            # skip if figure exists
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.png') != 0:
                print(' Figure already exists')
                print()
                continue
            # time window depends on distance (longer time window for triggers > 1,200 km from center of ISN)
            if gdist.distance((31.5, 35.), (axml.events[0].preferred_origin().latitude,
                                            axml.events[0].preferred_origin().longitude)).km > 1200.:
                win = 2. * wwin
            else:
                win = wwin
            # retrieve .mseed waveforms if necessary
            if os.path.exists(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.mseed') == 0:
                get_waveforms_deci(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.mseed',
                                   axml.events[0].preferred_origin().time.datetime, win, ntw, '(B|H|E|S)(H|N)Z', adir)
            # retrieve processed waveforms
            isn_traces = process_waveforms(f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.mseed',
                                           fpar, axml.events[0], vel_mod, isn_inv, True)
            # sort according to station latitude
            isn_traces.sort(['sta_lat'])
            # plot figure
            plot_auto_evt_sec(isn_traces, axml, win, f'{work_dir}/{year}-{month:02d}/false/{frow.AutoID}.png')
            print()


def plot_maps(work_dir):
    """
    :param work_dir: working directory
    :return: nothing
    """
    # lower magnitude limit
    mlo = 0.
    # upper magnitude limit
    mup = 3.5
    # geographical boundaries
    local_map = [29., 34., 33., 37.]
    # retrieve station inventory
    isn_inv = Client('http://172.16.46.102:8181/').get_stations(
        network='IS,GE', channel='ENZ,HHZ,BHZ,SHZ', level='response')
    # load monthly summary tables
    tab1 = pd.read_csv(f'{work_dir}/2022-12/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    tab2 = pd.read_csv(f'{work_dir}/2023-01/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    tab3 = pd.read_csv(f'{work_dir}/2023-02/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    # add-up tables
    tab0 = pd.concat([tab1, tab2, tab3], ignore_index=True)
    if mlo != 0:
        # keep catalogue EQs (i.e. True or Missed detections) with magnitude M>[mlo] and within monitoring area
        tab1 = tab0[((tab0.Detection == 'T') | (tab0.Detection == 'M')) & (tab0.CataM >= mlo) & (tab0.CataCat == 'EQ')
                    & (tab0.CataLat >= local_map[0]) & (tab0.CataLat <= local_map[1])
                    & (tab0.CataLon >= local_map[2]) & (tab0.CataLon <= local_map[3])].reset_index(drop=True)
        # keep False events with magnitude M>[mlo] and within monitoring area
        tab2 = tab0[(tab0.Detection == 'F') & (tab0.AutoM >= mlo)
                    & (tab0.AutoLat >= local_map[0]) & (tab0.AutoLat <= local_map[1])
                    & (tab0.AutoLon >= local_map[2]) & (tab0.AutoLon <= local_map[3])].reset_index(drop=True)
        tab = pd.concat([tab1, tab2], ignore_index=True).sort_values(by='DateTime')
    else:
        # all True, Missed and False events (not not-False events)
        tab = tab0[tab0.Detection != 'N'].reset_index(drop=True)
    # PLOT RESULTS
    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 9), dpi=200)
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=local_map[2] + .5, llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3] - .5, urcrnrlat=local_map[1], resolution='i', ax=ax1)
    m2 = Basemap(projection='cyl', llcrnrlon=local_map[2] + .5, llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3] - .5, urcrnrlat=local_map[1], resolution='i', ax=ax2)
    m3 = Basemap(projection='cyl', llcrnrlon=local_map[2] + .5, llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3] - .5, urcrnrlat=local_map[1], resolution='i', ax=ax3)
    # draw map
    m1.drawmapboundary(fill_color='none')
    m2.drawmapboundary(fill_color='none')
    m3.drawmapboundary(fill_color='none')
    # fill continents
    m1.fillcontinents(color='0.8', lake_color='white')
    m2.fillcontinents(color='0.8', lake_color='white')
    m3.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels=[left,right,top,bottom])
    m1.drawparallels(np.arange(m1.llcrnrlat, m1.urcrnrlat + 1, 1.), labels=[False, True, True, False], fontsize=5)
    m1.drawmeridians(np.arange(m1.llcrnrlon, m1.urcrnrlon + 1, 1.), labels=[True, False, False, True], fontsize=5)
    m2.drawparallels(np.arange(m2.llcrnrlat, m2.urcrnrlat + 1, 1.), labels=[False, True, True, False], fontsize=5)
    m2.drawmeridians(np.arange(m2.llcrnrlon, m2.urcrnrlon + 1, 1.), labels=[True, False, False, True], fontsize=5)
    m3.drawparallels(np.arange(m3.llcrnrlat, m3.urcrnrlat + 1, 1.), labels=[False, True, True, False], fontsize=5)
    m3.drawmeridians(np.arange(m3.llcrnrlon, m3.urcrnrlon + 1, 1.), labels=[True, False, False, True], fontsize=5)
    # faults
    flts_id = open(f"/home/{os.environ['LOGNAME']}/.seiscomp/bna/ActiveFaults/activefaults.bna", 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            ax1.plot(flt.lon, flt.lat, '.6', linewidth=.5)
            ax2.plot(flt.lon, flt.lat, '.6', linewidth=.5)
            ax3.plot(flt.lon, flt.lat, '.6', linewidth=.5)
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # LOCAL MAP TRUE EVENTS
    # ISN stations
    h10 = ax1.scatter([sta.longitude for net in isn_inv.networks for sta in net.stations],
                      [sta.latitude for net in isn_inv.networks for sta in net.stations],
                      s=9, c='blue', marker='^', edgecolors='none', alpha=.2, zorder=50, label='ISN')
    # compute distance between catalogue and automatic locations
    dist = []
    for _, trow in tab[tab.Detection == 'T'].iterrows():
        dis = gdist.distance((trow.CataLat, trow.CataLon), (trow.AutoLat, trow.AutoLon))
        dist.append(dis.km)
    # local true catalogue events colored according to location error
    h11 = ax1.scatter(tab[tab.Detection == 'T'].CataLon, tab[tab.Detection == 'T'].CataLat,
                      s=9, c=dist, cmap='Greens', alpha=.7, edgecolors='none', vmin=0, vmax=400,
                      zorder=100, label='GSI')
    # highlight M>3.5 events
    h12 = ax1.scatter(tab[(tab.Detection == 'T') & (tab.CataM >= mup)].CataLon,
                      tab[(tab.Detection == 'T') & (tab.CataM >= mup)].CataLat,
                      s=9, c='none', edgecolors='magenta', linewidth=.5, zorder=101, label=f'M>{mup}')
    # colour bar
    cbar = plt.colorbar(h11, ax=ax1, location='left', orientation='vertical',
                        label='Location error [km]', fraction=.05, pad=.05)
    cbar.ax.tick_params(labelsize=8)
    # title
    if mlo != 0:
        ax1.set_title(f"True M>{mlo} ({sum(tab.Detection == 'T')})", fontsize=8, color='green')
    else:
        ax1.set_title(f"True ({sum(tab.Detection == 'T')})", fontsize=8, color='green')
    # LOCAL MAP MISSED EVENTS
    # ISN stations
    h20 = ax2.scatter([sta.longitude for net in isn_inv.networks for sta in net.stations],
                      [sta.latitude for net in isn_inv.networks for sta in net.stations],
                      s=9, c='blue', marker='^', edgecolors='none', alpha=.2, zorder=50, label='ISN')
    # local missed catalogue events
    h21 = ax2.scatter(tab[tab.Detection == 'M'].CataLon, tab[tab.Detection == 'M'].CataLat,
                      s=9, c='red', edgecolors='none', alpha=.5, zorder=100, label='GSI')
    # highlight M>[mup] events
    h22 = ax2.scatter(tab[(tab.Detection == 'M') & (tab.CataM >= mup)].CataLon,
                      tab[(tab.Detection == 'M') & (tab.CataM >= mup)].CataLat,
                      s=9, c='none', edgecolors='magenta', linewidth=.5, zorder=101, label=f'M>{mup}')
    # title
    if mlo != 0:
        ax2.set_title(f"Missed M>{mlo} ({sum(tab.Detection == 'M')})", fontsize=8, color='red')
    else:
        ax2.set_title(f"Missed ({sum(tab.Detection == 'M')})", fontsize=8, color='red')
    # LOCAL MAP FALSE EVENTS
    # ISN stations
    h30 = ax3.scatter([sta.longitude for net in isn_inv.networks for sta in net.stations],
                      [sta.latitude for net in isn_inv.networks for sta in net.stations],
                      s=9, c='blue', marker='^', edgecolors='none', alpha=.2, zorder=50, label='ISN')
    # local False catalogue events
    h31 = ax3.scatter(tab[tab.Detection == 'F'].AutoLon, tab[tab.Detection == 'F'].AutoLat,
                      s=9, c='orange', edgecolors='none', alpha=.5, zorder=100, label='Auto')
    # highlight M>[mup] events
    h32 = ax3.scatter(tab[(tab.Detection == 'F') & (tab.AutoM >= mup)].AutoLon,
                      tab[(tab.Detection == 'F') & (tab.AutoM >= mup)].AutoLat,
                      s=9, c='none', edgecolors='magenta', linewidth=.5, zorder=101, label=f'M>{mup}')
    # title
    if mlo != 0:
        ax3.set_title(f"False M>{mlo} ({sum(tab.Detection == 'F')})", fontsize=8, color='orange')
    else:
        ax3.set_title(f"False ({sum(tab.Detection == 'F')})", fontsize=8, color='orange')
    # COMPLETENESS SCORES
    s = sum(tab.Detection == 'T') / (sum(tab.Detection == 'T') +
                                     sum(tab.Detection == 'M') + sum(tab.Detection == 'F')) * 100.
    sm = sum(tab[(tab.Detection == 'T') & ~pd.isna(tab.CataM)].CataM) / \
            (sum(tab[(tab.Detection == 'T') & ~pd.isna(tab.CataM)].CataM) +
             sum(tab[(tab.Detection == 'M') & ~pd.isna(tab.CataM)].CataM) +
             sum(tab[(tab.Detection == 'F') & ~pd.isna(tab.AutoM)].AutoM)) * 100.
    fig.suptitle(f"$s = {s:.1f}\\%$\n$s_M = {sm:.1f}\\%$", fontweight='bold', fontsize=8)
    # legend
    h1 = [h10, h11]
    if h12:
        h1.append(h12)
    h2 = [h20, h21]
    if h22:
        h2.append(h22)
    h3 = [h30, h31]
    if h32:
        h3.append(h32)
    ax1.legend(handles=h1, loc='lower right', fontsize=8)
    ax2.legend(handles=h2, loc='lower right', fontsize=8)
    ax3.legend(handles=h3, loc='lower right', fontsize=8)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.07, right=.95, wspace=.05)
    # show figure
    plt.show()


def plot_histograms(work_dir):
    """
    :param work_dir: working directory
    :return: nothing
    """
    # magnitude limit
    mlo = 2.
    # local geographic boundaries
    local_map = [29., 34., 33., 37.]
    # load monthly summary tables
    tab1 = pd.read_csv(f'{work_dir}/2022-12/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    tab2 = pd.read_csv(f'{work_dir}/2023-01/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    tab3 = pd.read_csv(f'{work_dir}/2023-02/summary.csv', parse_dates=['DateTime', 'AutoT', 'CataT'])
    # add-up tables
    tab = pd.concat([tab1, tab2, tab3], ignore_index=True)
    if mlo != 0:
        # keep catalogue EQs (i.e. True or Missed detections) with magnitude M>[mlo] and within monitoring area
        tab1 = tab[((tab.Detection == 'T') | (tab.Detection == 'M')) & (tab.CataM >= mlo) & (tab.CataCat == 'EQ')
                   & (tab.CataLat >= local_map[0]) & (tab.CataLat <= local_map[1])
                   & (tab.CataLon >= local_map[2]) & (tab.CataLon <= local_map[3])].reset_index(drop=True)
        # keep False events with magnitude M>[mlo] and within monitoring area
        tab2 = tab[(tab.Detection == 'F') & (tab.AutoM >= mlo)
                   & (tab.AutoLat >= local_map[0]) & (tab.AutoLat <= local_map[1])
                   & (tab.AutoLon >= local_map[2]) & (tab.AutoLon <= local_map[3])].reset_index(drop=True)
        tab = pd.concat([tab1, tab2], ignore_index=True).sort_values(by='DateTime')
    else:
        # all True, Missed and False events (not not-False events)
        tab = tab[tab.Detection != 'N'].reset_index(drop=True)
    # create new table for daily detection counts
    btab = tab[['DateTime', 'Detection', 'CataID']]
    # add column for date only
    btab = btab.assign(Date=pd.Series([''] * len(btab), dtype='datetime64[ms]'))
    btab.Date = btab['DateTime'].dt.date
    # get daily counts for each detection type, and catalogue
    ddat = pd.date_range('2022-12-27', '2023-02-28')
    true = btab[['Date', 'Detection']][btab.Detection == 'T'].groupby(['Date'], dropna=False). \
        count().reindex(ddat, fill_value=0.).rename(columns={"Detection": "True"})
    missed = btab[['Date', 'Detection']][btab.Detection == 'M'].groupby(['Date'], dropna=False) \
        .count().reindex(ddat, fill_value=0.).rename(columns={"Detection": "Missed"})
    false = btab[['Date', 'Detection']][btab.Detection == 'F'].groupby(['Date'], dropna=False) \
        .count().reindex(ddat, fill_value=0.).rename(columns={"Detection": "False"})
    cata = btab[['Date', 'CataID']][~pd.isna(btab.CataID)].groupby(['Date'], dropna=False) \
        .count().reindex(ddat, fill_value=0.).rename(columns={"CataID": "Catalogue"})
    total = btab[['Date', 'Detection']].groupby(['Date'], dropna=False) \
        .count().rename(columns={"Detection": "Total"})
    # merging all tables
    detec = cata.join(true).join(missed).join(false).join(total)
    # # convert to percents
    # detec['Catalogue'] = detec['Catalogue'] / detec['Total'] * 100.
    # detec['True'] = detec['True'] / detec['Total'] * 100.
    # detec['Missed'] = detec['Missed'] / detec['Total'] * 100.
    # detec['False'] = detec['False'] / detec['Total'] * 100.
    # figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(18, 9), dpi=200)
    # bar plot
    h1 = ax1.bar(x=detec.index, height=detec['True'], color='green', label='True')
    h2 = ax1.bar(x=detec.index, height=detec.Missed, bottom=detec['True'], color='red', label='Missed')
    h3 = ax1.bar(x=detec.index, height=detec['False'],
                 bottom=(detec.Missed + detec['True']), color='orange', label='False')
    h4 = ax1.bar(x=detec.index, height=detec.Catalogue, ec='blue', fc='none', label='Catalogue')
    # legend
    if mlo != 0:
        ax1.legend(handles=[h1, h2, h3, h4], title=f'M>{mlo}', loc='upper left', fontsize=8)
    else:
        ax1.legend(handles=[h1, h2, h3, h4], loc='upper left', fontsize=8)
    # rotate labels
    plt.xticks(rotation=45, ha='right')
    # adjust and show figure
    fig.subplots_adjust(left=.05, right=.98, wspace=.05, bottom=.17, top=.99)
    plt.show()


def plot_picks_histograms(detec_type, phase, work_dir):
    """
    :param detec_type: detection type to visualise; one-character string: T/F/C (True/False/Catalogue)
    :param phase: seismic phase of interest (P/S)
    :param work_dir: working directory
    :return: nothing
    """
    # read picks table
    if os.path.exists(f'{work_dir}/2022-12/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2022, 12, work_dir)
    tab1 = pd.read_csv(f'{work_dir}/2022-12/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    if os.path.exists(f'{work_dir}/2023-01/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2023, 1, work_dir)
    tab2 = pd.read_csv(f'{work_dir}/2023-01/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    if os.path.exists(f'{work_dir}/2023-02/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2023, 2, work_dir)
    tab3 = pd.read_csv(f'{work_dir}/2023-02/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    # add-up tables
    output_tab = pd.concat([tab1, tab2, tab3], ignore_index=True).set_index('EventID')
    output_tab = output_tab[~pd.isna(output_tab.SNR) & (output_tab.SNR != 0) & ~output_tab.Stream.str.contains('KO')
                            & (output_tab.Stream.str.contains('ENZ') | output_tab.Stream.str.contains('BHZ')
                               | output_tab.Stream.str.contains('HHZ') | output_tab.Stream.str.contains('SHZ'))]
    # prepare variable for station histogram
    var1 = output_tab.value_counts('Stream')
    # prepare variable for number of picks histogram
    var2 = output_tab[['Stream']].groupby('EventID').count().groupby('Stream').value_counts()
    # max. number of picks for station selection
    if phase == 'P' and detec_type == 'C':
        nmax = 250
    elif phase == 'P' and detec_type == 'F':
        nmax = 500
    elif phase == 'S' and detec_type == 'F':
        nmax = 2
    elif phase == 'P' and detec_type == 'T':
        nmax = 100
    else:
        nmax = 1
    # select stations with most picks
    var1 = var1[var1.values > nmax]
    # channel labelling
    xtick = var1.index
    # create figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    # plot bar histograms
    ax1.bar(x=xtick, height=var1.values)
    ax2.bar(x=var2.index, height=var2.values)
    ax2.set_xlim([0, 30])
    # rotate tick labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-45, fontsize=5, ha='left', rotation_mode='anchor')
    # labels
    ax1.set_xlabel(f'Stations (>{nmax} picks)')
    ax1.set_ylabel(f'Number of {phase} picks')
    ax2.set_xlabel(f'Number of {phase} picks per event')
    ax2.set_ylabel('Number of events')
    # title
    if detec_type == 'C':
        fig.suptitle(f'Catalogue events (N={sum(var2.values)})')
    elif detec_type == 'T':
        fig.suptitle(f'True automatic events (N={sum(var2.values)})')
    if detec_type == 'F':
        fig.suptitle(f'False automatic events (N={sum(var2.values)})')
    # adjust plot
    fig.subplots_adjust(left=.07, right=.97, top=.92, bottom=.15)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show plot
    plt.show()


def plot_picks_ratios(work_dir):
    """
    :param work_dir: working directory
    :return: nothing
    """
    # n_true/n_unassoc limit for plotting
    slim = .0005
    if os.path.exists(f'{work_dir}/pick-number-ratios-all.csv') == 0:
        # create table for all picks
        all_tab = pd.DataFrame({'Stream': pd.Series(dtype='string'), 'n_true': pd.Series(dtype='int64'),
                                'n_all': pd.Series(dtype='int64'), 'pick_ratio': pd.Series(dtype='float64')})
        # read monthly True event picks table
        tab1 = pd.read_csv(f'{work_dir}/2022-12/picks-P-T.csv', parse_dates=['PickTime'])
        tab2 = pd.read_csv(f'{work_dir}/2023-01/picks-P-T.csv', parse_dates=['PickTime'])
        tab3 = pd.read_csv(f'{work_dir}/2023-02/picks-P-T.csv', parse_dates=['PickTime'])
        # add-up tables
        true_tab = pd.concat([tab1, tab2, tab3], ignore_index=True).set_index('EventID')
        del tab1, tab2, tab3
        # loop over unique streams
        for stream, count in true_tab.value_counts('Stream').iteritems():
            # case no picks for stream in True events
            if count == 0:
                all_tab.loc[all_tab.shape[0]] = [stream, count, 0, np.nan]
                continue
            # read monthly all-picks tables
            tab1 = read_autopick_xml(2022, 12, str(stream), work_dir)
            tab2 = read_autopick_xml(2023, 1, str(stream), work_dir)
            tab3 = read_autopick_xml(2023, 2, str(stream), work_dir)
            # add-up tables
            stream_tab = pd.concat([tab1, tab2, tab3], ignore_index=True)
            # extract True event picks for stream of interest
            df2 = stream_tab[stream_tab.Phase == 'P']
            # fill all-picks table
            all_tab.loc[all_tab.shape[0]] = [stream, count, len(df2), count / len(df2)]
            # # get True event picks
            # df1 = true_tab[['Stream', 'PickTime']][assoc_tab.Stream == stream]
            # # get all picks for stream of interest
            # df2 = stream_tab[stream_tab.Phase == 'P']
            # # compare both tables
            # df_both = df2.merge(df1, how='left', on=['Stream', 'PickTime'], indicator=True)
            # # look for picks not in True events
            # df_notin = df_both[df_both['_merge'] != 'both'].reset_index(drop=True)
            # # fill unassociated-picks table
            # all_tab.loc[all_tab.shape[0]] = [stream, count, len(df_notin), count/len(df_notin)]
        # write output file
        all_tab.to_csv(f'{wdir}/pick-number-ratios-all.csv', float_format='%.4f', index=False)
        # remove Nan's, zero-counts, Turkish stations and non-vertical streams
        all_tab = all_tab[~pd.isna(all_tab.pick_ratio) & (all_tab.pick_ratio != 0)
                          & ~all_tab.Stream.str.contains('KO.')
                          & (all_tab.Stream.str.contains('ENZ') | all_tab.Stream.str.contains('BHZ') |
                             all_tab.Stream.str.contains('HHZ') | all_tab.Stream.str.contains('SHZ'))]
    else:
        # read file
        all_tab = pd.read_csv(f'{wdir}/pick-number-ratios-all.csv')
        # remove Nan's, zero-counts and Turkish streams
        all_tab = all_tab[~pd.isna(all_tab.pick_ratio) & (all_tab.pick_ratio != 0)
                          & ~all_tab.Stream.str.contains('KO.')
                          & (all_tab.Stream.str.contains('ENZ') | all_tab.Stream.str.contains('BHZ') |
                             all_tab.Stream.str.contains('HHZ') | all_tab.Stream.str.contains('SHZ'))]
    # create figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(18, 9), dpi=200)
    # plot pick ratio
    ax1.plot(all_tab.pick_ratio, range(0, len(all_tab)), '+', markersize=5, alpha=.7)
    # axis limits
    ax1.set_xlim([-.005, .05])
    # ax1.set_ylim([-1, len(all_tab)])
    # plot SNR limit
    ax1.plot([slim, slim], ax1.get_ylim(), c='red', alpha=.7)
    # display channel info (+ coloring depending on SNR limit)
    for i, x in all_tab.iterrows():
        if x.pick_ratio < slim:
            ax1.text(x.pick_ratio, i-.5, s=x.Stream, color='red', ha='right', va='top', fontsize=5)
    # labels
    ax1.set_xlabel('N_true / N_all')
    ax1.set_ylabel('Station index')
    # adjust plot
    fig.subplots_adjust(left=.07, right=.97, top=.92, bottom=.15)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show plot
    plt.show()


def plot_snr_histograms(detec_type, phase, work_dir):
    """
    :param detec_type: detection type to visualise; one-character string: T/F/C (True/False/Catalogue)
    :param phase: seismic phase of interest (P/S)
    :param work_dir: working directory
    :return: nothing
    """
    # # selection limits
    # xlim = 800.
    # ylim = 30.
    # read picks table
    if os.path.exists(f'{work_dir}/2022-12/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2022, 12, work_dir)
    tab1 = pd.read_csv(f'{work_dir}/2022-12/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    if os.path.exists(f'{work_dir}/2023-01/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2023, 1, work_dir)
    tab2 = pd.read_csv(f'{work_dir}/2023-01/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    if os.path.exists(f'{work_dir}/2023-02/picks-{phase}-{detec_type}.csv') == 0:
        write_picks_table(detec_type, phase, 2023, 2, work_dir)
    tab3 = pd.read_csv(f'{work_dir}/2023-02/picks-{phase}-{detec_type}.csv', parse_dates=['PickTime'])
    # add-up tables
    output_tab = pd.concat([tab1, tab2, tab3], ignore_index=True).set_index('EventID')
    # remove Nan's, zero-counts, Turkish stations and non-vertical streams
    output_tab = output_tab[~pd.isna(output_tab.SNR) & (output_tab.SNR != 0) & ~output_tab.Stream.str.contains('KO')
                            & (output_tab.Stream.str.contains('ENZ') | output_tab.Stream.str.contains('BHZ')
                               | output_tab.Stream.str.contains('HHZ') | output_tab.Stream.str.contains('SHZ'))]
    # # SNR station average variables
    # lsta = output_tab[['Stream', 'SNR']].reset_index(drop=True).groupby('Stream').mean().index.to_list()
    # asnr = output_tab[['Stream', 'SNR']].reset_index(drop=True).groupby('Stream').mean().SNR.to_list()
    # npic = output_tab[['Stream', 'SNR']].reset_index(drop=True).groupby('Stream').count().SNR.to_list()
    # # create figure
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(18, 9), dpi=200)
    # ax1.plot(npic, asnr, '+', markersize=5, alpha=.7)
    # # show streams
    # for i, s in enumerate(asnr):
    #     if s > ylim:
    #         ax1.text(npic[i], s+5, s=lsta[i], color='limegreen',
    #                  ha='center', va='center', fontsize=5)
    #     if npic[i] > xlim:
    #         ax1.text(npic[i], s+5, s=lsta[i], color='red', ha='center', va='center', fontsize=5)
    # # x=y line
    # ax1.plot([0, ax1.get_ylim()[1]], [0, ax1.get_ylim()[1]], color='black', label='x=y')
    # # selection limits
    # ax1.plot(ax1.get_xlim(), [ylim, ylim], color='limegreen', label=f'y={ylim}')
    # if detec_type == 'F':
    #     ax1.plot([xlim, xlim], ax1.get_ylim(), color='red', label=f'x={xlim}')
    # # labels
    # ax1.set_xlabel(f'Number of {phase} picks')
    # ax1.set_ylabel(f'Mean {phase} pick SNR')
    # # legend
    # ax1.legend(loc='upper right')
    # # title
    # if detec_type == 'C':
    #     fig.suptitle(f'Catalogue events')
    # elif detec_type == 'T':
    #     fig.suptitle(f'True automatic events')
    # if detec_type == 'F':
    #     fig.suptitle(f'False automatic events')
    # # adjust plot
    # fig.subplots_adjust(left=.08, right=.97, top=.93, bottom=.1)
    # # maximise figure
    # plt.get_current_fig_manager().full_screen_toggle()
    # # show figure
    # plt.show()
    # exit()
    # prepare table for boxplot
    all_snr = output_tab[['Stream', 'SNR']].reset_index(drop=True)
    # create figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    # plot bar histograms
    output_tab.plot.hist(column='SNR', bins=np.arange(-.5, 11.5), ec='white', legend=False, ax=ax1)
    # labels
    ax1.set_xlabel(f'{phase} pick SNR')
    ax1.set_ylabel(f'Number of {phase} picks')
    # boxplot station SNRs
    ax2.yaxis.tick_right()
    all_snr.boxplot(column='SNR', by='Stream', vert=False, showfliers=False,
                    boxprops=dict(linestyle='-', linewidth=.5),
                    medianprops=dict(linestyle='-', linewidth=.5, color='red'),
                    whiskerprops=dict(linestyle='-', linewidth=.5, color='blue'),
                    capprops=dict(linestyle='-', linewidth=.5, color='black'), ax=ax2)
    # remove automatic labels
    ax2.set_ylabel('')
    ax2.set_title('')
    # axis labels
    ax2.set_xlabel(f'{phase} pick SNR')
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=2.5)
    # title
    if detec_type == 'C':
        fig.suptitle(f'Catalogue events')
    elif detec_type == 'T':
        fig.suptitle(f'True automatic events')
    if detec_type == 'F':
        fig.suptitle(f'False automatic events')
    # adjust plot
    fig.subplots_adjust(left=.08, right=.95, top=.93, bottom=.1, wspace=.02)
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show plot
    plt.show()


# input parameters
ntw = 'IS,GE'
chn = '(B|H|E|S)(H|N)Z'
fpar = {'rmhp': 10., 'taper': 30., 'bworder': 4, 'bwminf': 4., 'bwmaxf': 8.,
        'sta': .2, 'lta': 10., 'trigon': 3., 'trigoff': 1.5}
vel_mod = 'giimod'

# working directory
wdir = f'/home/lewis/autopicker-rt'
if os.environ['LOGNAME'] == 'sysop':
    wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/autopicker-rt'
# data archive directory
adir = '/net/172.16.46.200/HI_Archive'

# time windows
wwin = timedelta(minutes=5)
ewin = timedelta(minutes=5)

check_missed_events(2023, 6, 1, wdir)
# list_auto_events(2023, 1, 1, wdir)
# check_false_events(2023, 2, 1, wdir)
# plot_histograms(wdir)
# plot_maps(wdir)
# plot_picks_histograms('F', 'P', wdir)
# plot_picks_ratios(wdir)
# plot_snr_histograms('T', 'P', wdir)
