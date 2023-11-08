import os
from os import path
import re
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy.distance as gdist
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import json
from collections import defaultdict
import numpy as np
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read
from obspy.taup import TauPyModel


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom log_chk with the given name and level
    """
    log = logging.getLogger('earthquakes')
    log.setLevel(level)
    format_string = "%(asctime)s.%(msecs)03d \u2013 %(levelname)s: %(message)s"
    log_format = logging.Formatter(format_string, '%Y-%m-%d %H:%M:%S')
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    if log.handlers:
        h = log.handlers[0]
        h.close()
        log.removeHandler(h)
    log.addHandler(file_handler)
    return log


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
    # delete file if existing but empty
    if path.exists(mpath) and os.path.getsize(mpath) == 0:
        os.remove(mpath)
    # define start and end times for data retrieval
    tbeg = str(datetime.strftime(ori_time-time_win, '%Y-%m-%d %H:%M:%S'))
    tend = str(datetime.strftime(ori_time+time_win, '%Y-%m-%d %H:%M:%S'))
    # retrieve data using SeisComP's 'scart' command
    os.system(f'/home/sysop/seiscomp/bin/scart -dsE -n "{networks}" '
              f'-c "{channels}" -t "{tbeg}~{tend}" {db_dir} > {mpath}')
    # delete file if empty
    if os.path.getsize(mpath) == 0:
        os.remove(mpath)
        return
    # read .mseed file
    stream = read(mpath)
    # remove problematic AMZNI (20 Hz) trace
    for trace in stream:
        if trace.stats.station == 'AMZNI' and trace.stats.channel == 'BHZ' and trace.stats.sampling_rate == 20.:
            stream.remove(trace)
    # merge traces from different networks/channels separately (different sampling rates)
    s1 = stream.select(network='IS', channel='BHZ').merge(fill_value='interpolate')
    s2 = stream.select(network='IS', channel='HHZ').merge(fill_value='interpolate')  # add locations!!
    s3 = stream.select(network='IS', channel='ENZ').merge(fill_value='interpolate')
    s4 = stream.select(network='IS', channel='SHZ').merge(fill_value='interpolate')
    s5 = stream.select(network='GE', channel='BHZ').merge(fill_value='interpolate')
    s6 = stream.select(network='GE', channel='HHZ').merge(fill_value='interpolate')
    s7 = stream.select(network='GE', channel='ENZ').merge(fill_value='interpolate')
    s8 = stream.select(network='GE', channel='SHZ').merge(fill_value='interpolate')
    stream = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
    # write .mseed file
    stream.write(mpath)
    return


def process_waveforms(mpath, filt_param, cat_evt, ref_mod, sta_inv, log_file, rm_acc=False):
    """
    :param mpath: data streamer containing waveforms to process
    :param filt_param: dictionary containing filter parameters for waveforms
    :param cat_evt: event entry from xml container with all event data and picks
    :param ref_mod: reference velocity model to use for theoretical travel times
    :param sta_inv: .xml inventory containing all station info.
    :param log_file: logging file
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
        # ONLY KEEP SHZ DATA FOR IS.PRNI
        if trace.stats.channel == 'SHZ':
            if trace.stats.station != 'PRNI' or trace.stats.network != 'IS':
                to_del.append(trace)
                continue
        # REMOVE ACCELEROMETER DATA WITHOUT PICKS
        if rm_acc and trace.stats.channel == 'ENZ':
            # initialise logical variable if pick found
            found = False
            # loop over picks
            for pick in cat_evt.picks:
                # check waveform has picks
                if pick.waveform_id.id == f'{trace.stats.network}.{trace.stats.station}.' \
                                          f'{trace.stats.location}.{trace.stats.channel}':
                    # loop over arrivals
                    for arrival in cat_evt.preferred_origin().arrivals:
                        # look for corresponding arrival
                        if arrival.pick_id.id == pick.resource_id.id:
                            found = True
            # add to to-delete list if pick not found
            if not found:
                to_del.append(trace)
                continue
        # SPECIAL CASE OF ISP (two BHZ channels on two locations --> keep GE.ISP.00.BHZ)
        if trace.stats.station == 'ISP' and trace.stats.location == '10' and trace.stats.channel == 'BHZ':
            to_del.append(trace)
            continue
        # REMOVE DUPLICATE VELOCITY DATA
        if trace.stats.channel == 'BHZ' and \
                stream.select(station=trace.stats.station, network=trace.stats.network, channel='HHZ') != [] or \
           trace.stats.channel == 'HHZ' and \
                stream.select(station=trace.stats.station, network=trace.stats.network, channel='BHZ') != []:
            # initialise logical variables
            f1 = False
            f2 = False
            # select waveforms for same station, different channels
            lst = stream.select(network=trace.stats.network, station=trace.stats.station, channel='[BH]HZ')
            # loop over waveforms found
            for tr in lst:
                # loop over picks
                for pick in cat_evt.picks:
                    # check waveform has picks
                    if pick.waveform_id.id == f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.BHZ':
                        # loop over arrivals
                        for arrival in cat_evt.preferred_origin().arrivals:
                            # look for corresponding arrival
                            if arrival.pick_id.id == pick.resource_id.id:
                                f1 = True
                    # check waveform has picks
                    if pick.waveform_id.id == f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.HHZ':
                        # loop over arrivals
                        for arrival in cat_evt.preferred_origin().arrivals:
                            # look for corresponding arrival
                            if arrival.pick_id.id == pick.resource_id.id:
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
        dis = gdist.distance((cat_evt.preferred_origin().latitude,
                              cat_evt.preferred_origin().longitude),
                             (station[0].stations[0].channels[0].latitude,
                              station[0].stations[0].channels[0].longitude))
        # add event-station distance to trace header
        trace.stats.distance = dis.m
        # add station location
        trace.stats['sta_lat'] = station[0].stations[0].channels[0].latitude
        trace.stats['sta_lon'] = station[0].stations[0].channels[0].longitude
        # THEORETICAL ARRIVAL
        # compute theoretical travel time for all possible first P arrivals
        theo_tp = theory.get_travel_times(source_depth_in_km=cat_evt.preferred_origin().depth/1000.,
                                          distance_in_degree=dis.km/(2.*np.pi*earth_rad/360.),
                                          phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
        if len(theo_tp) != 0:
            # add theoretical travel time to trace header
            trace.stats['theo_tp'] = cat_evt.preferred_origin().time.datetime + \
                                     timedelta(seconds=theo_tp[0].time)
        else:
            # save as NaT in case of no theoretical arrival
            trace.stats['theo_tp'] = np.datetime64('NaT')
            log_file.warning(f"{cat_evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')}: "
                             f"could not compute theoretical P arrivals")
        # compute theoretical travel time for all possible first S arrivals
        theo_ts = theory.get_travel_times(source_depth_in_km=cat_evt.preferred_origin().depth/1000.,
                                          distance_in_degree=dis.km/(2.*np.pi*earth_rad/360.),
                                          phase_list=['s', 'S', 'Sg', 'Sn', 'Sdiff'])
        if len(theo_ts) != 0:
            # add theoretical travel time to trace header
            trace.stats['theo_ts'] = cat_evt.preferred_origin().time.datetime + \
                                     timedelta(seconds=theo_ts[0].time)
        else:
            # save as NaT in case of no theoretical arrival
            trace.stats['theo_ts'] = np.datetime64('NaT')
            log_file.warning(f"{cat_evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')}: "
                             f"could not compute theoretical S arrivals")
    # delete selected waveforms (that were not already removed)
    for trace in to_del:
        try:
            stream.remove(trace)
        except:
            continue
    # sort according to station latitude
    stream.sort(['distance'], reverse=True)
    # apply taper to all traces
    stream.taper(max_percentage=.5, type='cosine', max_length=filt_param['taper'], side='left')
    # apply high-pass filter to all traces
    stream.filter('highpass', freq=1./filt_param['rmhp'])
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply Butterworth band-pass filter to all traces
    stream.filter('bandpass', freqmin=filt_param['bwminf'], freqmax=filt_param['bwmaxf'], corners=filt_param['bworder'])
    return stream


def plot_cata_evt_sec(stream, cat_evt, time_win, fig_name, log_file):
    """
    :param stream: data streamer of waveforms to process
    :param cat_evt: xml container with all catalogue data and picks
    :param time_win: half-length of time window as a datetime.timedelta() object
    :param fig_name: figure file full path
    :param log_file: logging file
    :return: nothing
    """
    # Earth radius
    earth_rad = 6371.
    # geographic boundaries
    local_map = [29., 34., 33.9, 36.5]
    regional_map = [23., 43., 25., 45.]
    # EMSC's FDSN client
    try:
        emsc_client = Client('EMSC')
    except:
        log_file.warning('EMSC FDSN server inaccessible')
        emsc_client = []
    regi_emsc = []
    tele_emsc = []
    if emsc_client:
        # regional events
        try:
            regi_emsc = emsc_client.get_events(
                starttime=UTCDateTime(cat_evt.preferred_origin().time.datetime-time_win),
                endtime=UTCDateTime(cat_evt.preferred_origin().time.datetime+time_win),
                minlatitude=regional_map[0], maxlatitude=regional_map[1], minlongitude=regional_map[2],
                maxlongitude=regional_map[3], minmagnitude=3.)
        except:
            regi_emsc = []
        # teleseismic events
        try:
            tele_emsc = emsc_client.get_events(
                starttime=UTCDateTime(cat_evt.preferred_origin().time.datetime-time_win),
                endtime=UTCDateTime(cat_evt.preferred_origin().time.datetime+time_win),
                minmagnitude=6.)
        except:
            tele_emsc = []
    # plotting mode
    mpl.use('Agg')
    # waveforms time window (min Tp and max Ts)
    tmin = min([tr.stats.theo_tp for tr in stream
                if not np.isnat(np.datetime64(tr.stats.theo_tp))]) - timedelta(minutes=1.)
    tmax = max([tr.stats.theo_ts for tr in stream
                if not np.isnat(np.datetime64(tr.stats.theo_ts))]) + timedelta(minutes=2.)
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
    axis2.set_xlim([-5., 5.])
    # AXIS 3: LOCAL STATIONS & EVENT MAP
    # define map boundaries & resolution
    m1 = Basemap(projection='cyl', llcrnrlon=local_map[2], llcrnrlat=local_map[0],
                 urcrnrlon=local_map[3], urcrnrlat=local_map[1], resolution='i', ax=axis3)
    # draw map
    m1.drawmapboundary(fill_color='none')
    # fill continents
    m1.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels=[left,right,top,bottom])
    m1.drawparallels(np.arange(m1.llcrnrlat, m1.urcrnrlat+1, 2.), labels=[True, False, True, False])
    m1.drawmeridians(np.arange(m1.llcrnrlon, m1.urcrnrlon+1, 2.), labels=[True, False, False, True])
    # faults
    flts_id = open('/home/sysop/.seiscomp/spatial/vector/ActiveFaults/activefaults.bna', 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    hf = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts)-1 and re.search('"', flts[iii + 1]):
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
                hq = axis3.scatter(float(l_line[0]), float(l_line[1]), s=25, c='blue', marker='s',
                                   edgecolors='black', label='Quarries', alpha=.1, zorder=2)
    # AXIS 4: INSET MAP FOR REGIONAL SETTINGS
    axis4 = inset_axes(axis3, '30%', '18%', loc='lower left')
    m2 = Basemap(projection='cyl', llcrnrlon=regional_map[2], llcrnrlat=regional_map[0],
                 urcrnrlon=regional_map[3], urcrnrlat=regional_map[1], resolution='l', ax=axis4)
    # draw map
    m2.drawmapboundary(fill_color='white')
    # fill continents
    m2.fillcontinents(color='0.8', lake_color='white')
    # plot GE station locations
    for station in isn_inv.networks[0].stations:
        if not (local_map[0] < station.latitude < local_map[1] and local_map[2] < station.longitude < local_map[3]):
            axis4.scatter(station.longitude, station.latitude,
                          s=25, color='cyan', marker='^', edgecolors='black', alpha=.2, zorder=5)
            axis4.text(station.longitude+.5, station.latitude+.1, station.code,
                       c='black', ha='left', va='bottom', fontsize=5, alpha=.2, clip_on=True)
    # highlight area of interest
    axis4.plot([local_map[2], local_map[2], local_map[3], local_map[3], local_map[2]],
               [local_map[0], local_map[1], local_map[1], local_map[0], local_map[0]], color='saddlebrown')
    # plot catalogue evt location
    axis4.scatter(cat_evt.preferred_origin().longitude, cat_evt.preferred_origin().latitude,
                  s=100, c='red', marker='*', edgecolors='black', zorder=100)
    # plot EMSC events location
    her = []
    for e in regi_emsc:
        if tmin-timedelta(minutes=2.) < e.preferred_origin().time.datetime < tmax+timedelta(minutes=2.):
            # in local map
            her = axis3.scatter(e.preferred_origin().longitude, e.preferred_origin().latitude, s=100,
                                c='orange', marker='*', edgecolors='black', alpha=.7, label='EMSC', zorder=150)
            # in regional map
            axis4.scatter(e.preferred_origin().longitude, e.preferred_origin().latitude,
                          s=100, c='orange', marker='*', edgecolors='black', alpha=.7, zorder=150)
    # LOOP OVER TRACES
    # intialise counters
    n_trace = 0
    # initialise tables for legend
    hw = []
    htp = []
    hts = []
    hp = []
    hs = []
    hrp = []
    hrs = []
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
        tvec = np.arange(0, len(trace)) * np.timedelta64(int(trace.stats.delta * 1000), '[ms]') \
            + np.datetime64(str(trace.stats.starttime)[:-1])
        # look for indexes closest to tmin and tmax for normalisation
        i1 = np.argmin(np.abs(tvec-np.datetime64(tmin)))
        i2 = np.argmin(np.abs(tvec-np.datetime64(tmax)))
        if i2-i1 < 200:
            i1 = 0
            i2 = len(trace.data)
        # maximum amplitude for normalisation
        wmax = 2.*max(trace.data[i1:i2])
        # plot waveform
        hw, = axis1.plot(tvec, trace.data / wmax + n_trace, color='grey', alpha=.7, label='Velocity')
        # plot theoretical travel time
        if trace.stats.theo_tp:
            htp, = axis1.plot([trace.stats.theo_tp, trace.stats.theo_tp], [n_trace - 1, n_trace + 1],
                              color='lime', linestyle='dotted', label=f'{vel_mod} (P)')
        if trace.stats.theo_ts:
            hts, = axis1.plot([trace.stats.theo_ts, trace.stats.theo_ts], [n_trace - 1, n_trace + 1],
                              color='cyan', linestyle='dotted', label=f'{vel_mod} (S)')
        # plot station in map
        st = isn_inv.select(network=trace.stats.network, station=trace.stats.station, channel=trace.stats.channel)
        if not st:
            log_file.warning(f'Missing station: {stn_lbl[n_trace-1]}')
        # use different symbols and colours for different network codes
        if trace.stats.network == 'IS':
            hn1 = axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude, s=25,
                                c='magenta', marker='^', edgecolors='black', alpha=.2,
                                label=trace.stats.network, zorder=3)
        elif trace.stats.network == 'GE':
            hn2 = axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude, s=25,
                                c='cyan', marker='^', edgecolors='black', alpha=.2, label=trace.stats.network, zorder=3)
        # CATALOGUE PICKS
        # loop over catalogue evt picks
        for pick in cat_evt.picks:
            if pick.waveform_id.network_code == trace.stats.network \
                    and pick.waveform_id.station_code == trace.stats.station \
                    and pick.waveform_id.channel_code == trace.stats.channel:
                # loop over arrivals
                for arrival in cat_evt.preferred_origin().arrivals:
                    if arrival.pick_id.id == pick.resource_id.id:
                        # highlight station in map
                        if trace.stats.network == 'IS':
                            axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                          s=25, c='magenta', marker='^', edgecolors='red', zorder=4)
                            axis3.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude+0.06,
                                       st.networks[0].stations[0].code, color='red', ha='center', va='center',
                                       fontsize=5, clip_on=True)
                        elif trace.stats.network == 'GE':
                            axis3.scatter(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude,
                                          s=25, c='cyan', marker='^', edgecolors='red', zorder=4)
                            axis3.text(st.networks[0].stations[0].longitude, st.networks[0].stations[0].latitude-0.07,
                                       st.networks[0].stations[0].code, color='red', ha='center', va='center',
                                       fontsize=5, clip_on=True)
                        # connect station to event in both maps
                        axis3.plot([cat_evt.preferred_origin().longitude, st.networks[0].stations[0].longitude],
                                   [cat_evt.preferred_origin().latitude, st.networks[0].stations[0].latitude],
                                   color='red', alpha=.2, zorder=2)
                        axis4.plot([cat_evt.preferred_origin().longitude, st.networks[0].stations[0].longitude],
                                   [cat_evt.preferred_origin().latitude, st.networks[0].stations[0].latitude],
                                   color='red', alpha=.2, zorder=2)
                        # show picks
                        if arrival.phase == 'P':
                            # pick
                            hp, = axis1.plot([pick.time.datetime, pick.time.datetime],
                                             [n_trace-1, n_trace+1], color='purple', label='P arrival')
                            # residual
                            hrp = axis2.scatter(arrival.time_residual, n_trace, s=49, c='purple',
                                                edgecolors='black', alpha=.7, label='Cat-Theo P', zorder=10)
                            break
                        elif arrival.phase == 'S':
                            # pick
                            hs, = axis1.plot([pick.time.datetime, pick.time.datetime],
                                             [n_trace-1, n_trace+1], color='magenta', label='S arrival')
                            # residual
                            hrs = axis2.scatter(arrival.time_residual, n_trace, s=49, c='magenta', marker='s',
                                                edgecolors='black', alpha=.7, label='Cat-Theo S', zorder=10)
                            break
    # axis limits
    axis1.set_ylim([0, n_trace+1])
    axis2.set_ylim([0, n_trace+1])
    # legend
    hh = [hw, htp, hts]
    if hp:
        hh.append(hp)
    if hs:
        hh.append(hs)
    axis1.legend(handles=hh, loc='upper right', fontsize=8)
    # show EMSC events origin time
    for e in regi_emsc:
        axis1.plot([e.preferred_origin().time.datetime, e.preferred_origin().time.datetime], [0, n_trace+1],
                   color='orange', alpha=.7)
        if tmin-timedelta(minutes=1.) < e.preferred_origin().time.datetime < tmax+timedelta(minutes=1.):
            axis1.text(e.preferred_origin().time.datetime, n_trace+1.1,
                       f'{e.event_descriptions[0].text} M{e.preferred_magnitude().mag}',
                       color='orange', ha='center', va='bottom', fontsize=8)
    for e in tele_emsc:
        axis1.plot([e.preferred_origin().time.datetime, e.preferred_origin().time.datetime], [0, n_trace+1],
                   color='green', alpha=.7)
        if tmin-timedelta(minutes=1.) < e.preferred_origin().time.datetime < tmax+timedelta(minutes=1.):
            axis1.text(e.preferred_origin().time.datetime, n_trace+1.1,
                       f'{e.event_descriptions[0].text} M{e.preferred_magnitude().mag}',
                       color='green', ha='center', va='bottom', fontsize=8)
    # highlight catalogue evt origin time
    axis1.plot([cat_evt.preferred_origin().time.datetime, cat_evt.preferred_origin().time.datetime],
               [0, n_trace+1], color='red', alpha=.7)
    # station and pick numbers
    axis1.text(-.01, -.01, f"N={n_trace}", ha='right', va='center', transform=axis1.transAxes)
    # replace numerical tick labels with station names
    axis1.set_yticks(np.arange(1, n_trace+1, 1))
    axis1.set_yticklabels(stn_lbl, fontsize=5)
    axis2.set_yticklabels([])
    # axis labels
    axis1.set_xlabel('Time', fontweight='bold')
    axis1.set_ylabel('Station', fontweight='bold')
    axis2.set_xlabel('\u0394t [s]', fontweight='bold')
    # EVENTS
    # plot catalogue evt location
    hce = axis3.scatter(cat_evt.preferred_origin().longitude, cat_evt.preferred_origin().latitude,
                        s=100, c='red', marker='*', edgecolors='black', zorder=100, label='GSI')
    # plot error ellipse
    hel = axis3.add_patch(mpl.patches.Ellipse(
        (cat_evt.preferred_origin().longitude, cat_evt.preferred_origin().latitude),
        width=cat_evt.preferred_origin().longitude_errors.uncertainty/(2.*np.pi*earth_rad/360.),
        height=cat_evt.preferred_origin().latitude_errors.uncertainty/(2.*np.pi*earth_rad/360.),
        facecolor='green', edgecolor='none', alpha=.2, zorder=101, label='Uncertainty'))
    axis4.add_patch(mpl.patches.Ellipse(
        (cat_evt.preferred_origin().longitude, cat_evt.preferred_origin().latitude),
        width=cat_evt.preferred_origin().longitude_errors.uncertainty/(2.*np.pi*earth_rad/360.),
        height=cat_evt.preferred_origin().latitude_errors.uncertainty/(2.*np.pi*earth_rad/360.),
        facecolor='green', edgecolor='none', alpha=.2, zorder=101))
    # map legend
    if hn2:
        hm = [hn1, hn2, hce, hel]
    else:
        hm = [hn1, hce, hel]
    if her:
        hm.append(her)
    hm.append(hf)
    hm.append(hq)
    axis3.legend(handles=hm, loc='upper left', fontsize=8)
    # residual plot legend
    h = []
    if hrp:
        h.append(hrp)
    if hrs:
        h.append(hrs)
    if h:
        axis2.legend(handles=h, loc='lower right', fontsize=8)
    # figure title
    emag = np.nan
    if cat_evt.preferred_magnitude():
        emag = f'{cat_evt.preferred_magnitude().magnitude_type}{cat_evt.preferred_magnitude().mag:3.1f}'
    if cat_evt.preferred_origin().depth_errors.uncertainty:
        edep = f"{cat_evt.preferred_origin().depth/1000.:.1f}\u00B1" \
               f"{cat_evt.preferred_origin().depth_errors.uncertainty/1000.:.1f} km"
    else:
        edep = f"{cat_evt.preferred_origin().depth/1000.:.1f}\u00B1nan km"
    tit = f"{cat_evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S')} \u2013 {edep} \u2013 {emag}"
    fig.suptitle(f"{tit}\n{cat_evt.event_type} ({cat_evt.preferred_origin().creation_info.author})", fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.07, right=.97, top=.92, wspace=.1)
    # show or save figure
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    plt.close()
    return


# input parameters
ntw = 'IS,GE'
chn = '(S|B|H|E)(H|N)Z'
fpar = {'rmhp': 10., 'taper': 30., 'bworder': 4, 'bwminf': 4., 'bwmaxf': 8.,
        'sta': .2, 'lta': 10., 'trigon': 3., 'trigoff': 1.5}

# area of interest
lmap = [29., 34., 33., 37.]
vel_mod = 'giimod'

# working directory
wdir = f'/net/172.16.46.200/archive/monitor/events'
# data archive directory
adir = '/net/172.16.46.200/HI_Archive'
# FDSN database
isn_client = Client('http://172.16.46.102:8181/')

# retrieve station inventory
isn_inv = isn_client.get_stations(network=ntw, channel='ENZ,HHZ,BHZ,SHZ', level='response')

# depth limit for local earthquakes
dlim = 40.

# time window for waveform selection
twin = timedelta(minutes=5)

# DAILY EVENTS
# starting day
sday = datetime.now() - timedelta(days=1)
# ending day
eday = datetime.now()
# number of days before today to check
nday = (eday-sday).days
# loop over days
for day in range(0, nday):
    # year & month
    year = (sday+timedelta(days=day)).year
    month = (sday+timedelta(days=day)).month
    # event list
    try:
        event_list = isn_client.get_events(starttime=datetime.strftime(sday+timedelta(days=day), '%Y-%m-%d 00:00:00'),
                                           endtime=datetime.strftime(sday+timedelta(days=day+1), '%Y-%m-%d 00:00:00'),
                                           includearrivals=True, includecomments=True, orderby='time-asc')
    except:
        continue
    print(f"{datetime.strftime(sday+timedelta(days=day), '%Y-%m-%d')}: {len(event_list)} "
          f"{('event' if len(event_list) == 1 else 'events')}")
    # check directories exist
    if path.exists(f'{wdir}/{year}') == 0:
        os.mkdir(f'{wdir}/{year}')
    if path.exists(f'{wdir}/{year}/{month:02d}') == 0:
        os.mkdir(f'{wdir}/{year}/{month:02d}')
    # daily directory name
    ddir = datetime.strftime(sday+timedelta(days=day), '%Y/%m/%d')
    if path.exists(f"{wdir}/{ddir}") == 0:
        os.mkdir(f"{wdir}/{ddir}")
    # daily file name
    dfil = datetime.strftime(sday+timedelta(days=day), '%Y%m%d')
    # logging files
    if path.exists(f"{wdir}/{ddir}/event_checks_log.txt") != 0:
        os.remove(f"{wdir}/{ddir}/event_checks_log.txt")
    if path.exists(f"{wdir}/{ddir}/event_checks_log.txt") != 0:
        os.remove(f"{wdir}/{ddir}/event_auto_log.txt")
    # checks for processed events
    log_chk = my_custom_logger(f"{wdir}/{ddir}/event_checks_log.txt", level=logging.DEBUG)
    # report number of events
    log_chk.info(f"{len(event_list)} {('event' if len(event_list) == 1 else 'events')} on FDSN server")
    log_chk.info('-' * 60)
    # create daily event dictionary
    evt_dict = defaultdict(dict)
    # loop over events
    for event in event_list.events:
        # retrieve event ID
        name = event.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')
        # only check manual events with 'gsi' event ID
        if (event.picks and event.picks[0].evaluation_mode != 'manual') or 'gsi' not in name:
            log_chk.warning(f'{name}: automatic event')
            log_chk.info('-' * 60)
            continue
        # check preferred origin
        if not event.preferred_origin():
            log_chk.warning(f'{name}: no preferred origin')
            log_chk.info('-' * 60)
            continue
        # general event status to track errors
        evt_dict[name]['status'] = f'{event.preferred_origin().evaluation_mode}-' \
                                   f'{event.preferred_origin().evaluation_status}'
        # type
        evt_dict[name]['type'] = (event.event_type if event.event_type else 'unknown event')
        # case Felt
        if event.comments:
            for c in event.comments:
                if c.text and c.text.lower() == 'felt':
                    evt_dict[name]['type'] = 'Felt' + event.event_type
                    break
        if evt_dict[name]['type'] == 'unknown event':
            evt_dict[name]['status'] = 'error-type'
        # magnitude
        evt_dict[name]['magnitude'] = (event.preferred_magnitude().mag if event.preferred_magnitude() else np.nan)
        evt_dict[name]['magnitude_type'] = (event.preferred_magnitude().magnitude_type
                                            if event.preferred_magnitude() else '')
        if not ('explosion' in evt_dict[name]['type'] or 'blast' in evt_dict[name]['type']) \
                and not event.preferred_magnitude():
            evt_dict[name]['status'] = 'error-magnitude'
        # region
        evt_dict[name]['region'] = (event.preferred_origin().region
                                    if event.preferred_origin().region else 'unknown region')
        if not event.preferred_origin().region:
            # look for region in descriptions
            if event.event_descriptions:
                for d in event.event_descriptions:
                    if d.type == 'region name' and d.text != 'noname':
                        evt_dict[name]['region'] = d.text
                    if evt_dict[name]['type'] == 'unknown' and d.type == 'earthquake name':
                        evt_dict[name]['region'] = d.text
        if evt_dict[name]['region'] == 'unknown region':
            evt_dict[name]['status'] = 'error-region'
        # QC TESTS
        # case no picks
        if not event.picks:
            evt_dict[name]['status'] = 'error-picks'
            log_chk.warning(f'{name}: no picks')
            log_chk.info('-' * 60)
            continue
        else:
            evt_dict[name]['n_picks'] = len(event.picks)
        # case no arrivals
        if not event.preferred_origin().arrivals:
            evt_dict[name]['status'] = 'error-arrivals'
            log_chk.warning(f'{name}: no arrivals')
            log_chk.info('-' * 60)
            continue
        else:
            evt_dict[name]['n_arrivals'] = len(event.preferred_origin().arrivals)
        # case depth >1,000 km
        if event.preferred_origin().depth > 800000.:
            evt_dict[name]['status'] = 'error-depth'
            log_chk.warning(f'{name}: depth >800 km (result divided by 1000)')
            event.preferred_origin().depth = event.preferred_origin().depth/1000.
        # depth
        evt_dict[name]['depth'] = event.preferred_origin().depth/1000.
        event.preferred_origin().depth = event.preferred_origin().depth/1000.
        # display event parameters
        log_chk.info(f"{name}: {evt_dict[name]['magnitude_type']}{evt_dict[name]['magnitude']:3.1f} "
                     f"{evt_dict[name]['type']} in {evt_dict[name]['region']} at {evt_dict[name]['depth']:.1f} km")
        # case earthquake has no preferred magnitude
        if 'explosion' not in evt_dict[name]['type'] and 'blast' not in evt_dict[name]['type'] \
                and np.isnan(evt_dict[name]['magnitude']):
            evt_dict[name]['status'] = 'error-magnitude'
            log_chk.warning(f'{name}: no preferred magnitude')
        # case earthquake magnitude is 0
        if 'explosion' not in evt_dict[name]['type'] and 'blast' not in evt_dict[name]['type'] \
                and not np.isnan(evt_dict[name]['magnitude']) and evt_dict[name]['magnitude'] == 0:
            evt_dict[name]['status'] = 'error-magnitude'
            log_chk.warning(f'{name}: magnitude is 0')
        # case local earthquake depth is >40 km
        if 'explosion' not in evt_dict[name]['type'] and 'blast' not in evt_dict[name]['type'] \
                and event.preferred_origin().depth/1000. > dlim \
                and lmap[0] < event.preferred_origin().latitude < lmap[1] \
                and lmap[2] < event.preferred_origin().longitude < lmap[3]:
            evt_dict[name]['status'] = 'error-depth'
            log_chk.warning(f'{name}: local earthquake depth too large (>{dlim} km): '
                            f'{event.preferred_origin().depth/1000.:.1f} km')
        # case explosion depth is not 0 km
        if ('explosion' in evt_dict[name]['type'] or 'blast' in evt_dict[name]['type']) \
                and evt_dict[name]['depth'] != 0.:
            evt_dict[name]['status'] = 'error-depth'
            log_chk.warning(f"{name}: explosion depth not 0 km: {evt_dict[name]['depth']:.1f} km")
        # write daily dictionary of errors
        with open(f'{wdir}/{ddir}/event_errors_log.json', 'w') as outfile:
            json.dump(evt_dict, outfile, indent=4)
        # skip if figure exists
        if path.exists(f'{wdir}/{ddir}/{name}.png') != 0:
            log_chk.info('-' * 60)
            continue
        ###############################
        log_chk.info('-' * 60)
        continue
        ###############################
        # WAVEFORMS
        # time window depends on distance (longer time window for events >1,200 km from center of ISN)
        if gdist.distance((31.5, 35.), (event.preferred_origin().latitude,
                                        event.preferred_origin().longitude)).km > 1200.:
            win = 2.*twin
        else:
            win = twin
        # retrieve .mseed waveforms if necessary
        if path.exists(f'{wdir}/{ddir}/{name}.mseed') == 0:
            get_waveforms_deci(f'{wdir}/{ddir}/{name}.mseed',
                               event.preferred_origin().time.datetime, win, ntw, chn, adir)
        if path.exists(f'{wdir}/{ddir}/{name}.mseed') == 0:
            log_chk.warning(f"{name}: no available waveforms")
            log_chk.info('-' * 60)
            continue
        # retrieve processed waveforms
        isn_traces = process_waveforms(f'{wdir}/{ddir}/{name}.mseed', fpar, event, vel_mod, isn_inv, log_chk, True)
        # FIGURE
        plot_cata_evt_sec(isn_traces, event, win, f'{wdir}/{ddir}/{name}.png', log_chk)
        log_chk.info('-' * 60)
    # CLEAN-UP
    # list events in .xml container
    elis = [evt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '') for evt in event_list]
    # check event figures correspond to events in list
    obj = os.scandir(f'{wdir}/{ddir}')
    # loop over files in daily directory
    for entry in obj:
        # look into .png .mseed and .xml files only
        if '.png' in entry.name or '.mseed' in entry.name:
            # retrieve the file name (event origin time)
            file = entry.name.replace('.png', '').replace('.mseed', '')
            # check if file name in list
            if file not in elis:
                # remove file if not in list
                os.remove(entry.path)
                log_chk.info(f'Removed {entry.path}')
