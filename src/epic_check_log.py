import os
import sys
import pwd
import re
import logging
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopy.distance as gdist
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from matplotlib.dates import DateFormatter
from obspy.core import UTCDateTime
from obspy import read
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees as km2deg
user_name = pwd.getpwuid(os.getuid())[0]
sys.path.append(f'/home/{user_name}/olmost/TRUAA/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TRUAA.settings")
import django
django.setup()
from epic.models import EVENT, TRIGGER


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


def get_epic_alerts_db(source='eew-b-jer', tbeg=datetime.now()-timedelta(days=7), tend=datetime.now(), alert_sent=True):
    """
    :param source: EPIC system source
    :param tbeg: starting date/time
    :param tend: ending date/time
    :param alert_sent: Boolean to retrieve SENT alerts (Ast=1) or NOT SENT alerts (Ast=0)
    :return: DataFrame containing all EPIC alerts for given period (default is one week)
    """
    # variables to query
    variables = ['eventid', 'ver', 'evlat', 'evlon', 'dep', 'mag', 'time', 'nS', 'alert_time', 'Ast']
    # output EPIC table
    evt_tab = pd.DataFrame(
        {'evt_id': pd.Series(dtype='int64'), 'evt_ver': pd.Series(dtype='int64'),
         'evt_time': pd.Series(dtype='datetime64[ms]'), 'evt_lat': pd.Series(dtype='float64'),
         'evt_lon': pd.Series(dtype='float64'), 'evt_nup': pd.Series(dtype='int64'),
         'evt_nsta': pd.Series(dtype='int64'), 'evt_max_mag': pd.Series(dtype='float64'),
         'evt_max_time': pd.Series(dtype='datetime64[ms]'), 'evt_alert_mag': pd.Series(dtype='float64'),
         'evt_alert_time': pd.Series(dtype='datetime64[ms]'), 'evt_1st_mag': pd.Series(dtype='float64'),
         'evt_1st_time': pd.Series(dtype='datetime64[ms]'), 'evt_M42_time': pd.Series(dtype='datetime64[ms]'),
         'evt_M45_time': pd.Series(dtype='datetime64[ms]')})
    # get alert list from database for month of interest
    al_list = EVENT.objects.filter(Ast=alert_sent, source__contains=source,
                                   alert_time__range=[str(UTCDateTime(tbeg)), str(UTCDateTime(tend))]) \
        .values('eventid').order_by('eventid').distinct()
    # loop over all retrieved alerts
    for item in al_list:
        # maximum magnitude alert
        al1 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source). \
            order_by('-mag').values(*variables).first()

        # first alert
        al2 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source, Ast=alert_sent). \
            order_by('time').values(*variables).first()
        # first version (always non-empty)
        al3 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source). \
            order_by('ver').values(*variables).first()
        # last alert (for location; always non-empty; can equal first version)
        al4 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source). \
            order_by('-ver').values(*variables).first()
        # M4.2 alert
        al5 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source, Ast=alert_sent, mag__gte=4.2). \
            order_by('alert_time').values(*variables).first()
        if not al5:
            t42 = np.datetime64('NaT')
        else:
            t42 = np.datetime64(str(al5['alert_time']).replace('+00:00', ''))
        # M4.5 alert
        al6 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source, Ast=alert_sent, mag__gte=4.5). \
            order_by('alert_time').values(*variables).first()
        if not al6:
            t45 = np.datetime64('NaT')
        else:
            t45 = np.datetime64(str(al6['alert_time']).replace('+00:00', ''))
        evt_tab.loc[evt_tab.shape[0]] = [al4['eventid'], al4['ver'],
                                         np.datetime64(str(al4['time']).replace('+00:00', '')),
                                         al4['evlat'], al4['evlon'], al4['ver']+1, al4['nS'],
                                         al1['mag'], np.datetime64(str(al1['alert_time']).replace('+00:00', '')),
                                         al2['mag'], np.datetime64(str(al2['alert_time']).replace('+00:00', '')),
                                         al3['mag'], np.datetime64(str(al3['alert_time']).replace('+00:00', '')),
                                         t42, t45]
    # sort catalogue events by OT
    evt_tab = evt_tab.sort_values(by=['evt_time'], ignore_index=True)
    return evt_tab


def get_epic_triggers_db(source='eew-b-jer', evt_id=None, evt_ver=None):
    """
    :param source: EPIC system source
    :param evt_id: EPIC event ID for triggers
    :param evt_ver: EPIC event version for triggers
    :return: DataFrame containing all triggers for specific EPIC event ID and version
    """
    # variables to query
    variables = ['eventid', 'ver', 'update', 'order', 'sta', 'chan', 'net', 'loc', 'lat', 'lon', 'trigger_time',
                 'distkm', 'azimuth']
    # output event table
    trig_tab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'evt_ver': pd.Series(dtype='int64'),
                             'trig_sta': pd.Series(dtype='string'), 'trig_net': pd.Series(dtype='string'),
                             'trig_chn': pd.Series(dtype='string'), 'trig_loc': pd.Series(dtype='string'),
                             'trig_lat': pd.Series(dtype='float64'), 'trig_lon': pd.Series(dtype='float64'),
                             'trig_time': pd.Series(dtype='datetime64[ms]'), 'trig_dis': pd.Series(dtype='float64'),
                             'trig_azi': pd.Series(dtype='float64')})
    # get alert list from database for month of interest
    tr_list = TRIGGER.objects.filter(source__contains=source, eventid=evt_id, ver=evt_ver).values(*variables).order_by(
        'trigger_time')
    # loop over all retrieved triggers
    for item in tr_list:
        trig_tab.loc[trig_tab.shape[0]] = [item['eventid'], item['ver'], item['sta'], item['chan'], item['net'],
                                           item['loc'], item['lat'], item['lon'], item['trigger_time'], item['distkm'],
                                           item['azimuth']]
    # sort catalogue events by OT
    trig_tab = trig_tab.sort_values(by=['trig_time'], ignore_index=True)
    return trig_tab


def get_catalogue_events_db(client, tbeg=datetime.now() - timedelta(days=7), tend=datetime.now()):
    """
    :param client: database client to retrieve local events from
    :param tbeg: starting date/time
    :param tend: ending date/time
    :return: DataFrame containing all catalogue events for given period (default is one week)
    """
    # load catalogue events for period of interest (if input with min. magnitude, no explosions)
    evt_lst = client.get_events(starttime=UTCDateTime(tbeg), endtime=UTCDateTime(tend),
                                includearrivals=False, orderby='time-asc', includecomments=True)
    # output event table
    cat_tab = pd.DataFrame({'evt_id': pd.Series(dtype='string'), 'evt_time': pd.Series(dtype='datetime64[ms]'),
                            'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                            'evt_dep': pd.Series(dtype='float64'), 'evt_mag': pd.Series(dtype='float64'),
                            'evt_type': pd.Series(dtype='string')})
    for event in evt_lst:
        # removing relocated (rel*) events from list
        if event.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')[0:3] != 'gsi':
            continue
        # removing 'not existing' events from list
        if event.event_type == 'not existing':
            continue
        # removing events with no preferred origin
        if not event.preferred_origin():
            continue
        # # removing automatic events from list
        # # (important when analysing recent months with not everything organised yet)
        # if event.preferred_origin().evaluation_mode != 'manual':  # or 'autoloc' in event.creation_info.author:
        #     continue
        # event magnitude
        if not event.preferred_magnitude():
            mag = np.nan
        else:
            mag = event.preferred_magnitude().mag
        # event type
        if event.comments and event.comments[0].text == 'FELT':
            typ = 'Felt'
        else:
            typ = (event.event_type or '')
        # fill catalogue events table
        cat_tab.loc[cat_tab.shape[0]] = [event.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', ''),
                                         np.datetime64(event.preferred_origin().time),
                                         event.preferred_origin().latitude, event.preferred_origin().longitude,
                                         event.preferred_origin().depth/1000., mag, (typ.title() or '')]
    return cat_tab


def get_catalogue_event_match(cat_tab, event_row, time_win, time_lim, loca_lim, logger_name):
    """
    :param cat_tab: DataFrame of catalogue events
    :param event_row: DataFrame row for EPIC alert to match
    :param time_win: half-length of time window
    :param time_lim: max. origin time error [s]
    :param loca_lim: max. location error [km]
    :param logger_name: log file to use
    :return: catalogue event ID if match; None if not
    """
    # look for indexes of CAT events within time window
    time_sel = cat_tab.index[(cat_tab.evt_time > event_row.evt_time-timedelta(minutes=time_win)) &
                             (cat_tab.evt_time <= event_row.evt_time+timedelta(minutes=time_win))].to_list()
    # FALSE if empty time window
    if len(time_sel) < 1:
        logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}: "
                         f"False (no events within time window)")
        return None
    # list CAT events within time window
    win_tab = cat_tab.loc[time_sel].reset_index(drop=True)
    # calculate OT difference between all CAT events in window and EPIC event
    time_dif = [abs(xx - event_row.evt_time).total_seconds() for xx in win_tab.evt_time]
    # calculate LOC difference between all CAT events in window and EPIC event
    loca_dif = [gdist.distance((event_row.evt_lat, event_row.evt_lon), (win_tab.evt_lat[j], win_tab.evt_lon[j])).km for
                j in range(len(win_tab))]
    # test both criteria
    and_test = [a and b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                        [True if xx < loca_lim else False for xx in loca_dif])]
    or_test = [a or b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                      [True if xx < loca_lim else False for xx in loca_dif])]
    # FALSE if both OT and LOC criteria not fullfilled
    kkk = []
    if sum(or_test) == 0:
        logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}:"
                         f" False (no OT/LOC match)")
        logger_name.debug(f" {win_tab.evt_id.to_list()}")
        logger_name.debug(f" {time_dif} {time_lim} s")
        logger_name.debug(f" {loca_dif} {loca_lim} km")
        return None
    elif sum(and_test) == 0 and sum(or_test) != 0:
        # find index where one criterion is verified
        kkk = [k for k, xx in enumerate(or_test) if xx]
    elif sum(and_test) != 0 and sum(or_test) != 0:
        # find index where both criteria are verified
        kkk = [k for k, xx in enumerate(and_test) if xx]
    if not kkk:
        return None
    # taking match with min. OT difference if >1 matches
    if len(kkk) > 1:
        kkk = time_dif.index(min(time_dif))
    else:
        kkk = kkk[0]
    # MATCH if either one or both OT and LOC criteria fullfilled
    logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}: "
                     f"Match with {datetime.strftime(win_tab.evt_time[kkk], '%d/%m/%Y %H:%M:%S')}"
                     f" [{win_tab.evt_id[kkk]}]")
    logger_name.debug(f" {win_tab.evt_type[kkk]}: dt = {time_dif[kkk]:.2f} s, dr = {loca_dif[kkk]:.2f} km")
    return win_tab.evt_id[kkk]


def get_emsc_match(client, event_row, cat_tab, time_win, time_lim, logger_name):
    """
    :param client: FDSN client to retrieve regional and teleseismic events
    :param event_row: DataFrame row for EPIC alert to match
    :param cat_tab: DataFrame containing catalogue events
    :param time_win: half-length of time window [s]
    :param time_lim: max. origin time error [s]
    :param logger_name: log file to use
    :return: catalogue event ID and event description if found both EMSC AND catalogue matches for that teleseism
    """
    llim = 5000.
    # look for M>3 events
    try:
        emsc_tab = client.get_events(starttime=UTCDateTime(event_row.evt_time-timedelta(minutes=time_win)),
                                     endtime=UTCDateTime(event_row.evt_time+timedelta(minutes=time_win)),
                                     minmagnitude=3., includearrivals=False, orderby='time-asc')
    except:
        emsc_tab = []
    if emsc_tab:
        # OT difference between EPIC event and EMSC events
        time_dif = [abs(np.datetime64(e.preferred_origin().time)-event_row.evt_time).total_seconds() for e in
                    emsc_tab]
        # LOC difference between EPIC event and EMSC events
        loca_dif = [gdist.distance((e.preferred_origin().latitude, e.preferred_origin().longitude),
                                   (event_row.evt_lat, event_row.evt_lon)).km for e in emsc_tab]
        # check if EMSC events match EPIC alert according to OT max. difference
        and_test = [a and b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                            [True if xx < time_lim*5.8 else False for xx in loca_dif])]
        # case at least one match is found
        if sum(and_test) >= 1:
            if sum(and_test) == 1:
                jjj = and_test.index(True)
            else:
                # index of events verifying both criteria
                ind = [ii for ii, t in enumerate(and_test) if t is True]
                # index of event with minimum OT error
                jjj = time_dif.index(min([time_dif[ii] for ii in ind]))
            # create DataFrame for EMSC match in order to use 'get_catalogue_event_match'
            evt_tab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'evt_time': pd.Series(dtype='datetime64[ms]'),
                                    'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64')})
            # fill DataFrame
            evt_tab.loc[evt_tab.shape[0]] = [event_row.evt_id, np.datetime64(emsc_tab[jjj].preferred_origin().time),
                                             emsc_tab[jjj].preferred_origin().latitude,
                                             emsc_tab[jjj].preferred_origin().longitude]
            # log match
            logger_name.info(f"[{event_row.evt_id}] "
                             f"{datetime.utcfromtimestamp(emsc_tab[jjj].preferred_origin().time.timestamp).strftime('%d/%m/%Y %H:%M:%S')}: "
                             f"{emsc_tab[jjj].event_descriptions[0].text} [{emsc_tab[jjj].preferred_origin().latitude},"
                             f" {emsc_tab[jjj].preferred_origin().longitude}] "
                             f"M{emsc_tab[jjj].preferred_magnitude().mag:4.2f}")
            # find corresponding catalogue match with less restricting location error criteria
            gsi_id = get_catalogue_event_match(cat_tab, evt_tab.loc[0], time_win, time_lim, 100., logger)
            if gsi_id and gsi_id != '':
                # calculate distance between EMSC and catalogue matching events
                dloc = gdist.distance((cat_tab[cat_tab.evt_id == gsi_id].evt_lat.iloc[0],
                                       cat_tab[cat_tab.evt_id == gsi_id].evt_lon.iloc[0]),
                                      (emsc_tab[jjj].preferred_origin().latitude,
                                       emsc_tab[jjj].preferred_origin().longitude)).km
                # case catalogue match is a regional/local event
                if cat_tab[cat_tab.evt_id == gsi_id].evt_type.iloc[0] != 'Teleseism' and dloc > llim:
                    logger_name.warning(f'[{event_row.evt_id}] LOC difference too large between EMSC and CAT'
                                        f'\n {cat_tab[cat_tab.evt_id == gsi_id].evt_type.iloc[0]}: {dloc:.1f} km')
                    return None, None
            return gsi_id, f'{emsc_tab[jjj].event_descriptions[0].text.lower()} ' \
                           f'M{emsc_tab[jjj].preferred_magnitude().mag:4.2f}'
        # case no match is found
        else:
            logger_name.info(f'[{event_row.evt_id}] No EMSC events with OT/LOC match')
            logger_name.debug(f' {[e.event_descriptions[0].text for e in emsc_tab]}')
            logger_name.debug(f' {time_dif} {time_lim} s')
            logger_name.debug(f' {loca_dif} {llim} km')
            return None, None
    else:
        logger_name.info(f'[{event_row.evt_id}] No EMSC events within time window')
        return None, None


def get_traces_full(client, event_row, time_win, work_dir):
    """
    :param client: FDSN client to retrieve wavefroms from
    :param event_row: DataFrame row for EPIC event
    :param time_win: half-length of time window [s]
    :param work_dir: working directory
    :return: full path of created .mseed file
    """
    # location for .mseed event files
    ddir = f'{work_dir}/events'
    # full path of .mseed file
    mseed = f'{ddir}/{event_row.evt_id}.full.raw.mseed'
    # load .mseed data
    if path.exists(mseed) == 0:
        stream = client.get_waveforms(network='IS,GE', station='*', location='*', channel='HHZ,ENZ',
                                      starttime=UTCDateTime(event_row.evt_time-timedelta(minutes=time_win)),
                                      endtime=UTCDateTime(event_row.evt_time+timedelta(minutes=time_win)))
        # resample to 50 Hz
        stream.resample(50., window='hann')
        # write miniSEED file
        stream.write(mseed)
    return mseed


def get_traces_deci(event_row, time_win, work_dir):
    """
    :param event_row: dataframe row for EPIC event
    :param time_win: half-length of time window
    :param work_dir: working directory
    :return: full path of created .mseed file
    """
    # location for jqdata archive
    adir = '/net/172.16.46.200/archive/jqdata/archive'
    # load .mseed data
    mseed = f"{work_dir}/events/{(event_row.evt_id if event_row.evt_id != '' else event_row.cat_id)}.deci.raw.mseed"
    if path.exists(mseed) == 0:
        # import miniSEED file
        tbeg = str(datetime.strftime(event_row.evt_time-timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(event_row.evt_time+timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
        os.system(f'scart -dsE -n "IS, GE" -c "(H|E)(H|N)Z" -t "{tbeg}~{tend}" {adir} > {mseed}')
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
        # if event_row.evt_id != '1008':
        #     tbeg = str(datetime.strftime(event_row.evt_time - timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
        #     tend = str(datetime.strftime(event_row.evt_time + timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
        #     print('Get .mseed file manually from jplayback:')
        #     print(f' scart -dsE -n "IS, GE" -c "(H|E)(H|N)Z" -t "{tbeg}~{tend}" {adir}'
        #           f' > {mseed.replace(f"{ddir}/", "")}')
        #     mseed = ''
        #     exit()
    else:
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    return mseed


def process_mseed(mseed_in, sta_inv):
    """
    :param mseed_in: full path to .mseed file to process
    :param sta_inv: station inventory
    :return: data streamer with event info.
    """
    # read raw miniSEED file
    stream = read(mseed_in)
    # output file name for processed waveforms
    mseed_out = mseed_in.replace('.raw', '')
    # checking if file exists
    if path.exists(mseed_out) != 0:
        logger.info(f' Waveform data already processed: {mseed_out}')
        return mseed_out
    # problem in inventory with IS.KRPN.21.ENZ (also noisy station); ENE in streamer instead of ENZ
    for tr in stream.select(network='IS', station='KRPN', channel='ENE'):
        stream.remove(tr)
    # problem in inventory from Mar 2022 with IS.HRFI.22.HHZ; empty location in streamer
    for tr in stream.select(network='IS', location='', station='HRFI', channel='HHZ'):
        stream.remove(tr)
    # no GE data since 30/04/2023
    if '1643' in mseed_in or '1683' in mseed_in in mseed_in:
        for tr in stream.select(network='GE', station='BALJ'):
            stream.remove(tr)
    # location issue with IS.KFSB.20/22.HHZ (also noisy station); two co-existing locations
    for tr in stream.select(network='IS', station='KFSB', channel='HHZ'):
        stream.remove(tr)
    # remove Meiron stations with HHZ channel (not in inventory)
    for tr in stream.select(network='IS', channel='HHZ'):
        if 'MMA' in tr.stats.station or 'MMB' in tr.stats.station or 'MMC' in tr.stats.station:
            stream.remove(tr)
    # merge different segments of same traces if needed
    stream = stream.merge(fill_value='interpolate')
    # remove response from all traces
    try:
        stream.remove_response(output='VEL', inventory=sta_inv)
    except:
        print(mseed_in)
        for tr in stream:
            print(tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel)
            tr.remove_response(output='VEL', inventory=sta_inv)
    # apply taper to all traces
    stream.taper(max_percentage=.5, type='cosine', max_length=30., side='left')
    # apply high-pass filter to all traces
    stream.filter('highpass', freq=1./10.)
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply channel priorities (HHZ>ENZ) and remove partial traces
    for tr in stream:
        # if HHZ channel, remove all others
        if tr.stats.channel == 'HHZ':
            x = stream.select(station=tr.stats.station, network=tr.stats.network, channel='ENZ')
            if x:
                stream.remove(x[0])
        # # remove incomplete traces
        # if tr.stats.npts != int(time_win * 60. * 2. / tr.stats.delta) + 1:
        #     y = stream.select(station=tr.stats.station, network=tr.stats.network, channel=tr.stats.channel,
        #                       location=tr.stats.location)
        #     if y:
        #         stream.remove(tr)
    # write miniSEED file
    stream.write(mseed_out)
    return mseed_out


def add_event_data(stream, event_row, sta_inv, velocity_model):
    """
    :param stream: data streamer of waveforms to plot
    :param event_row: DataFrame row for reference event
    :param sta_inv: station inventory
    :param velocity_model: 1-D reference Earth model to compute theoretical arrival times
    :return: data streamer of waveforms with event data
    """
    # Earth radius
    earth_rad = 6371.
    # calculate distances and theoretical arrival times
    for tr in stream:
        # find station of interest
        sta = sta_inv.select(station=tr.stats.station)
        # compute distance from reference event
        dist = gdist.distance((event_row.evt_lat, event_row.evt_lon),
                              (sta.networks[0].stations[0].latitude, sta.networks[0].stations[0].longitude))
        tr.stats.distance = dist.m
        # compute theoretical travel times for reference event
        if 'evt_dep' in event_row:
            # P first arrivals
            tp = velocity_model.get_travel_times(source_depth_in_km=event_row.evt_dep,
                                                 distance_in_degree=dist.km/(2.*np.pi*earth_rad/360.),
                                                 phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
            if len(tp) != 0:
                tr.stats['theo_tp'] = np.datetime64(event_row.evt_time+timedelta(seconds=tp[0].time))
            else:
                tr.stats['theo_tp'] = np.datetime64('NaT')
            # # S first arrivals
            # ts = velocity_model.get_travel_times(source_depth_in_km=event_row.evt_dep.iloc[0],
            #                                      distance_in_degree=dist.km/(2.*np.pi*earth_rad/360.),
            #                                      phase_list=['s', 'S', 'Sg', 'Sn', 'Sdiff'])
            # if len(ts) != 0:
            #     tr.stats['theo_ts'] = np.datetime64(event_row.evt_time.iloc[0]+timedelta(seconds=ts[0].time))
            # else:
            #     tr.stats['theo_ts'] = np.datetime64('NaT')
    return stream


def plot_event_rec_sec(stream, time_win=10., alert_row=None, match_row=None, trig_tab=None,
                       fig_name=None, event_label=None, earth_model='iasp91'):
    """
    :param stream: data streamer of waveforms to plot
    :param time_win: plot time window for waveforms
    :param alert_row: DataFrame row for EPIC event
    :param match_row: DataFrame row for matching catalogue event (if any)
    :param trig_tab: DataFrame containing all triggers to show over waveforms
    :param fig_name: full-path file name if figure to be saved (and not displayed)
    :param event_label: string to replace the 'evt_type' string to label the matching event
    :param earth_model: Earth model used for theoretical arrivals
    :return:
    """
    # plotting mode
    mpl.use('Agg')
    # create figure & axis
    ff, axis = plt.subplots(squeeze=True, figsize=(18, 9), dpi=200)
    # set x-axis limits from beginning
    axis.set_xlim([alert_row.evt_time - timedelta(minutes=time_win),
                   alert_row.evt_time + timedelta(minutes=time_win)])
    # event markers
    h1 = []
    h2 = []
    h31 = []
    h32 = []
    h5 = []
    # initialise counter
    n_trace = 0
    # EPIC event marker
    h4, = axis.plot([alert_row.evt_time, alert_row.evt_time], [-1, len(stream)],
                    color='purple', alpha=.7, label='Alert')
    # matching event marker (if any)
    if not match_row.empty:
        # in the extremely rare occasion there are two alerts for the same event (e.g. EPIC IDs 157 & 158)
        for _, evt in match_row.iterrows():
            # x-axis limits
            axis.set_xlim([min(evt.evt_time - timedelta(minutes=1),
                               alert_row.evt_time - timedelta(minutes=time_win)),
                           alert_row.evt_time + timedelta(minutes=time_win)])
            # time marker for matching event
            h5, = axis.plot([evt.evt_time, evt.evt_time], [-1, len(stream)],
                            color='green', alpha=.7, label='Match')
            # event description
            axis.text(evt.evt_time, len(stream)*1.005, (event_label or alert_row.cat_type),
                      color='green', ha='center', va='bottom')
    # loop over traces
    for trace in stream:
        # time vector
        t_vec = np.arange(0, len(trace.data))*np.timedelta64(int(trace.stats.delta * 1000), '[ms]') \
                + np.datetime64(str(trace.stats.starttime)[:-1])
        # plot waveform
        h1, = axis.plot(t_vec, trace.data/trace.max()+n_trace, color='grey', alpha=.7, label='Velocity')
        # display station name & distance
        ht1 = axis.text(axis.get_xlim()[1]+(axis.get_xlim()[1]-axis.get_xlim()[0])/150., n_trace,
                        f"{trace.stats.distance/1000.:.2f} km", ha='left', va='center', fontsize=5)
        ht2 = axis.text(axis.get_xlim()[0]-(axis.get_xlim()[1]-axis.get_xlim()[0])/150., n_trace,
                        f"{trace.stats.network}.{trace.stats.station}."
                        f"{trace.stats.location}.{trace.stats.channel}",
                        ha='right', va='center', fontsize=5)
        # EPIC trigger
        if not trig_tab.empty:
            sta_tab = trig_tab[trig_tab.trig_sta == trace.stats.station].reset_index(drop=True)
            if not sta_tab.empty:
                for iii in range(len(sta_tab)):
                    h2, = axis.plot([sta_tab.trig_time[iii], sta_tab.trig_time[iii]], [n_trace-1, n_trace+1],
                                    color='red', alpha=.7, label=f'Triggers ({len(trig_tab)})')
                ht1.set(color='red')
                ht2.set(color='red')
        # theoretical arrival
        if 'theo_tp' in trace.stats and not np.isnat(trace.stats.theo_tp):
            h31, = axis.plot([trace.stats.theo_tp, trace.stats.theo_tp], [n_trace-1, n_trace+1],
                             ':b', alpha=.7, label=f'{earth_model}')
        # if 'theo_ts' in trace.stats and not np.isnat(trace.stats.theo_ts):
        #     h32, = axis.plot([trace.stats.theo_ts, trace.stats.theo_ts], [n_trace-1, n_trace+1],
        #                      ':b', alpha=.7, label=f'{earth_model} (S)')
        n_trace += 1
    if not np.isnan(alert_row.evt_nup):
        # title
        if alert_row.evt_nup == 1:
            exp = 'st'
        elif alert_row.evt_nup == 2:
            exp = 'nd'
        else:
            exp = 'th'
        axis.set_title(f"{alert_row.evt_id}: M{alert_row.evt_max_mag:3.2f} "
                       f"{datetime.strftime(alert_row.evt_time, '%d/%m/%Y %H:%M:%S')}\n"
                       f"{alert_row.evt_nsta} stations in {alert_row.evt_nup}{exp} update", fontweight='bold', y=1.02)
    else:
        axis.set_title(f"{alert_row.evt_id}: M{alert_row.evt_max_mag:3.2f} "
                       f"{datetime.strftime(alert_row.evt_time, '%d/%m/%Y %H:%M:%S')}", fontweight='bold', y=1.02)
    # axes
    axis.grid(which='both', axis='both')
    axis.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # legend
    hh = [h4, h1]
    if h2:
        hh.append(h2)
    if h31:
        hh.append(h31)
    if h32:
        hh.append(h32)
    if h5:
        hh.append(h5)
    axis.legend(handles=hh, loc='lower left', fontsize=8)
    # replace y-axis tick labels with station names
    axis.set_yticks(np.arange(0, n_trace, 1))
    axis.set_yticklabels([])
    axis.set_ylim([-1, n_trace])
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # save figure
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    plt.close()


def plot_epic_check_summary(epic_tab, cat_tab, source, miss_map, fig_name=None):
    """
    :param epic_tab: DataFrame containing all EPIC events
    :param cat_tab: DataFrame containing all catalogue events
    :param source: TRUA'A system being checked
    :param miss_map: min/max latitude and longitude for missed events
    :param fig_name: full-path file name if figure to be saved (and not displayed)
    :return:
    """
    # geographical area
    local_map = [29., 34., 33.5, 36.5]
    regional_map = [23., 43., 25., 45.]
    # prepare table
    tab_txt = []
    tab_col = []
    loc_err = pd.Series(dtype=float)
    for ii, epic in epic_tab.iterrows():
        if epic.evt_id != '' and epic.cat_id != '':
            # match
            dr = gdist.distance(
                (epic.evt_lat, epic.evt_lon), (cat_tab[cat_tab.evt_id == epic.cat_id].evt_lat.iloc[0],
                                               cat_tab[cat_tab.evt_id == epic.cat_id].evt_lon.iloc[0])).km
            dt = (cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]-epic.evt_time).total_seconds()
            if epic.cat_type == 'Teleseism':
                tab_txt.append(
                    [f"{datetime.strftime(cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0], '%d/%m/%Y %H:%M:%S')}",
                     f"{cat_tab[cat_tab.evt_id == epic.cat_id].evt_mag.iloc[0]:.2f}".replace('nan', 'NaN'),
                     f"{epic.cat_type}", epic.evt_id, f"{epic.evt_1st_mag:.2f}", f"{epic.evt_max_mag:.2f}",
                     f"{(epic.evt_1st_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}",
                     f"{(epic.evt_M42_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}".replace('nan', 'NaN'),
                     f"{(epic.evt_M45_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}".replace('nan', 'NaN'),
                     f"{dt:.1f}", f"{dr:.1f}"])
                loc_err.loc[loc_err.shape[0]] = np.nan
                if epic.evt_max_mag >= smag:
                    tab_col.append(
                        ['none', 'none', 'none', 'blue', 'none', '.7', 'none', 'none', 'none', 'none', 'none'])
                else:
                    tab_col.append(
                        ['none', 'none', 'none', 'blue', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            else:
                tab_txt.append(
                    [f"{datetime.strftime(cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0], '%d/%m/%Y %H:%M:%S')}",
                     f"{cat_tab[cat_tab.evt_id == epic.cat_id].evt_mag.iloc[0]:.2f}".replace('nan', 'NaN'),
                     f"{epic.cat_type}", epic.evt_id, f"{epic.evt_1st_mag:.2f}", f"{epic.evt_max_mag:.2f}",
                     f"{(epic.evt_1st_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}",
                     f"{(epic.evt_M42_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}".replace('nan', 'NaN'),
                     f"{(epic.evt_M45_time-cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0]).total_seconds():.1f}".replace('nan', 'NaN'),
                     f"{dt:.1f}", f"{dr:.1f}"])
                loc_err.loc[loc_err.shape[0]] = dr
                if epic.cat_type == 'Felt':
                    if epic.evt_max_mag >= smag:
                        tab_col.append(['none', 'none', 'darkorange', 'limegreen', 'none',
                                        '.7', 'none', 'none', 'none', 'none', 'none'])
                    else:
                        tab_col.append(['none', 'none', 'darkorange', 'limegreen', 'none',
                                        'none', 'none', 'none', 'none', 'none', 'none'])
                else:
                    if epic.evt_max_mag >= smag:
                        tab_col.append(
                            ['none', 'none', 'none', 'limegreen', 'none', '.7', 'none', 'none', 'none', 'none', 'none'])
                    else:
                        tab_col.append(['none', 'none', 'none', 'limegreen', 'none',
                                        'none', 'none', 'none', 'none', 'none', 'none'])
        elif epic.evt_id != '' and epic.cat_id == '':
            # false
            tab_txt.append(
                [f"{np.datetime64('NaT')}", 'NaN', '\u2013', epic.evt_id, f"{epic.evt_1st_mag:.2f}",
                 f"{epic.evt_max_mag:.2f}", 'NaN', 'NaN', 'NaN', '\u2013', '\u2013'])
            if epic.evt_max_mag > smag:
                tab_col.append(
                    ['none', 'none', 'none', 'darkorange', 'none', '.7', 'none', 'none', 'none', 'none', 'none'])
            else:
                tab_col.append(
                    ['none', 'none', 'none', 'darkorange', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            loc_err.loc[loc_err.shape[0]] = np.nan
        elif epic.evt_id == '' and epic.cat_id != '':
            # ignoring missed M<[amag] CAT events (no alerts)
            if epic.cat_mag < 2.5:
                continue
            # missed
            tab_txt.append([f"{datetime.strftime(cat_tab[cat_tab.evt_id == epic.cat_id].evt_time.iloc[0], '%d/%m/%Y %H:%M:%S')}",
                            f"{cat_tab[cat_tab.evt_id == epic.cat_id].evt_mag.iloc[0]:.2f}".replace('nan', 'NaN'),
                            f"{epic.cat_type}", '\u2013', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '\u2013', '\u2013'])
            if epic.cat_type == 'Felt':
                tab_col.append(
                    ['none', 'none', 'darkorange', 'red', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            else:
                tab_col.append(
                    ['none', 'none', 'none', 'red', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            loc_err.loc[loc_err.shape[0]] = np.nan
    # plotting mode
    mpl.use('Agg')
    # create figure & axis
    ff, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    # AXIS 1: SUMMARY MAP
    map1 = Basemap(projection='cyl', llcrnrlon=local_map[2], llcrnrlat=local_map[0],
                   urcrnrlon=local_map[3], urcrnrlat=local_map[1], ax=axis1, resolution='i')
    # draw map
    map1.drawmapboundary(fill_color='none')
    map1.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels = [left,right,top,bottom])
    map1.drawparallels(
        np.arange(np.floor(map1.llcrnrlat), np.floor(map1.urcrnrlat) + 1., 1.), labels=[True, False, True, False])
    map1.drawmeridians(
        np.arange(np.floor(map1.llcrnrlon), np.floor(map1.urcrnrlon) + 1., 1.), labels=[True, False, False, True])
    # faults
    flts_id = ''
    if user_name == 'sysop':
        if 'osboxes' in os.environ['SESSION_MANAGER']:
            flts_id = open('/home/sysop/.seiscomp/bna/ActiveFaults/activefaults.bna', 'r')
        else:
            flts_id = open('/home/sysop/.seiscomp/spatial/vector/ActiveFaults/activefaults.bna', 'r')
    elif user_name == 'lewis':
        flts_id = open('/home/lewis/.seiscomp/bna/ActiveFaults/activefaults.bna', 'r')
    flts = flts_id.readlines()
    flts_id.close()
    flt = []
    for iii in range(len(flts)):
        if re.search('"', flts[iii]):
            flt = pd.DataFrame({'lat': pd.Series(dtype='float64'), 'lon': pd.Series(dtype='float64')})
        elif iii < len(flts) - 1 and re.search('"', flts[iii + 1]):
            axis1.plot(flt.lon, flt.lat, '.6', linewidth=.5, label='Faults', zorder=2)
        else:
            l_line = flts[iii].split(',')
            flt.loc[flt.shape[0]] = [float(l_line[1]), float(l_line[0])]
    # AXIS 3: INSET MAP FOR REGIONAL SETTINGS
    axis3 = inset_axes(axis1, '30%', '20%', loc='lower left')
    map2 = Basemap(projection='cyl', llcrnrlon=regional_map[2], llcrnrlat=regional_map[0],
                   urcrnrlon=regional_map[3], urcrnrlat=regional_map[1], resolution='l', ax=axis3)
    # draw map
    map2.drawmapboundary(fill_color='white')
    # fill continents
    map2.fillcontinents(color='0.8', lake_color='white')
    # highlight area of interest
    axis3.plot([miss_map[2], miss_map[2], miss_map[3], miss_map[3], miss_map[2]],
               [miss_map[0], miss_map[1], miss_map[1], miss_map[0], miss_map[0]], color='red')
    # slicing alert table according to alert categories
    match = epic_tab[(epic_tab.evt_id != '') & (epic_tab.cat_id != '')
                     & (epic_tab.cat_type != 'Teleseism')].reset_index(drop=True)
    false = epic_tab[(epic_tab.evt_id != '') & (epic_tab.cat_id == '')].reset_index(drop=True)
    missed = epic_tab[(epic_tab.evt_id == '') & (epic_tab.cat_id != '')
                      & (epic_tab.cat_mag >= 2.5)].reset_index(drop=True)
    teles = epic_tab[epic_tab.cat_type == 'Teleseism'].reset_index(drop=True)
    felt = cat_tab[cat_tab.evt_type == 'Felt']
    if not teles.empty:
        # removing possible duplicates (rare case of two alerts for one event, e.g. EPIC IDs 157 & 158)
        tele = teles[~teles.cat_id.duplicated()]
        for t, tevt in tele.iterrows():
            axin = axis1.inset_axes((.75, .8-t*.2, .25, .25), zorder=120)
            map3 = Basemap(projection='ortho', resolution='l', ax=axin,
                           lon_0=cat_tab[cat_tab.evt_id == tevt.cat_id].iloc[0].evt_lon,
                           lat_0=cat_tab[cat_tab.evt_id == tevt.cat_id].iloc[0].evt_lat)
            # converting event lat/lon
            x, y = map3(cat_tab[cat_tab.evt_id == tevt.cat_id].iloc[0].evt_lon,
                        cat_tab[cat_tab.evt_id == tevt.cat_id].iloc[0].evt_lat)
            # draw map
            map3.drawmapboundary(fill_color='white')
            # fill continents
            map3.fillcontinents(color='0.8', lake_color='white')
            # show teleseismic event
            axin.plot(x, y, 'o', color='blue', markeredgecolor='black', markersize=5, alpha=.7)
            # search EMSC database for event data
            emsc = Client('EMSC').get_events(
                starttime=UTCDateTime(datetime.strptime(tevt.cat_id, 'emsc%Y%m%d%H%M')-timedelta(minutes=1)),
                endtime=UTCDateTime(datetime.strptime(tevt.cat_id, 'emsc%Y%m%d%H%M')+timedelta(minutes=1)))
            if len(emsc.events) > 1:
                k = 0
                for k, evt in enumerate(emsc.events):
                    if evt.preferred_origin().latitude == cat_tab[cat_tab.evt_id == tevt.cat_id].iloc[0].evt_lat:
                        break
                emsc = emsc.events[k]
            else:
                emsc = emsc.events[0]
            # display event region and magnitude
            axin.text(x, y-y/3., f'{str(tevt.evt_id)}: M{emsc.preferred_magnitude().mag:3.1f}\n'
                                 f'{emsc.event_descriptions[0].text.lower().title()}',
                      color='blue', ha='center', va='center', fontsize=6)
    # Matches (lines)
    axis1.plot([[cat_tab.evt_lon[cat_tab.evt_id == eid].iloc[0] for eid in match.cat_id], match.evt_lon.to_list()],
               [[cat_tab.evt_lat[cat_tab.evt_id == eid].iloc[0] for eid in match.cat_id], match.evt_lat.to_list()],
               color='black', alpha=.7, linewidth=.5)
    axis3.plot([[cat_tab.evt_lon[cat_tab.evt_id == eid].iloc[0] for eid in match.cat_id], match.evt_lon.to_list()],
               [[cat_tab.evt_lat[cat_tab.evt_id == eid].iloc[0] for eid in match.cat_id], match.evt_lat.to_list()],
               color='black', alpha=.7, linewidth=.5)
    # Matches (EPIC)
    axis1.scatter(match.evt_lon, match.evt_lat, s=25, color='limegreen', edgecolors='black', alpha=.7, zorder=30)
    axis3.scatter(match.evt_lon, match.evt_lat, s=10, color='limegreen', edgecolors='none', alpha=.7, zorder=30)
    # Matches (CAT)
    axis1.scatter([cat_tab[cat_tab.evt_id == eid].iloc[0].evt_lon for eid in match.cat_id],
                  [cat_tab[cat_tab.evt_id == eid].iloc[0].evt_lat for eid in match.cat_id],
                  s=4, c='black', edgecolors='none', alpha=.7, zorder=20)
    # Felt catalogue events
    if not felt.empty:
        # local map
        axis1.scatter(felt.evt_lon, felt.evt_lat, s=225, c='red', marker='*', edgecolors='black', alpha=.7, zorder=90)
        # regional map
        axis3.scatter(felt.evt_lon, felt.evt_lat, s=110, c='red', marker='*', edgecolors='none', alpha=.7, zorder=90)
        # display magnitude
        for _, e in felt.iterrows():
            # local map
            axis1.text(e.evt_lon, e.evt_lat+.1, f'M{e.evt_mag:3.1f}',
                       color='red', fontsize=8, ha='center', va='center', clip_on=True)
            # regional map
            if e.evt_lat < mgrd[0] or e.evt_lat > mgrd[1] or e.evt_lon < mgrd[2] or e.evt_lon > mgrd[3]:
                axis3.text(e.evt_lon, e.evt_lat+1, f'M{e.evt_mag:3.1f}',
                           color='red', fontsize=5, ha='center', va='center', clip_on=True)
    # Teleseismic events
    if not teles.empty:
        axis1.scatter(teles.evt_lon, teles.evt_lat, s=100, c='blue', edgecolors='black', alpha=.7, zorder=90)
        # regional map
        axis3.scatter(teles.evt_lon, teles.evt_lat, s=10, c='blue', edgecolors='none', alpha=.7, zorder=90)
        # display event IDs
        for _, e in teles.iterrows():
            axis1.text(e.evt_lon, e.evt_lat+.1, e.evt_id,
                       color='black', fontsize=8, ha='center', va='center', clip_on=True)
            # regional map
            if e.evt_lat < mgrd[0] or e.evt_lat > mgrd[1] or e.evt_lon < mgrd[2] or e.evt_lon > mgrd[3]:
                axis3.text(e.evt_lon, e.evt_lat-1, e.evt_id,
                           color='black', fontsize=5, ha='center', va='center', clip_on=True)
    # False alerts
    if not false.empty:
        # local map
        axis1.scatter(false.evt_lon, false.evt_lat, s=100, c='darkorange', edgecolors='black', alpha=.7, zorder=80)
        # regional map
        axis3.scatter(false.evt_lon, false.evt_lat, s=10, color='darkorange', edgecolors='none', alpha=.7, zorder=80)
        # display event IDs
        for _, e in false.iterrows():
            # local map
            axis1.text(e.evt_lon, e.evt_lat+.1, e.evt_id,
                       color='black', fontsize=8, ha='center', va='center', clip_on=True)
            # regional map
            if e.evt_lat < mgrd[0] or e.evt_lat > mgrd[1] or e.evt_lon < mgrd[2] or e.evt_lon > mgrd[3]:
                axis3.text(e.evt_lon, e.evt_lat-1, e.evt_id,
                           color='black', fontsize=5, ha='center', va='center', clip_on=True)
    # Missed events
    if not missed.empty:
        # missed M<smag events
        axis1.scatter(missed[missed.cat_mag < smag].evt_lon, missed[missed.cat_mag < smag].evt_lat,
                      s=10, color='red', edgecolors='black', alpha=.7, zorder=100)
        # regional map
        axis3.scatter(missed[missed.cat_mag < smag].evt_lon, missed[missed.cat_mag < smag].evt_lat,
                      s=1, marker='.', color='red', edgecolors='red', alpha=.7, zorder=100)
        # missed M>smag events
        # local map
        axis1.scatter(missed[missed.cat_mag >= smag].evt_lon, missed[missed.cat_mag >= smag].evt_lat,
                      s=100, color='red', edgecolors='black', alpha=.7, zorder=100)
        # regional map
        axis3.scatter(missed[missed.cat_mag >= smag].evt_lon, missed[missed.cat_mag >= smag].evt_lat,
                      s=50, color='red', edgecolors='none', alpha=.7, zorder=100)
        # display magnitude
        for _, e in missed.iterrows():
            # local map
            axis1.text(e.evt_lon, e.evt_lat-.08, f'M{e.cat_mag:3.1f}',
                       color='red', fontsize=8, ha='center', va='center', clip_on=True)
            # regional map
            if e.evt_lat < mgrd[0] or e.evt_lat > mgrd[1] or e.evt_lon < mgrd[2] or e.evt_lon > mgrd[3]:
                axis3.text(e.evt_lon, e.evt_lat-1, f'M{e.cat_mag:3.1f}',
                           color='black', fontsize=5, ha='center', va='center', clip_on=True)
    # display EPIC alert numbers
    axis1.text(local_map[2]+.05, local_map[1]-.1,
               f"EPIC [Ast=1]: {len(epic_tab[epic_tab.evt_id != ''])}", ha='left', va='center', zorder=200)
    axis1.text(local_map[2]+.05, local_map[1]-.25,
               f"EPIC [M>{smag}]: {len(epic_tab[(epic_tab.evt_id != '') & (epic_tab.evt_max_mag >= smag)])}",
               ha='left', va='center', zorder=200)
    axis1.text(local_map[2]+.05, local_map[1]-.4, f"Match: {len(match)}",
               color='limegreen', ha='left', va='center', zorder=200)
    axis1.text(local_map[2]+.05, local_map[1]-.55, f"False [Ast=1]: {len(false)}",
               color='darkorange', ha='left', va='center', zorder=200)
    axis1.text(local_map[2]+.05, local_map[1]-.7, f"False [M>{smag}]: {len(false[false.evt_max_mag >= smag])}",
               color='darkorange', ha='left', va='center', zorder=200)
    axis1.text(local_map[2]+.05, local_map[1]-.85, f"Teleseism: {len(teles)}",
               color='blue', ha='left', va='center', zorder=200)
    # display CAT event numbers
    axis1.text(local_map[3]-.05, local_map[0]+.7, f"CAT [M>{amag}]: {len(cat_tab[cat_tab.evt_mag >= amag])}",
               ha='right', va='center', zorder=200)
    axis1.text(local_map[3]-.05, local_map[0]+.55, f"CAT [M>{smag}]: {len(cat_tab[cat_tab.evt_mag >= smag])}",
               ha='right', va='center', zorder=200)
    axis1.text(local_map[3]-.05, local_map[0]+.4, f"Felt: {len(felt)}",
               color='darkorange', ha='right', va='center', zorder=200)
    axis1.text(local_map[3]-.05, local_map[0]+.25, f"Missed [M>{amag}]: {len(missed)}",
               color='red', ha='right', va='center', zorder=200)
    axis1.text(local_map[3]-.05, local_map[0]+.1, f"Missed [M>{smag}]: {len(missed[missed.cat_mag >= smag])}",
               color='red', ha='right', va='center', zorder=200)
    # AXIS 2: SUMMARY TABLE
    axis2.axis('off')
    # show table
    ftab = axis2.table(cellText=tab_txt, loc='upper center', cellLoc='center', cellColours=tab_col,
                       colLabels=('Origin Time', 'M', 'Event Type', 'EPIC', '1st M',
                                  'Max M', 'M>2.5 dT', 'M>4.2 dT', 'M>4.5 dT', 'OT Err', 'Loc Err'),
                       colWidths=[.16, .06, .1, .05, .06, .06, .09, .09, .09, .08, .08])
    ftab.auto_set_font_size(False)
    ftab.set_fontsize(8)
    # title
    axis2.set_title(f"{datetime.strftime(tper, '%B %Y')}\n{source}", x=.16)
    # save figure
    ff.subplots_adjust(wspace=-.2)
    plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
    plt.close()


# EPIC source directories
src = 'eew-b-jer'
# working directory
if user_name == 'lewis':
    wdir = '/home/lewis'
else:
    wdir = '/mnt/c/Users/lewiss/Documents/Research'
# EPIC working directory
edir = f"{wdir}/EPIC/{src.split('-')[2].upper() + '-' + src.split('-')[1].upper()}"
# CAT event pictures directory for Missed events
pdir = '/net/172.16.46.200/archive/monitor/events'
print(f"EPIC directory: {edir}")
print()

# velocity model for theoretical arrivals
rmod = 'iasp91'
model = TauPyModel(model=rmod)

# SELECTION PARAMETERS
# OT difference limit
tlim = 30.          # [s]
# LOC difference limit [PERMISSIVE]
rlim = tlim*5.8     # [km]; Vp from PREM
# waveform & events time window
twin = 10.          # [min]
# # minimum number of triggers [UNUSED]
# ntri = 4            # [# triggers]
# alert magnitude
amag = 2.5          # [MAG]
# significance magnitude
smag = 3.5          # [MAG]

# EMSC FDSN database
emsc_client = Client('EMSC')
# ISN FDSN database
isn_client = Client('http://172.16.46.102:8181/')
# retrieve ISN station inventory
isn_inv = isn_client.get_stations(network='IS,GE', channel='ENZ,HHZ', level='response')
# selection area based on ISN distribution for missed events
mgrd = [29., 34., 33., 37.]

# variables to loop over months
m1 = datetime.strptime('Aug 2022', '%b %Y')
m2 = datetime.strptime('Sep 2022', '%b %Y')
dm = (m2.year-m1.year)*12+m2.month-m1.month
# loop over months
for m in range(dm):
    # month of interest
    tper = m1 + relativedelta(months=m)
    print(f"Processing: {datetime.strftime(tper, '%B %Y')}")
    # logging file
    if path.exists(f"{edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log") != 0:
        os.remove(f"{edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log")
    logger = my_custom_logger(f"{edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log", level=logging.DEBUG)
    # list of days
    dtab = [tper + timedelta(days=idx) for idx in range((tper+relativedelta(months=+1)-tper).days)]

    ####################
    # CATALOGUE EVENTS #
    # retrieve all CAT events for the month
    ctab = get_catalogue_events_db(isn_client, tper, tper+relativedelta(months=+1))
    # list local CAT events (will be used later for missed events)
    cloc = ctab[(ctab.evt_lat >= mgrd[0]) & (ctab.evt_lat <= mgrd[1]) &
                (ctab.evt_lon >= mgrd[2]) & (ctab.evt_lon <= mgrd[3])].reset_index(drop=True)
    # display number of catalogue events (and number of M>smag local events)
    logger.info(f"Loaded {len(ctab)} catalogue events ({len(cloc[cloc.evt_mag >= smag])} local M\u2265{smag})")
    logger.info('-'*60)

    ###############
    # EPIC ALERTS #
    # retrieve all EPIC alerts SENT (Ast=1) for the month
    etab = get_epic_alerts_db(src, tper, tper + relativedelta(months=+1), True)
    if etab.empty:
        logger.info(f'No EPIC alerts')
        continue
    logger.info(f'Loaded {len(etab)} EPIC alerts:')
    for _, alert in etab.iterrows():
        logger.info(f"[{alert.evt_id}] {datetime.strftime(alert.evt_time, '%d/%m/%Y %H:%M:%S')}: "
                    f"[{alert.evt_lat}, {alert.evt_lon}] M{alert.evt_max_mag:4.2f}"
                    f" {type(alert.evt_alert_mag) is float}")
    logger.info('-'*60)

    #################################
    # LOOKING FOR CATALOGUE MATCHES #
    logger.info('Matching EPIC alerts with catalogue events:')
    # add column in EPIC table for event IDs of matching catalogue events and corresponding event type
    etab = etab.assign(cat_id=pd.Series([''] * len(etab), dtype='string'))
    etab = etab.assign(cat_type=pd.Series([''] * len(etab), dtype='string'))
    etab = etab.assign(cat_mag=pd.Series([np.nan] * len(etab), dtype='float64'))
    # loop over EPIC alerts
    for i, alert in etab.iterrows():
        # # to investigate a specific EPIC event
        # if alert.evt_id != '1284':
        #     continue
        # looking for CAT match
        gsid = get_catalogue_event_match(ctab, alert, twin, tlim, rlim, logger)
        # unmatching events with wrong CAT matches
        wtab = pd.read_csv(f'{edir}/wrong-match-cat.csv', dtype={'evt_id': 'string'})
        if not wtab[wtab.evt_id == alert.evt_id].empty:
            gsid = ''
        del wtab
        # store CAT event ID into EPIC table if any
        etab.loc[i, 'cat_id'] = (gsid or '')
        # store CAT event type into EPIC table if any
        etab.loc[i, 'cat_type'] = (ctab[ctab.evt_id == gsid].evt_type.iloc[0].title() if gsid else '')
        # store CAT magnitude into EPIC table if any
        etab.loc[i, 'cat_mag'] = (ctab[ctab.evt_id == gsid].evt_mag.iloc[0] if gsid else np.nan)
        # initialise variables
        fig = ''
        evt_lbl = ''
        # checking origin time error and send events with large errors to 'checks' directory
        if gsid and abs(alert.evt_time-ctab.loc[ctab.evt_id == gsid].evt_time.iloc[0]).total_seconds() > tlim:
            logger.warning('Large origin time error!')
            # figure file name
            fig = f'{edir}/checks/{alert.evt_id}.png'
        # checking (1st alert time - CAT origin time)>0 and sending dT<0 event to 'checks' directory
        elif gsid and (alert.evt_1st_time-ctab[ctab.evt_id == gsid].evt_time.iloc[0]).total_seconds() < 0:
            logger.warning('Negative alert delay!')
            # figure file name
            fig = f'{edir}/checks/{alert.evt_id}.png'
        # checking false alerts with EMSC database
        if not gsid:
            if alert.evt_id == '930':
                gsid, desc = get_emsc_match(emsc_client, alert, ctab, twin * 2., tlim * 30. * 2., logger)
            else:
                gsid, desc = get_emsc_match(emsc_client, alert, ctab, twin * 2., tlim * 30., logger)
            # unmatching events with wrong EMSC matches
            wtab = pd.read_csv(f'{edir}/wrong-match-emsc.csv', dtype={'evt_id': 'string'})
            if not wtab[wtab.evt_id == alert.evt_id].empty:
                gsid = ''
            del wtab
            # store EMSC event ID into EPIC table if any
            etab.loc[i, 'cat_id'] = (gsid or '')
            # store EMSC event type into EPIC table if any
            etab.loc[i, 'cat_type'] = (ctab[ctab.evt_id == gsid].evt_type.iloc[0].title() if gsid else '')
            # store EMSC magnitude into EPIC table if any
            etab.loc[i, 'cat_mag'] = (ctab[ctab.evt_id == gsid].evt_mag.iloc[0] if gsid else np.nan)
            # event description for record section plot
            evt_lbl = (desc.title() if desc is not None else '')
            # figure file name
            fig = f'{edir}/false/{alert.evt_id}.png'
            # hard-coded exceptions in case no EMSC match
            if not gsid:
                # read exception file for teleseismic events without match
                mtab = pd.read_csv(
                    f'{edir}/missing-match-emsc.csv', dtype={'evt_id': 'string', 'emsc_id': 'string',
                                                             'emsc_type': 'string', 'emsc_time': 'string',
                                                             'emsc_lat': 'float64', 'emsc_lon': 'float64',
                                                             'emsc_dep': 'float64', 'emsc_mag': 'float64',
                                                             'emsc_reg': 'string'}, parse_dates=['emsc_time'])
                # select EPIC alert ID
                mtab = mtab[mtab.evt_id == alert.evt_id].reset_index(drop=True)
                if not mtab.empty:
                    # add EMSC event details to EPIC table
                    etab.loc[i, 'cat_id'] = mtab.emsc_id.iloc[0]
                    etab.loc[i, 'cat_type'] = str(mtab.emsc_type.iloc[0]).title()
                    # add EMSC event details to CAT table
                    ctab.loc[ctab.shape[0]] = [mtab.emsc_id.iloc[0], mtab.emsc_time.iloc[0], mtab.emsc_lat.iloc[0],
                                               mtab.emsc_lon.iloc[0], mtab.emsc_dep.iloc[0], mtab.emsc_mag.iloc[0],
                                               mtab.emsc_type.iloc[0]]
                    # event label for plotting (to replace 'evt_type')
                    evt_lbl = f'{mtab.emsc_reg.iloc[0]} M{mtab.emsc_mag.iloc[0]:3.1f}'
                    # sort table
                    ctab.sort_values(by='evt_time', ignore_index=True, inplace=True)
        # plotting record sections in case EPIC alert was False or to be checked
        if fig and ('false' in fig or 'check' in fig) and path.exists(fig) == 0:
            # get waveform data
            mfile = get_traces_deci(alert, twin, edir)
            # missing decimated data around EPIC alert 1009's OT
            if alert.evt_id == '1009' or '904':
                mfile = ''
            # case no decimated data
            if mfile == '':
                logger.warning(' No decimated data, retrieving full resolution data')
                mfile = get_traces_full(isn_client, alert, twin, edir)
            # process waveform data
            nfile = process_mseed(mfile, isn_inv)
            # read retrieved waveforms
            isn_traces = read(nfile)
            # initialise variables
            evid = None
            # case Match but to check
            if 'check' in fig:
                # add event data to waveform stream
                isn_traces = add_event_data(isn_traces, ctab[ctab.evt_id == gsid].iloc[0], isn_inv, model)
            # case False with EMSC match
            elif 'false' in fig:
                # case EMSC and CAT matches
                if gsid:
                    # add event data to waveform stream
                    isn_traces = add_event_data(isn_traces, ctab[ctab.evt_id == gsid].iloc[0], isn_inv, model)
                # case only EMSC match
                else:
                    # add event data to waveform stream
                    isn_traces = add_event_data(
                        isn_traces, (ctab[ctab.evt_id == evid].iloc[0]
                                     if evid else etab[etab.evt_id == alert.evt_id].iloc[0]),
                        isn_inv, model)
            # case False without EMSC match
            else:
                # add event data to waveform stream
                isn_traces = add_event_data(isn_traces, etab[etab.evt_id == alert.evt_id].iloc[0], isn_inv, model)
            # get triggers from database for EPIC alert
            ttab = get_epic_triggers_db(src, alert.evt_id, alert.evt_ver)
            if ttab.empty:
                logger.warning(f'No triggers for EPIC event {alert.evt_id}')
            # indexes sorted according to distance (descending)
            isn_traces.sort(['distance'], reverse=True)
            # matching CAT events with large OT/LOC error (Jordan explosions)
            wtab = pd.read_csv(f'{edir}/wrong-false-cat.csv', dtype={'evt_id': 'string', 'emsc_id': 'string',
                                                                     'cat_type': 'string'})
            # select EPIC alert ID
            wtab = wtab[wtab.evt_id == alert.evt_id].reset_index(drop=True)
            if not wtab.empty:
                gsid = wtab.cat_id.iloc[0]
                etab.loc[i, 'cat_id'] = wtab.cat_id.iloc[0]
                etab.loc[i, 'cat_type'] = wtab.cat_type.iloc[0].title()
            del wtab
            # table row containing Match event data
            if ctab[ctab.evt_id == gsid].empty and evid:
                evt_row = ctab[ctab.evt_id == evid]
            else:
                evt_row = ctab[ctab.evt_id == gsid]
            # plot record section
            plot_event_rec_sec(isn_traces, twin, alert, evt_row, ttab, fig, evt_lbl, rmod)
            logger.info(f'Record section saved: {fig}')
        # else:
        #     logger.info(f" Record section existing: {fig}")
    logger.info('-----------------------------------------------------------------------------------------------------')

    #######################################
    # LOOKING FOR MISSED CATALOGUE EVENTS #
    if not cloc.empty:
        logger.info('Checking missed detections:')
        # loop over local Earthquake & Felt events
        for _, catalogue in cloc[(cloc.evt_type == 'Earthquake') | (cloc.evt_type == 'Felt')].iterrows():
            # retrieve EPIC alerts NOT SENT (Ast=0) arount CAT event OT
            ntab = get_epic_alerts_db(src, tper, tper + relativedelta(months=+1), False)
            # remove EPIC alerts already in EPIC table (i.e., alert versions before reaching M2.5)
            ntab = ntab[~ntab.evt_id.isin(etab.evt_id)]
            # look for CAT match with EPIC alerts
            gsid = ''
            for _, alert in ntab.iterrows():
                gsid = get_catalogue_event_match(ctab[ctab.evt_id == catalogue.evt_id], alert, twin, tlim, rlim, logger)
                if gsid:
                    break
            # ignore catalogue event if Match was found in EPIC alerts NOT SENT
            if gsid:
                break
            # ignoring false alerts
            if etab[etab.cat_id == catalogue.evt_id].empty:
                # ignore duplicate events caused by changes in event IDing
                if '.' in catalogue.evt_id and not etab[etab.cat_id == catalogue.evt_id[0:-3]].empty:
                    continue
                # add empty entry to EPIC table with missing GSI event ID
                etab.loc[etab.shape[0]] = \
                    ['', np.nan, catalogue.evt_time, catalogue.evt_lat, catalogue.evt_lon, 0, 0, np.nan,
                     np.datetime64('NaT'), np.nan, np.datetime64('NaT'), np.nan, np.datetime64('NaT'),
                     np.datetime64('NaT'), np.datetime64('NaT'), catalogue.evt_id, str(catalogue.evt_type).title(),
                     catalogue.evt_mag]
                # to plot in case M>smag
                if catalogue.evt_mag >= smag:
                    # create fake event row to use in Missed event plot
                    otab = etab.copy(deep=True)
                    otab.loc[otab.shape[0]] = [catalogue.evt_id, np.nan, catalogue.evt_time, catalogue.evt_lat,
                                               catalogue.evt_lon, 0, 0, catalogue.evt_mag, np.datetime64('NaT'), np.nan,
                                               np.datetime64('NaT'), np.nan, np.datetime64('NaT'), np.datetime64('NaT'),
                                               np.datetime64('NaT'), catalogue.evt_id, str(catalogue.evt_type).title(),
                                               catalogue.evt_mag]
                    logger.info(f"[{catalogue.evt_type}] Missed: "
                                f"{datetime.strftime(catalogue.evt_time, '%d/%m/%Y %H:%M:%S')} "
                                f"M{catalogue.evt_mag:3.1f}")
                    # retrieve event picture (from plot_daily_events.py routine)
                    if path.exists(pdir):
                        ddir = datetime.strftime(catalogue.evt_time, '%Y/%m/%d')
                        if path.exists(f'{pdir}/{ddir}/{catalogue.evt_id}.png'):
                            os.system(f'cp -p {pdir}/{ddir}/{catalogue.evt_id}.png {edir}/missed/')
                            logger.info(f'Event picture copied: {pdir}/{ddir}/{catalogue.evt_id}.png')
                        else:
                            # foigure file name
                            fig = f'{edir}/missed/{catalogue.evt_id}.png'
                            # skipping if figure already exists
                            if path.exists(fig) != 0:
                                continue
                            logger.warning(f'No CAT event picture for {catalogue.evt_id}')
                            # get waveform data
                            mfile = get_traces_deci(otab[otab.evt_id == catalogue.evt_id].iloc[0], twin, edir)
                            # case no decimated data
                            if mfile == '':
                                logger.warning(' No decimated data, retrieving full resolution data')
                                mfile = get_traces_full(
                                    isn_client, otab.loc[otab.evt_id == catalogue.evt_id].iloc[0], twin, edir)
                            # process waveform data
                            nfile = process_mseed(mfile, isn_inv)
                            # read retrieved waveforms
                            isn_traces = read(nfile)
                            # add event data to waveform stream
                            isn_traces = add_event_data(
                                isn_traces, otab[otab.cat_id == catalogue.evt_id].iloc[0], isn_inv, model)
                            # plot record section
                            plot_event_rec_sec(
                                isn_traces, twin, etab.loc[etab.cat_id == catalogue.evt_id].iloc[0], pd.DataFrame({'A': []}),
                                pd.DataFrame({'A': []}), fig, catalogue.evt_type, rmod)
                    else:
                        logger.warning(f'Could not access CAT event pictures directory: {pdir}')
        logger.info('-------------------------------------------------------------------------------------------------')

    ########################
    # BUILD SUMMARY FIGURE #
    # sorting by OT & resetting indexes in EPIC table
    etab = etab.sort_values(by=['evt_time'], ignore_index=True)
    etab.to_csv(f"{edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.csv", float_format='%.4f', index=False)
    plot_epic_check_summary(etab, ctab, src, mgrd, f"{edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.png")
    logger.info(f"Monthly summary saved: {edir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.png")
    logger.info('-----------------------------------------------------------------------------------------------------')
