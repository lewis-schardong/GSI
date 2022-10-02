########################################################################################################################
import os
import sys
import re
import logging
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopy.distance as gdist
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from matplotlib.dates import DateFormatter
from obspy.core import UTCDateTime
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees as km2deg
sys.path.append(f'/home/{os.getlogin()}/olmost/TRUAA/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TRUAA.settings")
import django
django.setup()
from epic.models import EVENT, TRIGGER


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    format_string = "%(asctime)s.%(msecs)03d \u2013 %(levelname)s \u2013 %(funcName)s: %(message)s"
    log_format = logging.Formatter(format_string, '%Y-%m-%d %H:%M:%S')
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    log.addHandler(file_handler)
    return log


def get_epic_events_db(source='eew-b-jer', tbeg=datetime.now() - timedelta(days=7), tend=datetime.now()):
    """
    :param source: EPIC system source
    :param tbeg: starting date/time
    :param tend: ending date/time
    :return: DataFrame containing all EPIC alerts for given period (default is one week)
    """
    # variables to query
    variables = ['eventid', 'ver', 'evlat', 'evlon', 'dep', 'mag', 'time', 'nS', 'alert_time', 'Ast']
    # output event table
    evt_tab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'evt_ver': pd.Series(dtype='int64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                            'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'), 'evt_nsta': pd.Series(dtype='int64'),
                            'evt_maxmag_mag': pd.Series(dtype='float64'), 'evt_maxmag_time': pd.Series(dtype='datetime64[ns]'),
                            'evt_alert_mag': pd.Series(dtype='float64'), 'evt_alert_time': pd.Series(dtype='datetime64[ns]'),
                            'evt_1stver_mag': pd.Series(dtype='float64'), 'evt_1stver_time': pd.Series(dtype='datetime64[ns]')})
    # get alert list from database for month of interest
    al_list = EVENT.objects.filter(Ast=True, source__contains=source, alert_time__range=[str(UTCDateTime(tbeg)), str(UTCDateTime(tend))]) \
        .values('eventid').order_by('eventid').distinct()
    # loop over all retrieved alerts
    for item in al_list:
        # maximum magnitude alert
        al1 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source).order_by('-mag').values(
            *variables).first()
        # first alert
        al2 = EVENT.objects.filter(Ast=True, eventid=item['eventid'], source__contains=source).order_by('time').values(
            *variables).first()
        # first version (always non-empty)
        al3 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source).order_by('ver').values(
            *variables).first()
        # last alert (for location; always non-empty; can equal first version)
        al4 = EVENT.objects.filter(eventid=item['eventid'], source__contains=source).order_by('-ver').values(
            *variables).first()
        evt_tab.loc[evt_tab.shape[0]] = [al4['eventid'], al4['ver'],
                                         np.datetime64(str(al4['time']).replace('+00:00', '')),
                                         al4['evlat'], al4['evlon'], al4['nS'],
                                         al1['mag'], np.datetime64(str(al1['time']).replace('+00:00', '')),
                                         al2['mag'], np.datetime64(str(al2['time']).replace('+00:00', '')),
                                         al3['mag'], np.datetime64(str(al3['time']).replace('+00:00', ''))]
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
                             'trig_time': pd.Series(dtype='datetime64[ns]'), 'trig_dis': pd.Series(dtype='float64'),
                             'trig_azi': pd.Series(dtype='float64')})
    # get alert list from database for month of interest
    tr_list = TRIGGER.objects.filter(source__contains=source, eventid=evt_id, ver=evt_ver).values(*variables).order_by(
        'trigger_time')
    # loop over all retrieved triggers
    for item in tr_list:
        trig_tab.loc[trig_tab.shape[0]] = [item['eventid'], item['ver'], item['sta'], item['chan'], item['net'],
                                           item['loc'],
                                           item['lat'], item['lon'], item['trigger_time'], item['distkm'],
                                           item['azimuth']]
    # sort catalogue events by OT
    trig_tab = trig_tab.sort_values(by=['trig_time'], ignore_index=True)
    return trig_tab


def get_cat_events_db(tbeg=datetime.now() - timedelta(days=7), tend=datetime.now()):
    """
    :param tbeg: starting date/time
    :param tend: ending date/time
    :return: DataFrame containing all catalogue events for given period (default is one week)
    """
    # load catalogue events for period of interest
    evt_lst = Client('http://172.16.46.140:8181/').get_events(starttime=UTCDateTime(tbeg), endtime=UTCDateTime(tend), includearrivals=False)
    # output event table
    cat_tab = pd.DataFrame({'evt_id': pd.Series(dtype='string'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                            'evt_lat': pd.Series(dtype='float64'),
                            'evt_lon': pd.Series(dtype='float64'), 'evt_dep': pd.Series(dtype='float64'),
                            'evt_mag': pd.Series(dtype='float64'), 'evt_type': pd.Series(dtype='string')})
    for event_row in evt_lst:
        # removing relocated (rel*) events from list
        if str(event_row.resource_id).replace('smi:org.gfz-potsdam.de/geofon/', '')[0:3] == 'rel':
            continue
        # event magnitude
        if not event_row.preferred_magnitude():
            emag = np.nan
        else:
            emag = event_row.preferred_magnitude().mag
        # event type
        if not event_row.event_descriptions:
            etyp = str(event_row.event_type)
        else:
            etyp = event_row.event_descriptions[0].text
        # fill catalogue events table
        cat_tab.loc[cat_tab.shape[0]] = [str(event_row.resource_id).replace('smi:org.gfz-potsdam.de/geofon/', ''),
                                         np.datetime64(event_row.preferred_origin().time), event_row.preferred_origin().latitude,
                                         event_row.preferred_origin().longitude, event_row.preferred_origin().depth / 1000., emag, etyp]
    # sort catalogue events by OT
    cat_tab = cat_tab.sort_values(by=['evt_time'], ignore_index=True)
    return cat_tab


def get_cat_event_match(cat_tab, event_row, time_win, time_lim, loca_lim, logger_name):
    """
    :param cat_tab: DataFrame of catalogue events
    :param event_row: DataFrame row for EPIC alert to match
    :param time_win: half-length of time window
    :param time_lim: max. origin time error [s]
    :param loca_lim: max. location error [km]
    :param logger_name: log file to use
    :return: catalogue event ID if match; None if not
    """
    # cat_tab: table of catalogue events
    # event_row: dataframe row for EPIC alert to match
    # time_win: half-length of time window
    # time_lim: max. origin time error [s]
    # loca_lim: max. location error [km]
    # ________________________________________________________ #
    # look for indexes of catalogue events within time window
    time_sel = cat_tab.index[(cat_tab.evt_time > event_row.evt_time - timedelta(minutes=time_win)) &
                             (cat_tab.evt_time <= event_row.evt_time + timedelta(minutes=time_win))].to_list()
    # FALSE if empty time window
    if len(time_sel) < 1:
        logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}: False (no events within time window)")
        return None
    # list catalogue events within time window
    win_tab = cat_tab.loc[time_sel].reset_index(drop=True)
    # calculate OT error between all catalogue events in window and EPIC event
    time_dif = [abs(xx - event_row.evt_time).total_seconds() for xx in win_tab.evt_time]
    # calculate LOC error between all GSI events in window and EPIC event
    loca_dif = [gdist.distance((event_row.evt_lat, event_row.evt_lon), (win_tab.evt_lat[j], win_tab.evt_lon[j])).km for
                j in range(len(win_tab))]
    # test both criteria
    and_test = [a and b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                        [True if xx < loca_lim else False for xx in loca_dif])]
    or_test = [a or b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                      [True if xx < loca_lim else False for xx in loca_dif])]
    # FALSE if origin-time or distance criteria not fullfilled
    kkk = []
    if sum(or_test) == 0:
        logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}: False (no OT/LOC match)")
        logger_name.debug(f" {win_tab.evt_id.to_list()}")
        logger_name.debug(f" {time_dif} {time_lim} s")
        logger_name.debug(f" {loca_dif} {loca_lim} km")
        return None
    elif sum(and_test) == 0 and sum(or_test) != 0:  # loca_lim != time_lim*5.8 and ():
        # find index where one criterion is verified
        kkk = [k for k, xx in enumerate(or_test) if xx]
    elif sum(and_test) != 0 and sum(or_test) != 0:
        # find index where both criteria are verified
        kkk = [k for k, xx in enumerate(and_test) if xx]
    if not kkk:
        return None
    if len(kkk) > 1:
        kkk = time_dif.index(min(time_dif))
    else:
        kkk = kkk[0]
    # MATCH if both origin-time and distance criteria fullfilled
    logger_name.info(f"[{event_row.evt_id}] {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}: "
                     f"Match with {datetime.strftime(win_tab.evt_time[kkk], '%d/%m/%Y %H:%M:%S')} [{win_tab.evt_id[kkk]}]")
    logger_name.debug(f" {win_tab.evt_type[kkk]}: dt = {abs(win_tab.evt_time[kkk] - event_row.evt_time).total_seconds():.2f} s, "
                      f"dr = {gdist.distance((win_tab.evt_lat[kkk], win_tab.evt_lon[kkk]), (event_row.evt_lat, event_row.evt_lon)).km:.2f} km")
    return win_tab.evt_id[kkk]


def get_tele_event_match(client, event_row, cat_tab, time_win, time_lim, logger_name):
    """
    :param client: database client to retrieve teleseisms from
    :param event_row: DataFrame row for EPIC alert to match
    :param cat_tab: DataFrame containing catalogue events
    :param time_win: half-length of time window [s]
    :param time_lim: max. origin time error [s]
    :param logger_name: log file to use
    :return: catalogue event ID and event description if found both teleseism AND catalogue matches for that teleseism
    """
    llim = 5000.
    # look for M>5 events
    tele_tab = client.get_events(starttime=UTCDateTime(event_row.evt_time - timedelta(minutes=time_win)),
                                 endtime=UTCDateTime(event_row.evt_time + timedelta(minutes=time_win)), minmagnitude=3,
                                 includearrivals=False)
    # try:
    #     # look for M>5 events
    #     tele_tab = client.get_events(starttime=UTCDateTime(event_row.evt_time - timedelta(minutes=time_win)),
    #                                  endtime=UTCDateTime(event_row.evt_time + timedelta(minutes=time_win)), minmagnitude=5, includearrivals=False)
    # except:
    #     try:
    #         # look for M>4 events
    #         tele_tab = client.get_events(starttime=UTCDateTime(event_row.evt_time - timedelta(minutes=time_win)),
    #                                      endtime=UTCDateTime(event_row.evt_time + timedelta(minutes=time_win)), minmagnitude=4, includearrivals=False)
    #     except:
    #         try:
    #             # look for M>3 events
    #             tele_tab = client.get_events(starttime=UTCDateTime(event_row.evt_time - timedelta(minutes=time_win)),
    #                                          endtime=UTCDateTime(event_row.evt_time + timedelta(minutes=time_win)), minmagnitude=3, includearrivals=False)
    #         except:
    #             print(f'[{event_row.evt_id}] No M>3 EMSC events within time window')
    #             return None
    if tele_tab:
        # check if EMSC events match EPIC alert (using OT error only)
        time_dif = [abs(np.datetime64(e.preferred_origin().time) - event_row.evt_time).total_seconds() for e in
                    tele_tab]
        loca_dif = [gdist.distance((e.preferred_origin().latitude, e.preferred_origin().longitude),
                                   (event_row.evt_lat, event_row.evt_lon)).km for e in tele_tab]
        and_test = [a and b for a, b in zip([True if xx < time_lim else False for xx in time_dif],
                                            [True if xx < time_lim * 5.8 else False for xx in loca_dif])]
        if sum(and_test) == 1:
            jjj = and_test.index(True)
            evt_tab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                    'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64')})
            evt_tab.loc[evt_tab.shape[0]] = [event_row.evt_id, np.datetime64(tele_tab[jjj].preferred_origin().time),
                                             tele_tab[jjj].preferred_origin().latitude,
                                             tele_tab[jjj].preferred_origin().longitude]
            logger_name.info(f"[{event_row.evt_id}] {tele_tab[jjj].event_descriptions[0].text}"
                             f" {datetime.utcfromtimestamp(tele_tab[jjj].preferred_origin().time.timestamp).strftime('%d/%m/%Y %H:%M:%S')}: "
                             f"[{tele_tab[jjj].preferred_origin().latitude}, {tele_tab[jjj].preferred_origin().longitude}] "
                             f"M{tele_tab[jjj].preferred_magnitude().mag:4.2f}")
            gsi_id = get_cat_event_match(cat_tab, evt_tab.loc[0], time_win, time_lim, 100., logger)
            if gsi_id and gsi_id != '':
                dist = gdist.distance((cat_tab[cat_tab.evt_id == gsi_id].evt_lat.to_list()[0],
                                       cat_tab[cat_tab.evt_id == gsi_id].evt_lon.to_list()[0]),
                                      (tele_tab[jjj].preferred_origin().latitude,
                                       tele_tab[jjj].preferred_origin().longitude)).km
                if cat_tab[cat_tab.evt_id == gsi_id].evt_type.to_list()[0] != 'TELESEISM' and dist > llim:
                    logger_name.warning(f'[{event_row.evt_id}] Too close to be a teleseism'
                                        f'\n {cat_tab[cat_tab.evt_id == gsi_id].evt_type.to_list()[0]}: {dist:.1f} km')
                    return None, None
            return gsi_id, f'{tele_tab[jjj].event_descriptions[0].text} M{tele_tab[jjj].preferred_magnitude().mag:4.2f}'
        elif sum(and_test) > 1:
            # index of events verifying both criteria
            ind = [ii for ii, t in enumerate(and_test) if t is True]
            # index of event with minimum OT error
            jjj = time_dif.index(min([time_dif[ii] for ii in ind]))
            evt_tab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                    'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64')})
            evt_tab.loc[evt_tab.shape[0]] = [event_row.evt_id, np.datetime64(tele_tab[jjj].preferred_origin().time),
                                             tele_tab[jjj].preferred_origin().latitude,
                                             tele_tab[jjj].preferred_origin().longitude]
            logger_name.info(f"[{event_row.evt_id}] {tele_tab[jjj].event_descriptions[0].text}"
                             f" {datetime.utcfromtimestamp(tele_tab[jjj].preferred_origin().time.timestamp).strftime('%d/%m/%Y %H:%M:%S')}: "
                             f"[{tele_tab[jjj].preferred_origin().latitude}, {tele_tab[jjj].preferred_origin().longitude}] "
                             f"M{tele_tab[jjj].preferred_magnitude().mag:4.2f}")
            gsi_id = get_cat_event_match(cat_tab, evt_tab.loc[0], time_win, time_lim, 100., logger)
            if gsi_id and gsi_id != '':
                dist = gdist.distance((cat_tab[cat_tab.evt_id == gsi_id].evt_lat.to_list()[0],
                                       cat_tab[cat_tab.evt_id == gsi_id].evt_lon.to_list()[0]),
                                      (tele_tab[jjj].preferred_origin().latitude,
                                       tele_tab[jjj].preferred_origin().longitude)).km
                if cat_tab[cat_tab.evt_id == gsi_id].evt_type.to_list()[0] != 'TELESEISM' and dist > llim:
                    logger_name.warning(f'[{event_row.evt_id}] Too close to be a teleseism'
                                        f'\n {cat_tab[cat_tab.evt_id == gsi_id].evt_type.to_list()[0]}: {dist:.1f} km')
                    return None, None
            return gsi_id, f'{tele_tab[jjj].event_descriptions[0].text} M{tele_tab[jjj].preferred_magnitude().mag:4.2f}'
        else:
            logger_name.info(f'[{event_row.evt_id}] No EMSC events with OT/LOC match')
            logger_name.debug(f' {[e.event_descriptions[0].text for e in tele_tab]}')
            logger_name.debug(f' {time_dif} {time_lim} s')
            logger_name.debug(f' {loca_dif} {llim} km')
            return None, None
    else:
        logger_name.info(f'[{event_row.evt_id}] No EMSC events within time window')
        return None, None


def get_traces_full(event_row, time_win):
    """
    :param event_row: DataFrame row for EPIC event
    :param time_win: half-length of time window [s]
    :return: full path of created .mseed file
    """
    # ________________________________________________________ #
    # event_row: dataframe row for EPIC event
    # match_row: dataframe row for matching event
    # time_win: half-length of time window
    # sta_inv: station inventory
    # ________________________________________________________ #
    # location for .mseed event files
    ddir = f'/home/{os.getlogin()}/GoogleDrive/Research/GSI/Data/events'
    # full path of .mseed file
    mseed = f'{ddir}/{event_row.evt_id}.full.raw.mseed'
    # load .mseed data
    if path.exists(mseed) == 0:
        stream = Client('http://172.16.46.102:8181/')\
            .get_waveforms('IS,GE', '*', '*', 'HHZ,ENZ',
                           starttime=UTCDateTime(event_row.evt_time - timedelta(minutes=time_win)),
                           endtime=UTCDateTime(event_row.evt_time + timedelta(minutes=time_win))).merge()
        # write miniSEED file
        stream.write(mseed)
    return mseed


def get_traces_deci(event_row, time_win):
    """
    :param event_row: dataframe row for EPIC event
    :param time_win: half-length of time window
    :return: full path of created .mseed file
    """
    # location for .mseed event files
    ddir = f'/home/{os.getlogin()}/GoogleDrive/Research/GSI/Data/events'
    # location for jqdata archive
    adir = '/net/172.16.46.200/archive/jqdata/archive'
    # load .mseed data
    mseed = f'{ddir}/{event_row.evt_id}.deci.raw.mseed'
    if path.exists(mseed) == 0:
        if os.getlogin() != 'lewis':
            # import miniSEED file
            tbeg = str(datetime.strftime(event_row.evt_time - timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
            tend = str(datetime.strftime(event_row.evt_time + timedelta(minutes=time_win), '%Y-%m-%d %H:%M:%S'))
            os.system(f'scart -dsE -n "IS, GE" -c "(H|E)(H|N)Z" -t "{tbeg}~{tend}" {adir} > {mseed}')
            if os.path.getsize(mseed) == 0:
                os.remove(mseed)
                mseed = ''
        elif event_row.evt_id != '1008':
            t1 = str(datetime.strftime(evt.evt_time - timedelta(minutes=twin), '%Y-%m-%d %H:%M:%S'))
            t2 = str(datetime.strftime(evt.evt_time + timedelta(minutes=twin), '%Y-%m-%d %H:%M:%S'))
            print('Get .mseed file manually from jplayback:')
            print(f' scart -dsE -n "IS, GE" -c "(H|E)(H|N)Z" -t "{t1}~{t2}" '
                  f'/net/172.16.46.200/archive/jqdata/archive > {mseed.replace(f"{ddir}/", "")}')
            mseed = ''
            exit()
    else:
        if os.path.getsize(f'{ddir}/{event_row.evt_id}.deci.raw.mseed') == 0:
            os.remove(f'{ddir}/{event_row.evt_id}.deci.raw.mseed')
            mseed = ''
    return mseed


def process_mseed(mseed_in, sta_inv, time_win):
    """
    :param mseed_in: full path to .mseed file to process
    :param sta_inv: station inventory
    :param time_win: half-length of time window
    :return: data streamer with event info.
    """
    # read raw miniSEED file
    stream = read(mseed_in)
    mseed_out = mseed_in.replace('.raw', '')
    if path.exists(mseed_out) != 0:
        logger.info(f' Waveform data already processed: {mseed_out}')
        return mseed_out
    # remove problematic channels
    y = stream.select(network='IS', station='KRPN')
    for tr in y:
        stream.remove(tr)
    for tr in stream.select(network='IS', station='EIL', channel='BHZ'):
        stream.remove(tr)
    for tr in stream.select(network='IS', station='GEM', channel='BHZ'):
        stream.remove(tr)
    for tr in stream.select(network='IS', station='KFSB', channel='HHZ', location='22'):
        stream.remove(tr)
    for tr in stream.select(network='IS', station='HRFI', channel='HHZ', location=''):
        stream.remove(tr)
    # remove Meiron stations with HHZ channel (not in inventory)
    for t in stream.select(network='IS', channel='HHZ'):
        if 'MMA' in t.stats.station or 'MMB' in t.stats.station or 'MMC' in t.stats.station:
            stream.remove(t)
    # remove response from all traces
    stream.remove_response(output='VEL', inventory=sta_inv)
    # apply taper to all traces
    stream.taper(max_percentage=.5, type='cosine', max_length=10., side='left')
    # remove trend from all traces
    stream.detrend('spline', order=3, dspline=500)
    # apply channel priorities (HHZ>ENZ) and remove partial traces
    for tr in stream:
        # if HHZ channel, remove all others
        if tr.stats.channel == 'HHZ':
            x = stream.select(station=tr.stats.station, network=tr.stats.network, channel='ENZ')
            if x:
                stream.remove(x[0])
        # remove incomplete traces
        if tr.stats.npts != int(time_win * 60. * 2. / tr.stats.delta) + 1:
            y = stream.select(station=tr.stats.station, network=tr.stats.network, channel=tr.stats.channel,
                              location=tr.stats.location)
            if y:
                stream.remove(tr)
    # write miniSEED file
    stream.write(mseed_out)
    return mseed_out


def add_event_data(stream, event_row, sta_inv):
    # calculate distances and theoretical arrival times
    for tr in stream:
        # find station of interest
        # sta = station_tab[station_tab.sta == tr.stats.station]
        sta = sta_inv.select(station=tr.stats.station)
        # compute distance from EMSC event
        dist = gdist.distance((event_row.evt_lat.to_list()[0], event_row.evt_lon.to_list()[0]),
                              (sta.networks[0].stations[0].latitude, sta.networks[0].stations[0].longitude))
        tr.stats.distance = dist.m
        # compute theoretical travel times for EMSC event
        if 'evt_dep' in event_row:
            x = model.get_travel_times(source_depth_in_km=event_row.evt_dep.to_list()[0],
                                       distance_in_degree=dist.km / (2 * np.pi * rrad / 360),
                                       phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
            if len(x) != 0:
                tr.stats['theo_tt'] = np.datetime64(event_row.evt_time.to_list()[0] + timedelta(seconds=x[0].time))
            else:
                tr.stats['theo_tt'] = np.datetime64('NaT')
    return stream


def plot_event_rec_sec(stream, index=None, event_row=None, match_row=None, trig_tab=None, fig_name=None, event_label=None):
    """
    :param stream: data streamer of waveforms to plot
    :param index: index to re-order streamer
    :param event_row: DataFrame row for EPIC event
    :param match_row: DataFrame row for matching catalogue event (if any)
    :param trig_tab: DataFrame containing all triggers to show over waveforms
    :param fig_name: full-path file name if figure to be saved (and not displayed)
    :param event_label: string to replace the 'evt_type' string to label the matching event
    :return:
    """
    # plotting mode
    if fig_name:
        mpl.use('Agg')
    # create figure & axis
    ff, axis = plt.subplots(squeeze=True, figsize=(18, 9), dpi=200)
    if not fig_name:
        plt.show(block=False)
    # set x-axis limits from beginning
    axis.set_xlim([event_row.evt_time - timedelta(minutes=twin), event_row.evt_time + timedelta(minutes=twin)])
    # event markers
    h1 = []
    h2 = []
    h3 = []
    h5 = []
    # initialise counter
    n_trace = 0
    # EPIC event marker
    h4, = axis.plot([event_row.evt_time, event_row.evt_time], [-1, len(stream)], color='purple', alpha=.7, label='Alert')
    # matching event marker (if any)
    if not match_row.empty:
        # x-axis limits
        axis.set_xlim([min(match_row.evt_time.to_list()[0] - timedelta(minutes=1), event_row.evt_time - timedelta(minutes=twin)),
                       event_row.evt_time + timedelta(minutes=twin)])
        # time marker for matching event
        h5, = axis.plot([match_row.evt_time, match_row.evt_time], [-1, len(stream)], color='green', alpha=.7, label='Match')
        # event description
        axis.text(match_row.evt_time, len(stream), (event_label or match_row.evt_type.to_list()[0]), color='green', ha='center', va='bottom')
    # loop over traces
    for jjj in index:
        # time vector
        t_vec = np.arange(0, len(stream[jjj])) * np.timedelta64(int(stream[jjj].stats.delta * 1000), '[ms]') \
                + np.datetime64(str(stream[jjj].stats.starttime)[:-1])
        # plot waveform
        h1, = axis.plot(t_vec, stream[jjj].data / stream[jjj].max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # display station name & distance
        ht1 = axis.text(axis.get_xlim()[1] + (axis.get_xlim()[1] - axis.get_xlim()[0]) / 150., n_trace,
                        f"{stream[jjj].stats.distance / 1000.:.2f} km", ha='left', va='center', fontsize=5)
        ht2 = axis.text(axis.get_xlim()[0] - (axis.get_xlim()[1] - axis.get_xlim()[0]) / 150., n_trace,
                        f"{stream[jjj].stats.network}.{stream[jjj].stats.station}."
                        f"{stream[jjj].stats.location}.{stream[jjj].stats.channel}", ha='right', va='center', fontsize=5)
        # EPIC trigger
        sta_tab = trig_tab[trig_tab.trig_sta == stream[jjj].stats.station].reset_index(drop=True)
        if not sta_tab.empty:
            for iii in range(len(sta_tab)):
                h2, = axis.plot([sta_tab.trig_time[iii], sta_tab.trig_time[iii]], [n_trace - 1, n_trace + 1], color='red', alpha=.7, label='Trigger')
            ht1.set(color='red')
            ht2.set(color='red')
        # theoretical arrival
        if 'theo_tt' in stream[jjj].stats and not np.isnat(stream[jjj].stats.theo_tt):
            h3, = axis.plot([stream[jjj].stats.theo_tt, stream[jjj].stats.theo_tt], [n_trace - 1, n_trace + 1], ':b', alpha=.7, label=rmod)
        n_trace += 1
    # title
    axis.set_title(f"{event_row.evt_id}: M{event_row.evt_maxmag_mag:3.2f}"
                   f" {datetime.strftime(event_row.evt_time, '%d/%m/%Y %H:%M:%S')}", fontweight='bold')
    axis.set_xlabel('Time [s]', fontweight='bold')
    # axes
    axis.grid(which='both', axis='both')
    date_form = DateFormatter('%d/%m/%Y %H:%M:%S')
    axis.xaxis.set_major_formatter(date_form)
    # legend
    if not h2 and not h5:
        axis.legend(handles=[h4, h1], loc='lower left', fontsize=8)
    elif not h2 and h5:
        axis.legend(handles=[h4, h1, h3, h5], loc='lower left', fontsize=8)
    elif h2 and not h5:
        axis.legend(handles=[h4, h1, h2], loc='lower left', fontsize=8)
    else:
        axis.legend(handles=[h4, h1, h2, h3, h5], loc='lower left', fontsize=8)
    # replace y-axis tick labels with station names
    axis.set_yticks(np.arange(0, n_trace, 1))
    axis.set_yticklabels([])
    axis.set_ylim([-1, n_trace])
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show or save figure
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()


def plot_epic_check_summary(evt_tab, cat_tab, source, fig_name=None):
    """
    :param evt_tab: DataFrame containing all EPIC events
    :param cat_tab: DataFrame containing all catalogue events
    :param source: TRUA'A system being checked
    :param fig_name: full-path file name if figure to be saved (and not displayed)
    :return:
    """
    # prepare table
    tab_txt = []
    tab_col = []
    loc_err = pd.Series(dtype=float)
    for ii, event in evt_tab.iterrows():
        if event.evt_id != '' and event.gsi_id != '':
            # match
            dr = gdist.distance((event.evt_lat, event.evt_lon),
                                (cat_tab[cat_tab.evt_id == event.gsi_id].evt_lat.to_list()[0],
                                 cat_tab[cat_tab.evt_id == event.gsi_id].evt_lon.to_list()[0])).km
            dt = (cat_tab[cat_tab.evt_id == event.gsi_id].evt_time.to_list()[0] - event.evt_time).total_seconds()
            if cat_tab[cat_tab.evt_id == event.gsi_id].evt_type.to_list()[0] == 'TELESEISM':
                tab_txt.append([event.evt_id, f"{event.evt_maxmag_mag:.2f}", event.gsi_id,
                                f"{datetime.strftime(cat_tab[cat_tab.evt_id == event.gsi_id].evt_time.to_list()[0], '%d/%m/%Y %H:%M:%S')}",
                                f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_mag.to_list()[0]:.2f}".replace('nan',
                                                                                                              'NaN'),
                                f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_type.to_list()[0]}", "NaN", "NaN"])
                loc_err.loc[loc_err.shape[0]] = np.nan
                if event.evt_maxmag_mag > mmin:
                    tab_col.append(['blue', '.7', 'none', 'none', 'none', 'none', 'none', 'none'])
                else:
                    tab_col.append(['blue', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            else:
                tab_txt.append([event.evt_id, f"{event.evt_maxmag_mag:.2f}", event.gsi_id,
                                f"{datetime.strftime(cat_tab[cat_tab.evt_id == event.gsi_id].evt_time.to_list()[0], '%d/%m/%Y %H:%M:%S')}",
                                f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_mag.to_list()[0]:.2f}".replace('nan', 'NaN'),
                                f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_type.to_list()[0].replace('DISTANT_', '').replace('POSSIBLY_', '')}",
                                f"{dt:.2f} s", f"{dr:.2f} km"])
                loc_err.loc[loc_err.shape[0]] = dr
                if cat_tab[cat_tab.evt_id == event.gsi_id].evt_type.to_list()[0] == 'FELT':
                    if event.evt_maxmag_mag > mmin:
                        tab_col.append(
                            [(45/255, 201/255, 55/255), '.7', 'none', 'none', 'none', '.7', 'none', 'none'])
                    else:
                        tab_col.append(
                            [(45/255, 201/255, 55/255), 'none', 'none', 'none', 'none', '.7', 'none', 'none'])
                else:
                    if event.evt_maxmag_mag > mmin:
                        tab_col.append(
                            [(45/255, 201/255, 55/255), '.7', 'none', 'none', 'none', 'none', 'none', 'none'])
                    else:
                        tab_col.append(
                            [(45/255, 201/255, 55/255), 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
        elif event.evt_id != '' and event.gsi_id == '':
            # false
            tab_txt.append([event.evt_id, f"{event.evt_maxmag_mag:.2f}", '\u2013', f"{np.datetime64('NaT')}",
                            f"{np.nan:.2f}".replace('nan', 'NaN'),
                            '\u2013', f"{np.nan:.2f}".replace('nan', 'NaN'), f"{np.nan:.2f}".replace('nan', 'NaN')])
            if event.evt_maxmag_mag > mmin:
                tab_col.append(
                    [(204/255, 50/255, 50/255), '.7', 'none', 'none', 'none', 'none', 'none', 'none'])
            else:
                tab_col.append(
                    [(204/255, 50/255, 50/255), 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            loc_err.loc[loc_err.shape[0]] = np.nan
        elif event.evt_id == '' and event.gsi_id != '':
            # missed
            tab_txt.append(['\u2013', f"{np.nan:.2f}", event.gsi_id,
                            f"{datetime.strftime(cat_tab[cat_tab.evt_id == event.gsi_id].evt_time.to_list()[0], '%d/%m/%Y %H:%M:%S')}",
                            f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_mag.to_list()[0]:.2f}".replace('nan', 'NaN'),
                            f"{cat_tab[cat_tab.evt_id == event.gsi_id].evt_type.to_list()[0]}",
                            f"{np.nan:.2f}".replace('nan', 'NaN'), f"{np.nan:.2f}".replace('nan', 'NaN')])
            tab_col.append([(219/255, 123/255, 43/255), 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
            loc_err.loc[loc_err.shape[0]] = np.nan
    if fig_name:
        mpl.use('Agg')
    # create figure & axis
    ff, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(18, 9), dpi=200)
    if not fig_name:
        plt.show(block=False)
    # AXIS 1: SUMMARY MAP
    bmap = Basemap(projection='cyl', llcrnrlon=ngrd[2], llcrnrlat=ngrd[0], urcrnrlon=ngrd[3] + .5, urcrnrlat=ngrd[1],
                   ax=ax1, resolution='i')
    # draw map
    bmap.drawmapboundary(fill_color='none')
    bmap.fillcontinents(color='0.8', lake_color='white')
    # show parallels and meridians (labels = [left,right,top,bottom])
    bmap.drawparallels(np.arange(bmap.llcrnrlat, bmap.urcrnrlat + 1, 2.), labels=[True, False, True, False])
    bmap.drawmeridians(np.arange(bmap.llcrnrlon, bmap.urcrnrlon + 1, 2.), labels=[True, False, False, True])
    # fault lines (Sharon et al., 2020)
    f = open(f'/home/{os.getlogin()}/GoogleDrive/Research/GSI/Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz', 'r')
    flts = f.readlines()
    f.close()
    x = []
    y = []
    for ii in range(len(flts)):
        if re.search('NaN', flts[ii]):
            x = []
            y = []
        elif ii < len(flts) - 1 and re.search('NaN', flts[ii + 1]):
            ax1.plot(x, y, '.5', label='Faults')
        else:
            line = flts[ii].split()
            x.append(float(line[0]))
            y.append(float(line[1]))
    # event categories
    match = evt_tab.drop(
        evt_tab[(np.isnan(loc_err)) | ~((evt_tab.evt_id != '') & (evt_tab.gsi_id != ''))].index).reset_index(drop=True)
    false = evt_tab.drop(evt_tab[~((evt_tab.evt_id != '') & (evt_tab.gsi_id == ''))].index).reset_index(drop=True)
    missd = evt_tab.drop(evt_tab[~((evt_tab.evt_id == '') & (evt_tab.gsi_id != ''))].index).reset_index(drop=True)
    teles = evt_tab.drop(
        evt_tab[~(np.isnan(loc_err)) | ~((evt_tab.evt_id != '') & (evt_tab.gsi_id != ''))].index).reset_index(drop=True)
    # matches (lines)
    ax1.plot([[cat_tab.evt_lon[cat_tab.evt_id == evid].to_list()[0] for evid in
               evt_tab[~(np.isnan(loc_err)) & (evt_tab.gsi_id != '')].gsi_id], match.evt_lon.to_list()],
             [[cat_tab.evt_lat[cat_tab.evt_id == evid].to_list()[0] for evid in
               evt_tab[~(np.isnan(loc_err)) & (evt_tab.gsi_id != '')].gsi_id], match.evt_lat.to_list()], color='black', alpha=.7, linewidth=.5)
    # matches (EPIC)
    ax1.plot(match.evt_lon, match.evt_lat, 'o', color=(45/255, 201/255, 55/255), markeredgecolor='black', markersize=5, alpha=.7)
    # matches (CAT)
    ax1.plot([cat_tab[cat_tab.evt_id == evid].evt_lon for evid in evt_tab[evt_tab.gsi_id != ''].gsi_id],
             [cat_tab[cat_tab.evt_id == evid].evt_lat for evid in evt_tab[evt_tab.gsi_id != ''].gsi_id],
             'o', markerfacecolor='black', markeredgecolor='none', markersize=2, alpha=.7)
    # highlight FELT catalogue events
    ax1.plot(cat_tab[cat_tab.evt_type == 'FELT'].evt_lon, cat_tab[cat_tab.evt_type == 'FELT'].evt_lat,
             '*', color='red', markeredgecolor='black', markersize=10)
    # teleseisms
    ax1.plot(teles.evt_lon, teles.evt_lat, 'o', color='blue', markeredgecolor='black', markersize=5, alpha=.7)
    # false alerts
    ax1.plot(false.evt_lon, false.evt_lat, 'o', color=(204/255, 50/255, 50/255), markeredgecolor='black', markersize=10, alpha=.7)
    # missed events
    ax1.plot(missd.evt_lon, missd.evt_lat, 'o', color=(219/255, 123/255, 43/255), markeredgecolor='black', markersize=15, alpha=.7)
    # annotations
    ax1.text(ngrd[2] + .05, ngrd[0] + .70, f"Total: {len(evt_tab)}")
    ax1.text(ngrd[2] + .05, ngrd[0] + .55, f"Match: {len(match)}", color=(45/255, 201/255, 55/255))
    ax1.text(ngrd[2] + .05, ngrd[0] + .40, f"Missed [M>{mmin}]: {len(missd)}", color=(219/255, 123/255, 43/255))
    ax1.text(ngrd[2] + .05, ngrd[0] + .25, f"False: {len(false)}", color=(204/255, 50/255, 50/255))
    ax1.text(ngrd[2] + .05, ngrd[0] + .1, f"Teleseism: {len(teles)}", color='blue')
    # AXIS 2: SUMMARY TABLE
    ax2.axis('off')
    # show table
    ax2.table(cellText=tab_txt, loc='upper center', fontsize=25, cellLoc='center', cellColours=tab_col,
              colLabels=('EPIC ID', 'EPIC Mw', 'Catalogue ID', 'Origin Time', 'Mw', 'Type', 'OT Err.', 'XY Err.'),
              colWidths=[.08, .08, .16, .16, .08, .12, .1, .1])
    # title
    ax2.set_title(f"{datetime.strftime(tper, '%B %Y')}\n{source}", x=.16)
    # show or save figure
    ff.subplots_adjust(wspace=-.2)
    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()


# directories
src = 'eew-b-jer'
f_ext = src.split('-')[2].upper() + '-' + src.split('-')[1].upper()
wdir = f"/home/{os.getlogin()}/GoogleDrive/Research/GSI/EPIC/{f_ext}"
print(f"Working directory: {wdir}")
print()

# area of interest
rrad = 6371.
rgrd = [23., 43., 20., 50.]  # regional
lgrd = [29., 36., 32., 37.]  # local
ngrd = [29., 34.5, 33.5, 37.]  # national
rmod = 'iasp91'
model = TauPyModel(model=rmod)

# criteria to identify events
tlim = 30.  # [s]
rlim = tlim * 5.8  # [km]; Vp from PREM
twin = 10.  # [min]
# criteria for false alerts
ntri = 4  # [# triggers]
mmin = 3.5  # [MAG]

# FDSN databases
tele_client = Client('EMSC')
# read ISN station inventory
isn_inv = read_inventory(f'/home/{os.getlogin()}/GoogleDrive/Research/GSI/Autopicker/inventory.xml')
# selection area based on ISN distribution for missed events
sgrd = [min([st.latitude for st in isn_inv.networks[0].stations]) - km2deg(100.),
        max([st.latitude for st in isn_inv.networks[0].stations]) + km2deg(100.),
        min([st.longitude for st in isn_inv.networks[0].stations]) - km2deg(100.),
        max([st.longitude for st in isn_inv.networks[0].stations]) + km2deg(100.)]
# variables to loop over months
m1 = datetime.strptime('Apr 2022', '%b %Y')
m2 = datetime.strptime('May 2022', '%b %Y')
dm = (m2.year - m1.year) * 12 + m2.month - m1.month
for m in range(dm):
    # time period of interest
    tper = m1 + relativedelta(months=m)
    print(f"Processing: {datetime.strftime(tper, '%B %Y')}")
    # logging file
    if path.exists(f"{wdir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log") != 0:
        os.remove(f"{wdir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log")
    logger = my_custom_logger(f"{wdir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.log", level=logging.DEBUG)
    # list of days
    dtab = [tper + timedelta(days=idx) for idx in range((tper + relativedelta(months=+1) - tper).days)]

    #######################
    # LOADING EPIC EVENTS #
    logger.info('-----------------------------------------------------------------------------------------------------')
    etab = get_epic_events_db(src, tper, tper + relativedelta(months=+1))
    if etab.empty:
        logger.info(f'No EPIC alerts')
        continue
    logger.info(f'Loaded {len(etab)} EPIC alerts:')
    for _, evt in etab.iterrows():
        logger.info(f"[{evt.evt_id}] {datetime.strftime(evt.evt_time, '%d/%m/%Y %H:%M:%S')}: [{evt.evt_lat}, {evt.evt_lon}]"
                    f" M{evt.evt_maxmag_mag:4.2f} {type(evt.evt_alert_mag) is float}")
    logger.info('-----------------------------------------------------------------------------------------------------')

    ############################
    # LOADING CATALOGUE EVENTS #
    ctab = get_cat_events_db(tper, tper + relativedelta(months=+1))
    logger.info(f"Loaded {len(ctab)} catalogue events")
    logger.info('-----------------------------------------------------------------------------------------------------')

    #################################
    # LOOKING FOR CATALOGUE MATCHES #
    logger.info('Matching EPIC alerts with catalogue events:')
    # add column in EPIC table for event IDs of matching catalogue events
    etab = etab.assign(gsi_id=pd.Series([''] * len(etab), dtype='string'))
    etab = etab.assign(evt_type=pd.Series([''] * len(etab), dtype='string'))
    for i, evt in etab.iterrows():
        # if evt.evt_id != '1008':
        #     continue
        gsid = get_cat_event_match(ctab, evt, twin, tlim, rlim, logger)
        etab.loc[i, 'gsi_id'] = (gsid or '')
        if gsid:
            etab.loc[i, 'evt_type'] = ctab[ctab.evt_id == gsid].evt_type.to_list()[0]
        else:
            etab.loc[i, 'evt_type'] = 'UNKNOWN'
        ifplot = False
        fig = ''
        # check OT and LOC errors
        if gsid and abs(pd.to_datetime(evt.evt_time, utc=True) -
                        pd.to_datetime(ctab.loc[ctab.evt_id == gsid].evt_time, utc=True).iloc[0]).total_seconds() > 30.:
            logger.warning(f" Large OT or LOC error!")
            # plotting info.
            ifplot = True
            fig = f'{wdir}/checks/{evt.evt_id}.png'
        evt_lbl = ''
        # check false alerts with EMSC database
        if not gsid:
            if evt.evt_id == '930':
                gsid, desc = get_tele_event_match(tele_client, evt, ctab, twin * 2., tlim * 30. * 2., logger)
            else:
                gsid, desc = get_tele_event_match(tele_client, evt, ctab, twin * 2., tlim * 30., logger)
            etab.loc[i, 'gsi_id'] = (gsid or '')
            # event description for record section plot
            evt_lbl = desc
            # plotting info.
            ifplot = True
            fig = f'{wdir}/false/{evt.evt_id}.png'
        xevt = pd.DataFrame({})
        # build figure if non-existant
        if ifplot and fig and path.exists(fig) == 0:
            # get waveform data
            mfile = get_traces_deci(evt, twin)
            print(f'1: {mfile}')
            if mfile == '':
                logger.warning(' No decimated data, retrieving full resolution data')
                mfile = get_traces_full(evt, twin)
                print(f'2: {mfile}')
            # process waveform data
            nfile = process_mseed(mfile, isn_inv, twin)
            print(f'3: {nfile}')
            isn_traces = read(nfile)
            # add event info (theoretical arrivals & distance) to waveforms
            if 'check' in fig:
                isn_traces = add_event_data(isn_traces, ctab[ctab.evt_id == gsid], isn_inv)
            elif 'false' in fig:
                if gsid:
                    if evt.evt_id == '829':
                        xevt = pd.DataFrame({'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                                             'evt_dep': pd.Series(dtype='float64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                             'evt_type': pd.Series(dtype='string')})
                        xevt.loc[xevt.shape[0]] = [-6.78, 105.36, 40., np.datetime64('2022-01-14 09:05:43.5'), 'TELESEISM']
                        isn_traces = add_event_data(isn_traces, xevt, isn_inv)
                        evt_lbl = 'INDONESIA M6.6'
                    else:
                        isn_traces = add_event_data(isn_traces, ctab[ctab.evt_id == gsid], isn_inv)
                else:
                    if evt.evt_id == '435':
                        xevt = pd.DataFrame({'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                                             'evt_dep': pd.Series(dtype='float64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                             'evt_type': pd.Series(dtype='string')})
                        xevt.loc[xevt.shape[0]] = [-20.12, -177.7, 494., np.datetime64('2021-03-10 20:12:38.5'), 'TELESEISM']
                        evt_lbl = 'FIJI M5.9'
                        # xevt.loc[xevt.shape[0]] = [36.89, 28.88, 6., np.datetime64('2021-03-10 20:29:34.9'), 'REGIONAL']
                        # evt_lbl = 'DODECANESE M2.0'
                        # xevt.loc[xevt.shape[0]] = [34.2, 58.21, 10., np.datetime64('2021-03-10 20:30:17.5'), 'REGIONAL']
                        # evt_lbl = 'IRAN M4.7'
                        isn_traces = add_event_data(isn_traces, xevt, isn_inv)
                    elif evt.evt_id == '523':
                        xevt = pd.DataFrame({'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                                             'evt_dep': pd.Series(dtype='float64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                             'evt_type': pd.Series(dtype='string')})
                        xevt.loc[xevt.shape[0]] = [37.73, 141.75, 41., np.datetime64('2021-05-13 23:58:14.8'), 'TELESEISM']
                        isn_traces = add_event_data(isn_traces, xevt, isn_inv)
                        evt_lbl = 'JAPAN M6.0'
                    elif evt.evt_id == '930':
                        xevt = pd.DataFrame({'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                                             'evt_dep': pd.Series(dtype='float64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                             'evt_type': pd.Series(dtype='string')})
                        xevt.loc[xevt.shape[0]] = [-20.38, -178.36, 566., np.datetime64('2022-03-07 05:34:17.1'), 'TELESEISM']
                        isn_traces = add_event_data(isn_traces, xevt, isn_inv)
                        evt_lbl = 'FIJI M6.1'
                    elif evt.evt_id == '949':
                        xevt = pd.DataFrame({'evt_lat': pd.Series(dtype='float64'), 'evt_lon': pd.Series(dtype='float64'),
                                             'evt_dep': pd.Series(dtype='float64'), 'evt_time': pd.Series(dtype='datetime64[ns]'),
                                             'evt_type': pd.Series(dtype='string')})
                        xevt.loc[xevt.shape[0]] = [37.73, 141.62, 49., np.datetime64('2022-03-16 14:36:32.4'), 'TELESEISM']
                        isn_traces = add_event_data(isn_traces, xevt, isn_inv)
                        evt_lbl = 'JAPAN M7.3'
                    else:
                        isn_traces = add_event_data(isn_traces, etab[etab.evt_id == evt.evt_id], isn_inv)
            else:
                isn_traces = add_event_data(isn_traces, etab[etab.evt_id == evt.evt_id], isn_inv)
            # get all triggers from database for month of interest
            ttab = get_epic_triggers_db(src, evt.evt_id, evt.evt_ver)
            if ttab.empty:
                print(evt.evt_id)
            # indexes sorted according to distance (descending)
            sorted_ind = np.argsort([tr.stats.distance / 1000. for tr in isn_traces])[::-1][:len(isn_traces)]
            # plot record section
            if ctab[ctab.evt_id == gsid].empty:
                evt_row = xevt
            else:
                evt_row = ctab[ctab.evt_id == gsid]
            plot_event_rec_sec(isn_traces, sorted_ind, evt, evt_row, ttab, fig, evt_lbl)
            logger.info(f" Record section saved: {fig}")
        else:
            logger.info(f" Record section existing: {fig}")
        # check other catalogues if needed
    logger.info('-----------------------------------------------------------------------------------------------------')

    #######################################
    # LOOKING FOR MISSED CATALOGUE EVENTS #
    # ignore events far away from ISN and low-magnitude events
    missed = ctab[(sgrd[0] > ctab.evt_lat) & (ctab.evt_lat < sgrd[1]) &
                  (sgrd[2] > ctab.evt_lon) & (ctab.evt_lon < sgrd[3]) & (ctab.evt_mag > mmin)].reset_index(drop=True)
    if not missed.empty:
        logger.info('Checking missed detections:')
        for _, evt in missed.iterrows():
            # check if event is false alert
            if etab[etab.gsi_id == evt.evt_id].empty:
                logger.info(f"[{evt.evt_type}] Missed: {datetime.strftime(evt.evt_time, '%d/%m/%Y %H:%M:%S')} M{evt.evt_mag:3.1f}")
        logger.info('-------------------------------------------------------------------------------------------------')

    ########################
    # BUILD SUMMARY FIGURE #
    plot_epic_check_summary(etab, ctab, src, f"{wdir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.png")
    logger.info(f"Monthly summary saved: {wdir}/epic_match_{datetime.strftime(tper, '%Y-%m')}.png")
    logger.info('-----------------------------------------------------------------------------------------------------')
exit()

# ###########################
# # EPIC/GSI EVENT MATCHING #
# to_check = 941
# fil3 = f"{wdir}/epic_match_{datetime.strftime(datetime.strptime(tper, '%b %Y'), '%Y_%b')}.csv"
# if to_check or path.exists(fil3) == 0:
#     # EPIC event classification tables
#     evt_detec = pd.DataFrame({'epic_id': pd.Series(dtype='int'),
#                               'gsi_id': pd.Series(dtype='string')})      # event detection (EPIC event matched in GSI catalogue)
#     evt_false = pd.DataFrame({'epic_id': pd.Series(dtype='int')})        # false detection (EPIC event missing in GSI catalogue)
#     evt_missd = pd.DataFrame({'gsi_id': pd.Series(dtype='string')})      # missed detection (GSI event missed by EPIC)
#     # loop over EPIC event IDs
#     for _, row in etab.loc[ievt].iterrows():
#         if to_check and row.evt_id != to_check:
#             continue
#         # look for indexes of GSI events within time window
#         isel = ctab.index[(ctab.evt_ot > row.evt_ot - timedelta(minutes=twin)) &
#                           (ctab.evt_ot <= row.evt_ot + timedelta(minutes=twin))].to_list()
#         if to_check:
#             row['evt_mag'] = etab[etab.evt_id == row.evt_id].evt_mag.max()
#             print(row)
#         if len(isel) < 1:
#             print(f"[{row.evt_id}] {datetime.strftime(row.evt_ot, '%d/%m/%Y %H:%M:%S')}: False (no events within time window)")
#             evt_false.loc[evt_false.shape[0]] = [row.evt_id]
#             continue
#         # list GSI events within time window
#         itab = ctab.loc[isel].reset_index(drop=True)
#         # calculate OT error between all GSI events in window and EPIC event
#         ot_dif = [abs(x - row.evt_ot).total_seconds() for x in itab.evt_ot]
#         ot_test = [True if x < tlim else False for x in ot_dif]
#         # calculate LOC error between all GSI events in window and EPIC event
#         loc_dif = [gdist.distance((row.evt_lat, row.evt_lon), (itab.evt_lat[j], itab.evt_lon[j])).km for j in range(len(itab))]
#         loc_test = [True if x < rlim else False for x in loc_dif]
#         # combine both criteria
#         evt_test = [a and b for a, b in zip(ot_test, loc_test)]
#         if sum(evt_test) == 0:
#             print(f"[{row.evt_id}] {datetime.strftime(row.evt_ot, '%d/%m/%Y %H:%M:%S')}: False (no OT/LOC match)")
#             if to_check:
#                 print(f"   {ot_dif} (< {tlim} s)")
#                 print(f"   {loc_dif} (< {rlim} km)")
#                 print(f"   {itab.evt_id.to_list()}")
#             evt_false.loc[evt_false.shape[0]] = [row.evt_id]
#             continue
#         # find indexes that verify both criteria
#         kk = [k for k, x in enumerate(evt_test) if x]
#         if len(kk) > 1:
#             kk = ot_dif.index(min(ot_dif))
#         else:
#             kk = kk[0]
#         # check if match with already-matched event
#         if not evt_detec.empty and not evt_detec.loc[evt_detec.gsi_id == itab.evt_id[kk]].empty:
#             print(f" WARNING // catalogue event already matched: {itab.evt_id[kk]}")
#         print(f"[{row.evt_id}] {datetime.strftime(row.evt_ot, '%d/%m/%Y %H:%M:%S')}: "
#               f"Match with {datetime.strftime(itab.evt_ot[kk], '%d/%m/%Y %H:%M:%S')} [{itab.evt_id[kk]}]")
#         evt_detec.loc[evt_detec.shape[0]] = [row.evt_id, itab.evt_id[kk]]
#     if to_check:
#         exit()
#     # loop over catalogue events to list missed ones
#     for _, row in ctab.iterrows():
#         # check if event already listed as detected by EPIC
#         if evt_detec[evt_detec.gsi_id == row.evt_id].empty:
#             # print('[{row.evt_id}] {row.evt_ot.strftime('%d/%m/%Y %H:%M:%S')}}: Missed (M{row.evt_mag:3.2f} {row.evt_type})'
#             evt_missd.loc[evt_missd.shape[0]] = [row.evt_id]
#     # summary
#     print()
#     print(f"{len(evt_detec)} correct detections (EPIC events matched in GSI catalogue)")
#     print(f"{len(evt_false)} false detections (EPIC events missing in GSI catalogue)")
#     print(f"{len(evt_missd)} missed detections (GSI events missed by EPIC)")
#     print()
#     # add columns to EPIC and GSI event tables with event-matching info
#     etab = etab.assign(detec_type=pd.Series([] * len(etab), dtype='int64'), gsi_id=pd.Series([''] * len(etab), dtype='string'))
#     ctab = ctab.assign(detec_type=pd.Series([] * len(ctab), dtype='int64'), epic_id=pd.Series([0] * len(ctab), dtype='int64'))
#     # loop over correct detections
#     for i in range(len(evt_detec)):
#         ie = etab.index[etab.evt_id == evt_detec.epic_id[i]].to_list()
#         ic = ctab.index[ctab.evt_id == evt_detec.gsi_id[i]].to_list()
#         etab.loc[ie, 'detec_type'] = 2
#         etab.loc[ie, 'gsi_id'] = evt_detec.gsi_id[i]
#         ctab.loc[ic, 'detec_type'] = 2
#         ctab.loc[ic, 'epic_id'] = evt_detec.epic_id[i]
#     # loop over false detections
#     for i in range(len(evt_false)):
#         ie = etab.index[etab.evt_id == evt_false.epic_id[i]].to_list()
#         etab.loc[ie, 'detec_type'] = 1
#     # loop over missed detections
#     for i in range(len(evt_missd)):
#         ic = ctab.index[ctab.evt_id == evt_missd.gsi_id[i]].to_list()
#         ctab.loc[ic, 'detec_type'] = 0
#     # write corresponding file
#     etab.to_csv(fil3, index=False, float_format='%.4f')
#     ctab.to_csv(fil3.replace('epic', 'gsi'), index=False, float_format='%.4f')
# else:
#     print('Event matching was already ran for that period:')
#     print(' ' + fil3)
#     print()
#     etab = pd.read_csv(fil3, parse_dates=['evt_ot', 'alrt_t'])
#     ctab = pd.read_csv(fil3.replace('epic', 'gsi'), parse_dates=['evt_ot'])
#
# ##########################
# # CHECK FALSE DETECTIONS #
# print('Checking false detections:')
# for ii, row in etab.loc[ievt].iterrows():
#     # if row.evt_id != 958:
#     #     continue
#     # check if event is false alert
#     if row.detec_type == 1:
#         n_ver = len(etab[etab.evt_id == row.evt_id])
#         m_max = etab[etab.evt_id == row.evt_id].evt_mag.max()
#         if n_ver > 1:
#             # search which version to work with
#             if sum(etab.alrt_sent[etab.evt_id == row.evt_id]) and not row.alrt_sent:
#                 # if alert sent, work with this version
#                 ii = etab.index[(etab.evt_id == row.evt_id) & etab.alrt_sent].to_list()[0]
#             else:
#                 # if no alert, work with version with largest magnitude
#                 ii = etab.loc[(etab.evt_id == row.evt_id) & (etab.evt_mag == m_max)]\
#                     .drop_duplicates(subset=['evt_mag'], keep='first').index.to_list()[0]
#         # regardless of version choice, always use max. magnitude
#         row.loc['evt_mag'] = m_max
#         print(f" [{row.evt_id}] False: {datetime.strftime(row.evt_ot, '%d/%m/%Y %H:%M:%S')} M{row.evt_mag:3.2f}")
#         if path.exists(f"{wdir}/false/{row.evt_id}.png") != 0:
#             print('   Event already inspected')
#             continue
#         # extract triggers
#         trig = ttab[(ttab.evt_id == row.evt_id) & (ttab.ori_ver == row.ori_ver)].reset_index(drop=True)
#         if row.alrt_sent:
#             print('   !! ALERT SENT !!')
#         else:
#             # not a false alert if M < [mmin]
#             if row.evt_mag < mmin:
#                 print(f'   Alert ignored because M < {mmin}')
#                 continue
#             # not a false alert if < [ntri] triggers
#             if len(trig) < ntri:
#                 print(f'   Alert ignored because < {ntri} triggers')
#                 continue
#         ################################################################################################################
#         # IF ONLINE
#         try:
#             cat = tele_client.get_events(starttime=UTCDateTime(row.evt_ot - timedelta(minutes=twin * 2)),
#                                          endtime=UTCDateTime(row.evt_ot + timedelta(minutes=twin*2)), minmagnitude=5, includearrivals=False)
#         except:
#             print('   No M>5 teleseisms')
#             try:
#                 cat = tele_client.get_events(starttime=UTCDateTime(row.evt_ot - timedelta(minutes=twin * 2)),
#                                              endtime=UTCDateTime(row.evt_ot + timedelta(minutes=twin*2)), minmagnitude=4, includearrivals=False)
#             except:
#                 print('   No M>4 teleseisms')
#                 cat = []
#         # initialise EMSC events table
#         xtab = pd.DataFrame({'OriginTime': pd.Series(dtype='datetime64[ns]'), 'Latitude': pd.Series(dtype='float64'),
#                              'Longitude': pd.Series(dtype='float64'), 'Depth': pd.Series(dtype='float64'),
#                              'Magnitude': pd.Series(dtype='float64'), 'RegionName': pd.Series(dtype='string')})
#         ################################################################################################################
#         # # IF OFFLINE
#         # # read event info
#         # xtab = pd.read_csv(f"{wdir.replace(f_ext, '')}/{tper.strftime('%b_%Y_EMSC_M5.csv')}", parse_dates=['OriginTime'])
#         # cat = xtab[(xtab.OriginTime > row.evt_ot - timedelta(minutes=twin * 2.)) &
#         #            (xtab.OriginTime < row.evt_ot + timedelta(minutes=twin * 2.))].reset_index(drop=True)
#         ################################################################################################################
#         # initialise event array
#         otab = pd.DataFrame({'evt_ot': pd.Series(dtype='datetime64[ns]'), 'evt_lat': pd.Series(dtype='float64'),
#                              'evt_lon': pd.Series(dtype='float64'), 'evt_dep': pd.Series(dtype='float64'), 'label': pd.Series(dtype='string')})
#         if cat:  # and not cat.empty:
#             ################################################################################################################
#             # IF ONLINE
#             for k in range(len(cat)):
#                 xtab.loc[xtab.shape[0]] = [np.datetime64(cat[k].origins[0].time), cat[k].origins[0].latitude, cat[k].origins[0].longitude,
#                                            cat[k].origins[0].depth/1000., cat[k].magnitudes[0].mag, cat[0].event_descriptions[0].text]
#             # check if matches with event list
#             ot_dif = [abs(x - row.evt_ot).total_seconds() for x in xtab.OriginTime]
#             ot_test = [True if x < tlim*10. else False for x in ot_dif]
#             jj = ot_dif.index(min(ot_dif))
#             if xtab.OriginTime[jj] > row.evt_ot:
#                 jj = None
#             if jj is not None:
#                 print(f"   Event match: {datetime.strftime(xtab.OriginTime[jj], '%d/%m/%Y %H:%M:%S')}"
#                       f" M{xtab.Magnitude[jj]:3.2f} ({xtab.RegionName[jj]})")
#                 for j in range(len(xtab)):
#                     otab.loc[otab.shape[0]] = [np.datetime64(xtab.OriginTime[j]), xtab.Latitude[j], xtab.Longitude[j], xtab.Depth[j],
#                                                f'm{xtab.Magnitude[j]:3.1f} ({xtab.RegionName[j]})']
#         # elif not cat and row.evt_id == 958:
#         #     otab.loc[otab.shape[0]] = [np.datetime64('2022-03-21 11:02:11.500'), 35.18, 23.17, 17., 'EMSC']
#         else:
#             print('   No match found')
#         # load ISN traces
#         if path.exists(f'{wdir}/false/{row.evt_id}.mseed') == 0:
#             tic = timeit.default_timer()
#             isn_traces = isn_client.get_waveforms('IS,GE', '*', '*', 'HHZ,ENZ',
#                                                   starttime=UTCDateTime(row.evt_ot-timedelta(minutes=twin)),
#                                                   endtime=UTCDateTime(row.evt_ot+timedelta(minutes=twin))).merge()
#             toc = timeit.default_timer()
#             print(f'   Loading waveforms took {toc-tic:.2f} s')
#             # remove response from all traces
#             isn_traces.remove_response(output='VEL', inventory=isn_inv)
#             # apply taper to all traces
#             isn_traces.taper(max_percentage=.5, type='cosine', max_length=10., side='left')
#             # remove trend from all traces
#             isn_traces.detrend('spline', order=3, dspline=500)
#             # write miniSEED file
#             isn_traces.write(f'{wdir}/false/{row.evt_id}.mseed')
#         else:
#             # read miniSEED file
#             isn_traces = read(f'{wdir}/false/{row.evt_id}.mseed').merge()
#         # apply channel priorities (HHZ>ENZ) and remove partial traces
#         for tr in isn_traces:
#             # if HHZ channel, remove all others
#             if tr.stats.channel == 'HHZ':
#                 x = isn_traces.select(station=tr.stats.station, network=tr.stats.network, channel='ENZ')
#                 if x:
#                     isn_traces.remove(x[0])
#             # remove incomplete traces
#             if tr.stats.npts != int(twin*60./tr.stats.delta)+1:
#                 y = isn_traces.select(station=tr.stats.station, network=tr.stats.network, channel=tr.stats.channel, location=tr.stats.location)
#                 if y:
#                     isn_traces.remove(tr)
#         # remove ENZ channel if HHZ exist for given station, calculate distance and theoretical arrival times (
#         for tr in isn_traces:
#             # find station of interest
#             st = stab[stab.sta == tr.stats.station]
#             if not otab.empty:
#                 # find index of event closest to EPIC event
#                 ot_diff = [abs(row.evt_ot.timestamp() - date.timestamp()) for date in otab.evt_ot]
#                 ie = ot_diff.index(min(ot_diff))
#                 # compute distance from EMSC event
#                 dis = gdist.distance((otab.evt_lat[ie], otab.evt_lon[ie]), (st.lat.to_list()[0], st.lon.to_list()[0]))
#                 tr.stats.distance = dis.m
#                 # compute theoretical travel times for EPIC event
#                 x = model.get_travel_times(source_depth_in_km=otab.evt_dep[ie], distance_in_degree=dis.km / (2 * np.pi * rrad / 360),
#                                            phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
#                 if len(x) != 0:
#                     tr.stats['theo_tt'] = np.datetime64(otab.evt_ot[ie] + timedelta(seconds=x[0].time))
#                 else:
#                     tr.stats['theo_tt'] = np.datetime64('NaT')
#             else:
#                 tr.stats['theo_tt'] = np.datetime64('NaT')
#                 # calculate distance
#                 dis = gdist.distance((row.evt_lat, row.evt_lon), (st.lat.to_list()[0], st.lon.to_list()[0]))
#                 tr.stats.distance = dis.m
#         # indexes to sort according to distance (descending)
#         sorted_ind = np.argsort([tr.stats.distance/1000. for tr in isn_traces])[::-1][:len(isn_traces)]
#         # event list appended with EPIC alert
#         if not otab.empty:
#             list_ot = otab.evt_ot.to_list()
#             list_lbl = otab.label.to_list()
#             if len(list_ot) > 1:
#                 # limit to closest event
#                 tdif = [abs((ot-row.evt_ot).total_seconds()) for ot in list_ot]
#                 imin = tdif.index(min(tdif))
#                 list_ot = [list_ot[imin]]
#                 list_lbl = [list_lbl[imin]]
#         else:
#             list_ot = []
#             list_lbl = []
#         list_ot.append(np.datetime64(row.evt_ot))
#         list_lbl.append(f'EPIC [{row.evt_id}]: M{row.evt_mag:3.2f}')
#         # plot record section
#         mpl.rcParams['savefig.directory'] = f'{wdir}/false'
#         print('   Plotting record section...')
#         event_rec_sec(isn_traces, sorted_ind, list_ot, list_lbl, trig, row.alrt_sent, f'{wdir}/false/{row.evt_id}.png')

# def read_epic_log(idate, ifall=None):
#     # ________________________________________________________ #
#     # idate: date (datetime format) for daily log file to read
#     # ifall: boolean to include negative ID events or not
#     # ________________________________________________________ #
#     # ending of search pattern depends on requested events
#     if ifall:
#         ending = '*'
#     else:
#         ending = '*[0-9]'
#     # log file name
#     ifile = '%s/E2_%s.log' % (ldir, datetime.strftime(idate, '%Y%m%d'))
#     # check file exists
#     if path.exists(ifile) == 0:
#         print(' File does not exist in %s' % ldir)
#         return [], [], 0
#     # read log file
#     fid = open(ifile, 'r')
#     f_lines = fid.readlines()
#     fid.close()
#     # check how many origins in log file
#     patt = re.compile('\\| INFO \\| E:I:(F:|\\s) ' + ending)
#     n_ori = 0
#     for jjj, line in enumerate(open(ifile)):
#         for _ in re.finditer(patt, line):
#             n_ori += 1
#     if n_ori == 0:
#         # print(' No origin was found in: %s/E2_%s.log' % (ldir, datetime.strftime(idate, '%Y%m%d')))
#         return [], [], 0
#     # initialise data tables
#     evt_tab = pd.DataFrame(columns=['evt_id', 'ori_ver', 'evt_lat', 'evt_lon', 'evt_dep', 'evt_mag',
#                                     'evt_ot', 'alrt_t', 'alrt_sent', 'n_trig', 'n_sta'])
#     trig_tab = pd.DataFrame(columns=['evt_id', 'ori_ver', 'trig_ver', 'trig_ord', 'trig_sta', 'trig_net', 'trig_chn',
#                                      'trig_loc', 'trig_lat', 'trig_lon', 'trig_time', 'trig_dis', 'trig_azi'])
#     # loop over lines
#     for line in f_lines:
#         alrt = 0
#         # search event entries
#         if re.search('\\| INFO \\| E:I:\\s ' + ending, line) or re.search('\\| INFO \\| E:I:F:\\s ' + ending, line):
#             # check if alert sent
#             if re.search('\\| INFO \\| E:I:F:\\s ' + ending, line):
#                 alrt = 1
#             # fill EPIC event table
#             evt_tab.loc[evt_tab.shape[0]] = [line.split()[4], line.split()[5], line.split()[6], line.split()[7],
#                                              line.split()[8], line.split()[9], line.split()[10], line.split()[-1],
#                                              alrt, line.split()[19], line.split()[20]]
#         # search trigger entries
#         if re.search('\\| INFO \\| E:I:T:\\s ' + ending, line):
#             # fill EPIC trigger table
#             trig_tab.loc[trig_tab.shape[0]] = [line.split()[4], line.split()[5], line.split()[6], line.split()[7],
#                                                line.split()[8], line.split()[10], line.split()[9], line.split()[11],
#                                                line.split()[12], line.split()[13], line.split()[14], line.split()[35], line.split()[36]]
#     # output variables
#     return evt_tab, trig_tab, n_ori

# # query all triggers from database for month of interest
# log = TRIGGER.objects.filter(source__contains=src,
#                              trigger_time__range=[str(UTCDateTime(tper)),
#                                                   str(UTCDateTime(tper+relativedelta(months=+1)))]).order_by('eventid', 'ver', 'order')
# # loop over triggers
# for line in log:
#     line = str(line)
#     ttab.loc[ttab.shape[0]] = [line.split()[1], line.split()[2], line.split()[3], line.split()[4],
#                                line.split()[5], line.split()[7], line.split()[6], line.split()[8],
#                                line.split()[9], line.split()[10], (line.split()[11] + ' ' + line.split()[12]).replace('+00:00', ''),
#                                line.split()[33], line.split()[34]]

# # catalogue arrival parameters
# for j in range(len(evt_lst[i].origins[0].arrivals)):
#     for k in range(len(evt_lst[i].picks)):
#         # match pick & arrival
#         if evt_lst[i].origins[0].arrivals[j].pick_id == evt_lst[i].picks[k].resource_id and evt_lst[i].origins[0].arrivals[j].phase == 'P':
#             # identifying station
#             kk = stab.index[(stab.sta == evt_lst[i].picks[k].waveform_id.station_code) &
#                             (stab.net == evt_lst[i].picks[k].waveform_id.network_code)]
#             if len(kk) > 1:
#                 print(f'WARNING // several station matches: {evt_lst[i].picks[k].waveform_id.station_code}')
#             elif len(kk) == 0:
#                 print(f'WARNING // no station match: {evt_lst[i].picks[k].waveform_id.station_code}')
#             else:
#                 kk = kk[0]
#             slat = stab.iloc[kk].lat
#             slon = stab.iloc[kk].lon
#             sele = stab.iloc[kk].ele
#             # fill GSI arrivals table
#             atab.loc[atab.shape[0]] = [str(evt_lst[i].origins[0].arrivals[j].resource_id).split('#')[1],
#                                        evt_lst[i].picks[k].waveform_id.station_code, evt_lst[i].picks[k].waveform_id.network_code,
#                                        slat, slon, sele, evt_lst[i].picks[k].waveform_id.location_code,
#                                        evt_lst[i].picks[k].waveform_id.channel_code,
#                                        datetime.strptime(str(evt_lst[i].picks[k].time), '%Y-%m-%dT%H:%M:%S.%fZ'),
#                                        evt_lst[i].origins[0].arrivals[j].azimuth,
#                                        deg2km(evt_lst[i].origins[0].arrivals[j].distance), evt_lst[i].origins[0].arrivals[j].time_residual]
#             break

# ########
# # PLOT #
# # histogram variables
# E = etab[etab.detec_type == 2].drop_duplicates(subset=['evt_id'], keep='last').reset_index(drop=True)
# C = ctab[ctab.detec_type == 2].reset_index(drop=True)
# # tables for matched events are the same
# if len(E) != len(C) or not E.gsi_id.to_list() == C.evt_id.to_list():
#     print(' WARNING // event IDs do not match')
# dr = []         # location error
# dm = []         # magnitude error
# dt = []         # alert time w.r.t. OT
# for i in range(len(E)):
#     dr.append(gdist.distance((E.evt_lat.loc[i], E.evt_lon.loc[i]), (C.evt_lat.loc[i], C.evt_lon.loc[i])).km)
#     if C.evt_mag[i] != 0.:
#         dm.append(E.evt_mag[i]-C.evt_mag[i])
#     else:
#         dm.append(np.nan)
#     dt.append((E.alrt_t[i]-C.evt_ot[i]).total_seconds())
# ##########
# # FIGURE #
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, squeeze=True)
# plt.show(block=False)
# # AXIS 1: EVENT ALERT MAP
# # geographic area
# grd = ngrd
# # draw map
# M = Basemap(projection='cyl', llcrnrlon=grd[2], llcrnrlat=grd[0], urcrnrlon=grd[3], urcrnrlat=grd[1], ax=ax1)  # , resolution='i'
# M.drawmapboundary(fill_color='none')
# M.fillcontinents(color='0.8', lake_color='white')
# # parallels and meridians (labels = [left,right,top,bottom])
# M.drawparallels(np.arange(M.llcrnrlat, M.urcrnrlat + 1, 2.), labels=[True, False, True, False])
# M.drawmeridians(np.arange(M.llcrnrlon, M.urcrnrlon + 1, 2.), labels=[True, False, False, True])
# # fault lines (Sharon et al., 2020)
# f = open('/mnt/c/Users/lewiss/Documents/Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz', 'r')
# flts = f.readlines()
# f.close()
# X = []
# Y = []
# for i in range(len(flts)):
#     if re.search('NaN', flts[i]):
#         X = []
#         Y = []
#     elif i < len(flts) - 1 and re.search('NaN', flts[i + 1]):
#         ax1.plot(X, Y, '.6', label='Faults')
#     else:
#         line = flts[i].split()
#         X.append(float(line[0]))
#         Y.append(float(line[1]))
# # table selection for each category
# p1 = etab.drop(etab[etab.detec_type != 2].index).drop_duplicates(subset=['evt_id'], keep='last').reset_index(drop=True)     # correct detections (EPIC)
# p2 = ctab.drop(ctab[ctab.detec_type != 2].index).reset_index(drop=True)                                                     # correct detections (GSI)
# p3 = etab.drop(etab[(etab.detec_type != 1) | (etab.evt_mag < mmin) | (etab.evt_lat < sgrd[0]) |
#                     (etab.evt_lat > sgrd[1]) | (etab.evt_lon < sgrd[2]) | (etab.evt_lon > sgrd[3])
#                     ].index).drop_duplicates(subset=['evt_id'], keep='last').reset_index(drop=True)                         # false detections (EPIC)
# p4 = ctab.drop(ctab[(ctab.detec_type != 0) | (ctab.evt_type == 'EXPLOSION') | (ctab.evt_mag < mmin) |
#                     (ctab.evt_lat < sgrd[0]) | (ctab.evt_lat > sgrd[1]) | (ctab.evt_lon < sgrd[2]) |
#                     (ctab.evt_lon > sgrd[3])].index).reset_index(drop=True)                                                 # missed detections (GSI)
# p5 = ctab.drop(ctab[(ctab.detec_type != 2) | (ctab.evt_type != 'EXPLOSION')].index).reset_index(drop=True)                  # explosions (GSI)
# p6 = etab.drop(etab[etab.alrt_sent == False].index).drop_duplicates(subset=['evt_id'], keep='last').reset_index(drop=True)  # alert sent (EPIC)
# p7 = ctab.drop(ctab[(ctab.detec_type != 2) | (ctab.evt_type != 'FELT') | (ctab.evt_lat < sgrd[0]) |
#                     (ctab.evt_lat > sgrd[1]) | (ctab.evt_lon < sgrd[2]) |
#                     (ctab.evt_lon > sgrd[3])].index).reset_index(drop=True)   # felt events (GSI)
# # EPIC/GSI location difference
# ax1.plot([p2.evt_lon, p1.evt_lon], [p2.evt_lat, p1.evt_lat], '-', color='green', linewidth=.5, zorder=10)
# # connect alerts to final location if shown
# i1 = p1.index[p1['evt_id'].isin(p6['evt_id'])]
# i6 = p6.index[p6['evt_id'].isin(p1['evt_id'])]
# ax1.plot([p6.evt_lon.iloc[i6], p1.evt_lon.iloc[i1]], [p6.evt_lat.iloc[i6], p1.evt_lat.iloc[i1]], '-', color='magenta', zorder=10, linewidth=.5)
# # correct detections
# h1 = ax1.scatter(p1.evt_lon, p1.evt_lat, s=49, c=dt, cmap='YlGn',
#                  vmin=0., vmax=30., alpha=.7, zorder=10, marker='o', edgecolors='green', label='Correct detections (N=%i)' % len(p1))
# # false detections (with magnitude)
# h3 = ax1.scatter(p3.evt_lon, p3.evt_lat, s=25, c='orange', alpha=.7, zorder=6, marker='o', label='False detections (N=%i)' % len(p3))
# dxy = .06   # text offset
# for i, txt in enumerate(p3.evt_mag.astype(str).to_list()):
#     if i % 2 == 0:
#         ax1.annotate(txt, xy=(p3.evt_lon[i], p3.evt_lat[i]), xytext=(p3.evt_lon[i]+dxy, p3.evt_lat[i]+dxy),
#                      ha='center', va='center', fontsize=6, color='orange')
#     else:
#         ax1.annotate(txt, xy=(p3.evt_lon[i], p3.evt_lat[i]), xytext=(p3.evt_lon[i]-dxy, p3.evt_lat[i]-dxy),
#                      ha='center', va='center', fontsize=6, color='orange')
# # missed detections (with magnitude)
# h4 = ax1.scatter(p4.evt_lon, p4.evt_lat, s=25, c='red', alpha=.7, zorder=4, marker='o', label='Missed detections (N=%i)' % len(p4))
# for i, txt in enumerate(p4.evt_mag.astype(str).to_list()):
#     if i % 2 == 0:
#         ax1.annotate(txt, xy=(p4.evt_lon[i], p4.evt_lat[i]), xytext=(p4.evt_lon[i]+dxy, p4.evt_lat[i]+dxy),
#                      ha='center', va='center', fontsize=6, color='red')
#     else:
#         ax1.annotate(txt, xy=(p4.evt_lon[i], p4.evt_lat[i]), xytext=(p4.evt_lon[i]-dxy, p4.evt_lat[i]-dxy),
#                      ha='center', va='center', fontsize=6, color='red')
# # explosions (flag)
# h5 = ax1.scatter(p5.evt_lon, p5.evt_lat, s=16, c='none', alpha=.7, zorder=10, marker='s', edgecolors='blue', label='Explosions (N=%i)' % len(p5))
# # alerts sent (flag; with magnitude)
# h6 = ax1.scatter(p6.evt_lon, p6.evt_lat, s=16, c='none',
#                  zorder=10, marker='^', edgecolors='magenta', linewidth=.5, label='Alerts sent (N=%i)' % len(p6))
# for i, txt in enumerate(p6.evt_mag.astype(str).to_list()):
#     if i % 2 == 0:
#         ax1.annotate(txt, xy=(p6.evt_lon[i], p6.evt_lat[i]), xytext=(p6.evt_lon[i]-dxy, p6.evt_lat[i]-dxy),
#                      ha='center', va='center', fontsize=6, color='magenta')
#     else:
#         ax1.annotate(txt, xy=(p6.evt_lon[i], p6.evt_lat[i]), xytext=(p6.evt_lon[i]+dxy, p6.evt_lat[i]+dxy),
#                      ha='center', va='center', fontsize=6, color='magenta')
# # felt events
# h7 = ax1.scatter(p7.evt_lon, p7.evt_lat, s=100, c='red', zorder=10, marker='*', edgecolors='black', linewidth=.5, label='Felt events (N=%i)' % len(p7))
# # colour bar
# divider = make_axes_locatable(ax1)
# cax = divider.new_vertical(size='2%', pad=.4, pack_start=True)
# fig.add_axes(cax)
# c = plt.colorbar(h1, cax=cax, orientation='horizontal')
# c.ax.set_xlabel('Time after OT [s]')
# # legend
# ax1.legend(handles=[h1, h3, h4, h5, h6, h7], loc='upper left')
# # labels
# ax1.set_title('Alert time', fontsize=15, fontweight='bold')
# # AXIS 2: MAGNITUDE DIFFERENCE
# if dm and sum(np.isnan(dm)) != len(dm):
#     # histogram
#     ax2.hist(dm, bins=np.linspace(start=0, stop=mmax, num=11, endpoint=True), edgecolor='white', zorder=2)
#     # statistics
#     ax2.text(.97*ax2.get_xlim()[1], .98*ax2.get_ylim()[1], '{:.2f} \u00B1 {:.2f}\nmax = {:.2f}'.
#              format(np.nanmean(dm), np.nanstd(dm), np.nanmax(dm)), ha='right', va='top')
# # labels
# ax2.set_title('Alert magnitude (N=%i)' % len([x for x in dm if not np.isnan(x)]), fontsize=15, fontweight='bold')
# ax2.set_xlabel('Magnitude difference', fontsize=10)
# ax2.set_ylabel('# events', fontsize=10)
# # axes
# ax2.grid(which='both', axis='both')
# # AXIS 3: LOCATION DIFFERENCE
# if dr and sum(np.isnan(dr)) != len(dr):
#     # histogram
#     ax3.hist(dr, bins=np.linspace(start=0, stop=rlim, num=11, endpoint=True), edgecolor='white', zorder=2)
#     # statistics
#     ax3.text(.97*ax3.get_xlim()[1], .98*ax3.get_ylim()[1], '{:.2f} \u00B1 {:.2f} km\nmax = {:.2f} km'.
#              format(np.nanmean(dr), np.nanstd(dr), np.nanmax(dr)), ha='right', va='top')
# # labels
# ax3.set_title('Final location (N=%i)' % len(dr), fontsize=15, fontweight='bold')
# ax3.set_xlabel('Location difference [km]', fontsize=10)
# ax3.set_ylabel('# events', fontsize=10)
# # axes
# ax3.grid(which='both', axis='both')
# ax3.yaxis.set_ticks_position('right')
# ax3.yaxis.set_label_position('right')
# # figure title
# fig.suptitle('EPIC: %s \u2013 %s (%s) \u2013 %i events \u2013 %i alerts' %
#              (datetime.strftime(tbeg, '%d-%m-%Y'), datetime.strftime(tend, '%d-%m-%Y'), f_ext,
#               len(etab.drop_duplicates(subset=['evt_id'], keep='last')),
#               len(etab[etab.alrt_sent == False].drop_duplicates(subset=['evt_id'], keep='last'))), fontsize=20, fontweight='bold')
# # maximise & show figure
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# # adjust plots
# fig.subplots_adjust(left=.05, right=.95, wspace=.2)
# plt.show()

# TO CHECK TRIGGERS VS. ARRIVALS
# continue
# j1 = None
# j2 = None
# j3 = None
# match_ok = 0
# # loop over GSI events within time window
# for j in range(len(itab)):
#     if_evt = 0
#     # time difference
#     if ot_dif[j] < tlim:
#         match_ok += 1
#         j1 = ctab.index[ctab.evt_id == itab.evt_id[j]].to_list()[0]
#     # location difference
#     loc_dif = gdist.distance((etab.evt_lat[ievt[i]], etab.evt_lon[ievt[i]]), (itab.evt_lat[j], itab.evt_lon[j]))
#     if loc_dif.km < rlim:
#         match_ok += 2
#         j2 = ctab.index[ctab.evt_id == itab.evt_id[j]].to_list()[0]
# if match_ok != 3:
#     for j in range(len(itab)):
#         # table for EPIC triggers
#         are = ttab[ttab.evt_id == etab.evt_id[ievt[i]]].reset_index(drop=True)
#         # only keep last version
#         are = are[are.ori_ver == max(are.ori_ver)].reset_index(drop=True)
#         # table for GSI arrivals
#         arc = atab[atab.evt_id == itab.evt_id[j]].reset_index(drop=True)
#         at_dif = []
#         # loop over EPIC triggers
#         for k1 in range(len(are)):
#             # print('  ' + are.trig_sta[k1])
#             # loop over GSI arrivals
#             for k2 in range(len(arc)):
#                 # print('   ' + arc.arr_sta[k2])
#                 if are.trig_sta[k1] == arc.arr_sta[k2] and are.trig_net[k1] == arc.arr_net[k2] and are.trig_chn[k1] == arc.arr_chn[k2]:
#                     tres = abs(are.trig_time[k1] - arc.arr_time[k2])
#                     at_dif.append(tres.total_seconds())
#         if len(at_dif) > 1 and statistics.mean(at_dif) <= tmax:
#             match_ok += 4
#             j3 = ctab.index[ctab.evt_id == itab.evt_id[j]].to_list()[0]
# print(match_ok)
# # check location and trigger tests
# if match_ok == 3 or match_ok == 7:
#     if ctab.evt_id[j1] != ctab.evt_id[j2] != ctab.evt_id[j3]:
#         print('WARNING: All tests did not match with same event')
#         exit()
#     print('%i: True positive' % etab.evt_id[i])
#     tpos.loc[tpos.shape[0]] = [etab.evt_id[ievt[i]], ctab.evt_id[j1]]
# elif match_ok == 0:
#     print('%i: False positive (no acceptable GSI events within %g min of EPIC event)' % (etab.evt_id[i], twin))
#     fpos.loc[fpos.shape[0]] = [etab.evt_id[ievt[i]]]
# elif not ot_ok and loc_ok and not trig_ok:
#     print('%i: Potential match (location)' % etab.evt_id[ievt[i]])
#     to_chk.loc[to_chk.shape[0]] = [etab.evt_id[ievt[i]], ctab.evt_id[j2]]
# elif ot_ok and not loc_ok and not trig_ok:
#     print('%i: Potential match (origin time)' % etab.evt_id[ievt[i]])
#     to_chk.loc[to_chk.shape[0]] = [etab.evt_id[ievt[i]], ctab.evt_id[j1]]
# else:
#     print('%i: Potential match (triggers)' % etab.evt_id[ievt[i]])
#     to_chk.loc[to_chk.shape[0]] = [etab.evt_id[ievt[i]], ctab.evt_id[j3]]

# #####################
# # POTENTIAL MATCHES #
# ifyes = False
# if ifyes:
#     TLim = [-.5, .5]
#     # read station inventory
#     isn_inv = read_inventory('%s/inventory_station_all.xml' % wdir, format='STATIONXML')
#     # read event-matching table
#     evt_to_chk = etab[etab.detec_type == 3].reset_index(drop=True)
#     for i in range(len(evt_to_chk)):
#         # event type
#         if evt_to_chk.mag[i] is not None and evt_to_chk.mag[i] != 0.0:
#             if evt_to_chk.if_exp[i]:
#                 evt_type = 'M%3.2f explosion' % evt_to_chk.mag[i]
#             else:
#                 evt_type = 'M%3.2f earthquake' % evt_to_chk.mag[i]
#         else:
#             if evt_to_chk.if_exp[i]:
#                 evt_type = 'Explosion'
#             else:
#                 evt_type = 'Earthquake'
#         if evt_to_chk.if_felt[i]:
#             evt_type = evt_type + ' (felt)'
#         # events
#         E1 = ctab[ctab.evt_id == evt_to_chk.gsi_id[i]].reset_index(drop=True)
#         E1 = E1.iloc[len(E1)-1]
#         E2 = etab[etab.evt_id == evt_to_chk.epic_id[i]].reset_index(drop=True)
#         E2 = E2.iloc[len(E2)-1]
#         # stations
#         st1 = atab[atab.evt_id == evt_to_chk.gsi_id[i]].reset_index(drop=True)
#         xs1 = []
#         ys1 = []
#         dr1 = []
#         dt1 = []
#         dc1 = []
#         tt1 = pd.DataFrame({'sta': pd.Series(dtype='string'), 'net': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string')})
#         for j in range(len(st1)):
#             S = isn_inv.select(network=st1.arr_net[j], station=st1.arr_sta[j])
#             xs1.append(S.networks[0].stations[0].longitude)
#             ys1.append(S.networks[0].stations[0].latitude)
#             dr1.append(st1.arr_dis[j])
#             dt1.append((st1.arr_time[j] - E1.evt_ot).total_seconds())
#             dc1.append(st1.arr_res[j])
#             tt1.loc[tt1.shape[0]] = [st1.arr_net[j], st1.arr_sta[j], st1.arr_chn[j]]
#         st2 = ttab[ttab.evt_id == evt_to_chk.epic_id[i]].reset_index(drop=True)
#         st2 = st2[st2.ori_ver == max(st2.ori_ver)].reset_index(drop=True)
#         xs2 = []
#         ys2 = []
#         dr2 = []
#         dt2 = []
#         tt2 = pd.DataFrame({'sta': pd.Series(dtype='string'), 'net': pd.Series(dtype='string'), 'chn': pd.Series(dtype='string')})
#         for j in range(len(st2)):
#             xs2.append(st2.trig_lon[j])
#             ys2.append(st2.trig_lat[j])
#             dr2.append(st2.trig_dis[j])
#             dt2.append((st2.trig_time[j] - E2.evt_ot).total_seconds())
#             tt2.loc[tt2.shape[0]] = [st2.trig_net[j], st2.trig_sta[j], st2.trig_chn[j]]
#         # figure
#         fig = plt.figure()
#         fig.suptitle('%s \u2013 %i \n %s' % (evt_to_chk.gsi_id[i], evt_to_chk.epic_id[i], evt_type), fontsize=20, fontweight='bold')
#         gs = GridSpec(2, 3, figure=fig)
#         ax1 = fig.add_subplot(gs[0:, 0])
#         ax2 = fig.add_subplot(gs[0, 1:])
#         ax3 = fig.add_subplot(gs[1, 1:])
#         # AXIS 1: LOCATION MAP
#         # draw map
#         M = Basemap(projection='cyl', llcrnrlon=ngrd[2], llcrnrlat=ngrd[0], urcrnrlon=ngrd[3], urcrnrlat=ngrd[1], resolution='i', ax=ax1)
#         M.drawmapboundary(fill_color='none')
#         M.fillcontinents(color='0.8', lake_color='white')
#         # parallels and meridians (labels = [left,right,top,bottom])
#         M.drawparallels(np.arange(M.llcrnrlat, M.urcrnrlat + 1, 2.), labels=[True, False, True, False])
#         M.drawmeridians(np.arange(M.llcrnrlon, M.urcrnrlon + 1, 2.), labels=[True, False, False, True])
#         # fault lines (Sharon et al., 2020)
#         f = open('/mnt/c/Useres/lewiss/Documents/Data/mapping/Sharon20/Main_faults_shapefile_16.09.2020_1.xyz' % wdir, 'r')
#         flts = f.readlines()
#         f.close()
#         X = []
#         Y = []
#         for k in range(len(flts)):
#             if re.search('NaN', flts[k]):
#                 X = []
#                 Y = []
#             elif k < len(flts) - 1 and re.search('NaN', flts[k + 1]):
#                 ax1.plot(X, Y, '.6', label='Faults')
#             else:
#                 line = flts[k].split()
#                 X.append(float(line[0]))
#                 Y.append(float(line[1]))
#         # locations
#         h1 = ax1.scatter(evt_to_chk.gsi_x[i], evt_to_chk.gsi_y[i], s=100, c='magenta',
#                          edgecolor='black', alpha=.7, zorder=4, marker='*', linewidth=1, label='GSI location')
#         h2 = ax1.scatter(evt_to_chk.epic_x[i], evt_to_chk.epic_y[i], s=100, c='cyan',
#                          edgecolor='black', alpha=.7, zorder=6, marker='*', linewidth=1, label='EPIC location')
#         # stations
#         h3 = ax1.scatter(xs1, ys1, s=25, c=dc1, cmap='RdYlBu', vmin=TLim[0], vmax=TLim[1],
#                          zorder=4, marker='s', label='GSI stations (N=%i)' % len(st1))
#         h4 = ax1.scatter(xs2, ys2, s=16, c='none', edgecolors='black', alpha=.7, zorder=4, marker='o', label='EPIC stations (N=%i)' % len(st2))
#         # colour bar
#         divider = make_axes_locatable(ax1)
#         cax = divider.new_vertical(size='2%', pad=.4, pack_start=True)
#         fig.add_axes(cax)
#         c = plt.colorbar(h3, cax=cax, orientation='horizontal')
#         c.ax.set_xlabel('Travel-time residual [s]')
#         # legend
#         ax1.legend(handles=[h1, h2, h3, h4], loc='lower left')
#         # AXIS 2: GSI ARRIVALS
#         # axes
#         ax2.grid(which='both', axis='both')
#         ax2.set_xlim([0., 100.])
#         ax2.set_ylim([0., 15.])
#         # arrivals
#         hh = ax2.scatter(dr1, dt1, s=25, c=dc1, cmap='RdYlBu', vmin=TLim[0], vmax=TLim[1], marker='s', label='Arrivals (N=%i)' % len(st1))
#         # station list
#         print(tt1[['sta', 'net', 'chn']].agg('.'.join, axis=1).to_list())
#         # ax2.text([1.01*ax2.get_xlim()[1]]*len(tt1), np.arange(0, len(tt1), 1.),
#         #          tt1[['sta', 'net', 'chn']].agg('.'.join, axis=1).to_list(), ha='right', va='bottom')
#         # ax2.scatter([90., 90.], [1., 2.], c='blue', marker='x')
#         # labels
#         ax2.yaxis.tick_right()
#         ax2.yaxis.set_label_position('right')
#         ax2.set_ylabel('Time [s]', fontsize=10)
#         # AXIS 3: EPIC TRIGGERS
#         # labels
#         ax3.yaxis.tick_right()
#         ax3.yaxis.set_label_position('right')
#         ax3.set_xlabel('Distance [km]', fontsize=10)
#         ax3.set_ylabel('Time [s]', fontsize=10)
#         # arrivals
#         ax3.scatter(dr2, dt2, s=16, c='none', edgecolors='black', marker='o', label='Arrivals (N=%i)' % len(st2))
#         # station list
#         # ax3.text(.99*ax3.get_xlim()[1], 1.01*ax3.get_ylim()[0], tt2[['sta', 'net', 'chn']].agg('.'.join, axis=1), ha='right', va='bottom')
#         # axes
#         ax3.grid(which='both', axis='both')
#         ax3.set_xlim([0., 100.])
#         ax3.set_ylim([0., 15.])
#         # maximise & show figure
#         mng = plt.get_current_fig_manager()
#         mng.full_screen_toggle()
#         # adjust plots
#         fig.subplots_adjust(left=.02, right=.96, wspace=.01)
#         plt.show()
#         exit()
