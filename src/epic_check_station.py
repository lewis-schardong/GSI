########################################################################################################################
import os
import sys
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopy.distance as gdist
import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from obspy.core import UTCDateTime
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
import pandas as pd
sys.path.append(f'/home/sysop/olmost/TRUAA/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TRUAA.settings")
import django
django.setup()
from epic.models import EVENT, TRIGGER


def event_rec_sec(stream, index=None, ot_list=None, lbl_list=None, trig_tab=None, ifalert=None, fig_name=None):
    # ________________________________________________________ #
    # stream: data streamer
    # index: index to re-order streamer
    # ot_list: list of event OTs to show over waveforms
    # lbl_list: list of corresponding event labels
    # trig_tab: table containing all triggers to show over waveforms
    # ifalert:
    # ________________________________________________________ #
    # plotting mode
    if fig_name is not None:
        mpl.use('Agg')
    # indexes
    if index is None:
        index = list(range(0, len(stream)))
    # distance limit for vertical axis
    dlim = 1000000.
    # create figure & axis
    if fig_name is not None:
        ff, axis = plt.subplots(squeeze=True, figsize=(18, 9), dpi=200)
    else:
        ff, axis = plt.subplots(squeeze=True)
    if fig_name is None:
        plt.show(block=False)
    # event markers
    if lbl_list is not None:
        h4 = []
        h5 = []
        ind_tit = [kkk for kkk, sss in enumerate(lbl_list) if 'EPIC' in sss][0]
        if ot_list:
            for iii, ot in enumerate(ot_list):
                if iii == ind_tit:
                    h4, = axis.plot([np.datetime64(ot), np.datetime64(ot)], [-1, len(stream)], color='purple', alpha=.7, label='EPIC')
                else:
                    h5, = axis.plot([np.datetime64(ot), np.datetime64(ot)], [-1, len(stream)], color='red', alpha=.7, label='EMSC')
        if lbl_list:
            for iii, lbl in enumerate(lbl_list):
                if iii != ind_tit:
                    axis.text(np.datetime64(ot_list[iii]), len(stream), lbl, color='red', ha='center', va='bottom')
        # title
        ot = str(datetime.strftime(datetime.strptime(str(ot_list[ind_tit]), '%Y-%m-%d %H:%M:%S.%f'), '%d/%m/%Y %H:%M:%S'))
        if trig_tab is not None:
            if ifalert:
                axis.set_title(f"{lbl_list[ind_tit]} {ot}\n({len(trig_tab)} triggers, alert sent)\n\n", fontweight='bold')
            else:
                axis.set_title(f"{lbl_list[ind_tit]} {ot}\n({len(trig_tab)} triggers, no alert)\n\n", fontweight='bold')
    # loop over stream channels
    stn_lbl = []
    h1 = []
    h2 = []
    h3 = []
    # initialise counter
    n_trace = 0
    n_max = None
    for jjj in index:
        stn_lbl.append(f"{stream[jjj].stats.network}.{stream[jjj].stats.station}.{stream[jjj].stats.location}.{stream[jjj].stats.channel}")
        # time vector
        t_vec = np.arange(0, len(stream[jjj])) * np.timedelta64(int(stream[jjj].stats.delta*1000), '[ms]')\
            + np.datetime64(str(stream[jjj].stats.starttime)[:-1])
        # plot waveform
        h1, = axis.plot(t_vec, stream[jjj].data / stream[jjj].max() + n_trace, color='grey', alpha=.7, label='Velocity')
        # # display distance
        # ht = None
        # if stream[jjj].stats.distance/1000. <= dlim:
        #     ht = axis.text(axis.get_xlim()[1]+(axis.get_xlim()[1]-axis.get_xlim()[0])/100., n_trace,
        #                    f"{stream[jjj].stats.distance/1000.:.2f} km", ha='left', va='center', fontsize=8)
        #     if n_max is None:
        #         n_max = n_trace
        if trig_tab is not None:
            # EPIC triggers
            sta_tab = trig_tab[trig_tab.trig_sta == stream[jjj].stats.station].reset_index(drop=True)
            if not sta_tab.empty:
                for iii in range(len(sta_tab)):
                    h2, = axis.plot([sta_tab.trig_time[iii], sta_tab.trig_time[iii]], [n_trace-1, n_trace+1], color='orange', alpha=.7, label='Trigger')
                # if ht:
                #     ht.set_bbox(dict(facecolor='white', alpha=.6, edgecolor='orange'))
        # theoretical arrivals
        if not np.isnat(stream[jjj].stats.theo_tt):
            h3, = axis.plot([stream[jjj].stats.theo_tt, stream[jjj].stats.theo_tt], [n_trace-1, n_trace+1], ':b', alpha=.7, label=rmod)
        n_trace += 1
    # # case no stations are < [dlim] km
    # n_again = 0
    # if n_max is None:
    #     for jjj in index:
    #         # display distance
    #         ht = axis.text(axis.get_xlim()[1]+(axis.get_xlim()[1]-axis.get_xlim()[0])/100., n_again,
    #                        f"{stream[jjj].stats.distance/1000.:.2f} km", ha='left', va='center', fontsize=8)
    #         # highlight EPIC triggers
    #         sta_tab = trig_tab[trig_tab.trig_sta == stream[jjj].stats.station].reset_index(drop=True)
    #         if not sta_tab.empty:
    #             if ht:
    #                 ht.set_bbox(dict(facecolor='white', alpha=.6, edgecolor='orange'))
    #         n_again += 1
    axis.set_xlabel('Time [s]', fontweight='bold', fontsize=15)
    axis.set_ylabel('Station', fontweight='bold', fontsize=15)
    # axes
    axis.grid(which='both', axis='both')
    date_form = DateFormatter('%d/%m/%Y %H:%M:%S')
    axis.xaxis.set_major_formatter(date_form)
    # # legend
    # if lbl_list is None:
    #     axis.legend(handles=h1, loc='lower left', fontsize=8)
    # else:
    #     if not h2 and not h5:
    #         axis.legend(handles=[h4, h1], loc='lower left', fontsize=8)
    #     elif not h2 and h5:
    #         axis.legend(handles=[h4, h1, h3, h5], loc='lower left', fontsize=8)
    #     elif h2 and not h5:
    #         axis.legend(handles=[h4, h1, h2], loc='lower left', fontsize=8)
    #     else:
    #         axis.legend(handles=[h4, h1, h2, h3, h5], loc='lower left', fontsize=8)
    # replace y-axis tick labels with station names
    axis.set_yticks(np.arange(0, n_trace, 1))
    axis.set_yticklabels(stn_lbl, fontsize=8)
    # find TMR index
    tmr = [iii for iii in range(len(stream)) if stream[index[iii]].stats.station == 'TMR'][0]
    if tmr+10 > len(stream):
        axis.set_ylim([tmr-10.5, len(stream)])
    else:
        axis.set_ylim([tmr-10.5, tmr+10.5])
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # show or save figure
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight', dpi='figure')
        plt.close()
    else:
        plt.show()


# station of interest
stn = 'TMR'
ntw = 'IS'
src = 'eew-b-jer'

# area of ineterest
rrad = 6371.
rmod = 'gitt05'

# directories
wdir = '/mnt/c/Users/lewiss/Documents/Research/EPIC'
# make sure station directory exists
f_ext = ''
if '-b-' in src:
    if 'jer' in src:
        f_ext = 'JER-B'
    elif 'lod' in src:
        f_ext = 'LOD-B'
elif '-r-' in src:
    if 'jer' in src:
        f_ext = 'JER-R'
    elif 'lod' in src:
        f_ext = 'LOD-R'
if path.exists(f"{wdir}/{f_ext}/stations/{stn}") == 0:
    os.mkdir(f"{wdir}/{f_ext}/stations/{stn}")
# figure directory
mpl.rcParams['savefig.directory'] = f"{wdir}/{f_ext}/stations"

# FDSN databases
# isn_client = Client('http://172.16.46.102:8181/')       # jfdsn
isn_client = Client('http://172.16.46.140:8181/')       # jtfdsn
# # read ISN station inventory
fdsn_inv = read_inventory('/mnt/c/Users/lewiss/Documents/Research/Autopicker/inventory.xml', format='STATIONXML')
# station period of activity
start_date = fdsn_inv.select(network=ntw, station=stn).networks[0].stations[0].creation_date.date
print(f"Looking into {src} data recorded at {stn} from {start_date.strftime('%d/%m/%Y')}")

#####################
# RETRIEVE TRIGGERS #
if path.exists(f"{wdir}/{f_ext}/stations/{stn}/triggers.csv") == 0:
    # querry triggers for station of interest from database
    triggers = TRIGGER.objects.filter(source__contains=src, sta__contains=stn, net__contains=ntw,
                                      trigger_time__range=[UTCDateTime(start_date), UTCDateTime(datetime.now())]).order_by('eventid', 'ver')
    # initialise output table
    ttab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'ori_ver': pd.Series(dtype='int64'), 'trig_ver': pd.Series(dtype='int64'),
                         'trig_ord': pd.Series(dtype='int64'), 'trig_sta': pd.Series(dtype='string'), 'trig_net': pd.Series(dtype='string'),
                         'trig_chn': pd.Series(dtype='string'), 'trig_loc': pd.Series(dtype='string'), 'trig_lat': pd.Series(dtype='float64'),
                         'trig_lon': pd.Series(dtype='float64'), 'trig_time': pd.Series(dtype='datetime64[ns]')})
    # loop over triggers
    for i in range(len(triggers)):
        tr = triggers.values()[i]
        ttab.loc[ttab.shape[0]] = [tr['eventid'], tr['ver'], tr['update'], tr['order'], tr['sta'], tr['chan'], tr['net'], tr['loc'],
                                   tr['lat'], tr['lon'], np.datetime64(str(tr['trigger_time']).replace('+00:00', ''))]
    # keep unique version of triggers
    ttab = ttab.sort_values(by=['evt_id', 'ori_ver'], ascending=False).drop_duplicates(subset=['evt_id'], keep='first').reset_index(drop=True)
    # order by datetime
    ttab = ttab.sort_values(by=['trig_time']).reset_index(drop=True)
    # save table
    ttab.to_csv(f"{wdir}/{f_ext}/stations/{stn}/triggers.csv", index=False, float_format='%.4f')
else:
    ttab = pd.read_csv(f"{wdir}/{f_ext}/stations/{stn}/triggers.csv", parse_dates=['trig_time'])
print()
if len(ttab) > 0:
    print(f"Found triggers for {stn} in {len(ttab)} events")
else:
    print(f"No triggers found for {stn}")
# initialise event table
etab = pd.DataFrame({'evt_id': pd.Series(dtype='int64'), 'ori_ver': pd.Series(dtype='int64'), 'evt_lat': pd.Series(dtype='float64'),
                     'evt_lon': pd.Series(dtype='float64'), 'evt_dep': pd.Series(dtype='float64'), 'evt_mag': pd.Series(dtype='float64'),
                     'evt_ot': pd.Series(dtype='datetime64[ns]')})
# query event info
evt_lbl = []
for i in range(len(ttab)):
    event = EVENT.objects.filter(source__contains=src, eventid=ttab.evt_id[i], ver=ttab.ori_ver[i])\
        .values('eventid', 'ver', 'evlat', 'evlon', 'dep', 'mag', 'time')[0]
    # event table
    etab.loc[etab.shape[0]] = [event['eventid'], event['ver'], event['evlat'], event['evlon'],
                               event['dep'], event['mag'], np.datetime64(str(event['time']).replace('+00:00', ''))]
    # event label
    evt_lbl.append(f"{event['eventid']}: {event['time'].strftime('%d/%m/%Y %H:%M:%S')} [M{event['mag']:3.1f}]")

#########################
# EVENT RECORD SECTIONS #
# initialising TauP
model = TauPyModel(model=rmod)
for i in range(len(ttab)):
    print(etab.evt_id[i])
    seed = f"{wdir}/{f_ext}/stations/{stn}/{ttab.evt_id[i]}_all.mseed"
    if path.exists(seed) == 0:
        # retrieve waveforms
        traces = isn_client.get_waveforms('IS,GE', '*', '*', 'ENZ,HHZ', starttime=UTCDateTime(ttab.trig_time[i])-timedelta(minutes=5),
                                          endtime=UTCDateTime(ttab.trig_time[i])+timedelta(minutes=5)).merge()
        # remove response
        for tr in traces:
            if tr.stats.station == 'KRPN':
                traces.remove(tr)
                continue
        traces.remove_response(output='VEL', inventory=fdsn_inv)
        # apply taper
        traces.taper(max_percentage=.5, type='cosine', max_length=10., side='left')
        # remove trend
        traces.detrend('spline', order=3, dspline=500)
        # write miniSEED file
        traces.write(seed)
    if path.exists(seed.replace('.mseed', '.png')) == 0:
        # read waveforms
        traces = read(seed)
        # calculate distances
        for tr in traces:
            # station info
            st = fdsn_inv.select(station=tr.stats.station, network=tr.stats.network,
                                 location=tr.stats.location, channel=tr.stats.channel).networks[0].stations[0]
            dis = gdist.distance((etab.evt_lat[i], etab.evt_lon[i]), (st.latitude, st.longitude))
            tr.stats.distance = dis.m
            # theoretical arrival time
            x = model.get_travel_times(source_depth_in_km=etab.evt_dep[i], distance_in_degree=dis.km / (2. * np.pi * rrad / 360.),
                                       phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
            tr.stats['theo_tt'] = np.datetime64(etab.evt_ot[i] + timedelta(seconds=x[0].time))
        # indexes to sort according to distance (descending)
        sorted_ind = np.argsort([tr.stats.distance/1000. for tr in traces])[::-1][:len(traces)]
        # plot record section
        event_rec_sec(traces, sorted_ind, [etab.evt_ot[i]], ['EPIC'])
exit()

##################
# SUMMARY FIGURE #
i = 0
# create figure & axis
ff, ax = plt.subplots(squeeze=True)
ax.grid(which='both', axis='both')
plt.show(block=False)
h1 = []
h2 = []
h3 = []
# looping over events
for i in range(len(ttab)):
    # load ISN traces
    seed = f"{wdir}/{f_ext}/stations/{stn}/{ttab.evt_id[i]}.mseed"
    if path.exists(seed) == 0:
        # retrieve waveform
        tr = isn_client.get_waveforms('*', stn, '*', '??Z', starttime=UTCDateTime(ttab.trig_time[i])-timedelta(minutes=5),
                                      endtime=UTCDateTime(ttab.trig_time[i])+timedelta(minutes=5)).merge()[0]
        # remove response
        tr.remove_response(output='VEL', inventory=fdsn_inv)
        # apply taper
        tr.taper(max_percentage=.5, type='cosine', max_length=10., side='left')
        # remove trend
        tr.detrend('spline', order=3, dspline=500)
        # write miniSEED file
        tr.write(seed)
    else:
        # read miniSEED file
        tr = read(seed)[0]
    # time vector
    t = (np.arange(0, len(tr)) * np.timedelta64(int(tr.stats.delta * 1000.), '[s]')) / 1000. + \
        np.timedelta64(np.datetime64(tr.stats.starttime)-np.datetime64(etab.evt_ot[i]), '[s]')
    # plot waveform
    h1, = ax.plot(t, tr.data / tr.max() + i, color='grey', alpha=.7, label='Velocity')
    # plot trigger
    trig = (ttab.trig_time[i] - etab.evt_ot[i]).total_seconds()
    h2, = ax.plot([trig, trig], [i-1, i+1], color='orange', alpha=.7, label='Trigger')
    # compute distance from event
    dis = gdist.distance((etab.evt_lat[i], etab.evt_lon[i]), (ttab.trig_lat[i], ttab.trig_lon[i]))
    # compute theoretical travel times for EPIC event
    x = model.get_travel_times(source_depth_in_km=etab.evt_dep[i], distance_in_degree=dis.km / (2. * np.pi * rrad / 360.),
                               phase_list=['p', 'P', 'Pg', 'Pn', 'Pdiff'])
    # plot theoretical arrival
    h3, = ax.plot([x[0].time, x[0].time], [i-1, i+1], ':b', alpha=.7, label=rmod)
# replace y-axis tick labels with station names
ax.text(-305., i+1, 'Origin times', fontsize=8, fontweight='bold', ha='right', va='center')
ax.set_yticks(np.arange(0, i+1, 1))
ax.set_yticklabels(evt_lbl, fontsize=8)
ax.set_xlim([-300., 300.])
ax.set_ylim([-1., i+1])
# display trigger times
ax.text(305., i+1, 'Trigger times', fontsize=8, fontweight='bold', ha='left', va='center')
for i in range(len(ttab)):
    ax.text(305., i, ttab.trig_time[i].strftime('%d/%m/%Y %H:%M:%S'), fontsize=8, ha='left', va='center')
ax.set_title(stn, fontweight='bold', fontsize=20)
# labels
ax.set_xlabel('Time [s]', fontweight='bold')
# legend
ax.legend(handles=[h1, h2, h3], loc='upper right')
# maximise figure
plt.get_current_fig_manager().full_screen_toggle()
plt.show()
