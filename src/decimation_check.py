########################################################################################################################
import os
import re
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from obspy import read
from obspy.clients.fdsn import Client


def get_traces_deci(evt_id, time_win, networks='IS', channels='(B|H|E)(H|N)(Z|N|E)'):
    """
    :param evt_id: event ID
    :param time_win: half-length of time window; default is 5 minutes
    :param networks: list of desired seismic networks; default is IS
    :param channels: list of desired seismic channels; default is BH/EN/HH for all three components (Z/N/E)
    :return: full path of created miniSEED file
    """
    # directory to store .mseed file
    ddir = '/mnt/c/Users/lewiss/Documents/Research/Data'
    # directory of archive for decimated data
    adir = '/net/172.16.46.200/HI_Archive'
    # define file extension based on time window length
    # load miniSEED data
    mseed = f'{ddir}/{evt_id}.deci.mseed'
    if path.exists(mseed) == 0:
        ori_time = ''
        if re.search('gsi', evt_id):
            ori_time = datetime.strptime(evt_id.replace('gsi', ''), '%Y%m%d%H%M')
        elif re.search('RTLOC2', evt_id):
            ori_time = datetime.strptime(evt_id.replace('RTLOC2', ''), '%Y%j%H%M%S%f')
        # import miniSEED file
        tbeg = str(datetime.strftime(ori_time - time_win, '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(ori_time + time_win, '%Y-%m-%d %H:%M:%S'))
        os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{tbeg}~{tend}" {adir} > {mseed}')
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    else:
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    return mseed


def get_traces_full(evt_id, time_win, networks='IS', channels='(B|H|E)(H|N)(Z|N|E)'):
    """
    :param evt_id: event ID
    :param time_win: half-length of time window; default is 5 minutes
    :param networks: list of desired seismic networks; default is IS
    :param channels: list of desired seismic channels; default is BH/EN/HH for all three components (Z/N/E)
    :return: full path of created miniSEED file
    """
    # directory to store .mseed file
    ddir = '/mnt/c/Users/lewiss/Documents/Research/Data'
    # directory of archive for full-resolution data
    adir = '/net/172.16.46.122/mnt/archivet'
    # define file extension based on time window length
    # load miniSEED data
    mseed = f'{ddir}/{evt_id}.full.mseed'
    if path.exists(mseed) == 0:
        ori_time = ''
        if re.search('gsi', evt_id):
            ori_time = datetime.strptime(evt_id.replace('gsi', ''), '%Y%m%d%H%M')
        elif re.search('RTLOC2', evt_id):
            ori_time = datetime.strptime(evt_id.replace('RTLOC2', ''), '%Y%j%H%M%S%f')
        # import miniSEED file
        tbeg = str(datetime.strftime(ori_time - time_win, '%Y-%m-%d %H:%M:%S'))
        tend = str(datetime.strftime(ori_time + time_win, '%Y-%m-%d %H:%M:%S'))
        os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{tbeg}~{tend}" {adir} > {mseed}')
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    else:
        if os.path.getsize(mseed) == 0:
            os.remove(mseed)
            mseed = ''
    return mseed


########################################################################################################################
# working directory
wdir = '/mnt/c/Users/lewiss/Documents/Research/Data'
print(f'Working directory: {wdir}\n')
mpl.rcParams['savefig.directory'] = f"{wdir}"

# FDSN database
isn_client = Client('http://172.16.46.102:8181/')

# event info
# event_id = 'gsi202201222136'
# event_id = 'gsi202202160321'
# event_id = 'gsi202206291744'
# event_id = 'gsi202211181135'
# event_id = 'gsi202212090504'
event_id = 'RTLOC2202303710255492'
evt = isn_client.get_events(eventid=event_id)[0]
# retrieve decimated data
if path.exists(f'{wdir}/{event_id}.deci.mseed') == 0:
    get_traces_deci(event_id, timedelta(minutes=5.), 'IS', '(E|H)(N|H)(N|E|Z)')
st2 = read(f'{wdir}/{event_id}.deci.mseed')
st2.detrend('spline', order=3, dspline=500)
# retrieve full-resolution data
if path.exists(f'{wdir}/{event_id}.full.mseed') == 0:
    get_traces_full(event_id, timedelta(minutes=5.), 'IS', '(E|H)(N|H)(N|E|Z)')
st1 = read(f'{wdir}/{event_id}.full.mseed')
st1.detrend('spline', order=3, dspline=500)

# data selection and filtering
cmp = 'HH'
# vertical component
tr11 = st1.select(network='IS', station='DSI', channel=f'{cmp}Z')[0]
tr12 = st2.select(network='IS', station='DSI', channel=f'{cmp}Z')[0]
# north-south component
tr21 = st1.select(network='IS', station='DSI', channel=f'{cmp}N')[0]
tr22 = st2.select(network='IS', station='DSI', channel=f'{cmp}N')[0]
# east-west component
tr31 = st1.select(network='IS', station='DSI', channel=f'{cmp}E')[0]
tr32 = st2.select(network='IS', station='DSI', channel=f'{cmp}E')[0]
# filter all selected data
bp_freq = []
# bp_freq = [8., 20.]
# bp_freq = [1., 10.]
# bp_freq = [2., 6.]
if bp_freq:
    tr11.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)
    tr12.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)
    tr21.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)
    tr22.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)
    tr31.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)
    tr32.filter('bandpass', freqmin=bp_freq[0], freqmax=bp_freq[1], corners=4)


if_frq_dom = True
if not if_frq_dom:
    # TIME DOMAIN FIGURE
    # create figure & axes
    fig, (axis1, axis2, axis3) = plt.subplots(nrows=3, ncols=1)
    # show axis grids
    axis1.grid(which='both', axis='both')
    axis2.grid(which='both', axis='both')
    axis3.grid(which='both', axis='both')
    # set x-axis limits
    if event_id == 'RTLOC2202303710255492':
        axis1.set_xlim([datetime.strptime('2023-02-06 10:26:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2023-02-06 10:32:00', '%Y-%m-%d %H:%M:%S')])
        axis2.set_xlim([datetime.strptime('2023-02-06 10:26:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2023-02-06 10:32:00', '%Y-%m-%d %H:%M:%S')])
        axis3.set_xlim([datetime.strptime('2023-02-06 10:26:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2023-02-06 10:32:00', '%Y-%m-%d %H:%M:%S')])
    # display channels as y-axis labels
    axis1.set_ylabel(f'{tr11.stats.network}.{tr11.stats.station}.{tr11.stats.location}.{tr11.stats.channel}', fontsize=10, fontweight='bold')
    axis2.set_ylabel(f'{tr21.stats.network}.{tr21.stats.station}.{tr21.stats.location}.{tr21.stats.channel}', fontsize=10, fontweight='bold')
    axis3.set_ylabel(f'{tr31.stats.network}.{tr31.stats.station}.{tr31.stats.location}.{tr31.stats.channel}', fontsize=10, fontweight='bold')
    # set date format to x-axis tick labels
    axis1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    axis2.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    axis3.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # time vectors
    t11 = np.arange(0, len(tr11)) * np.timedelta64(int(tr11.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr11.stats.starttime)[:-1])
    t12 = np.arange(0, len(tr12)) * np.timedelta64(int(tr12.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr12.stats.starttime)[:-1])
    t21 = np.arange(0, len(tr21)) * np.timedelta64(int(tr21.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr21.stats.starttime)[:-1])
    t22 = np.arange(0, len(tr22)) * np.timedelta64(int(tr22.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr22.stats.starttime)[:-1])
    t31 = np.arange(0, len(tr31)) * np.timedelta64(int(tr31.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr31.stats.starttime)[:-1])
    t32 = np.arange(0, len(tr32)) * np.timedelta64(int(tr32.stats.delta * 1000), '[ms]')\
        + np.datetime64(str(tr32.stats.starttime)[:-1])
    # plot waveforms
    h11, = axis1.plot(t11, tr11.data, linewidth=.5, alpha=.7, label=f'{tr11.stats.sampling_rate} Hz')
    h12, = axis1.plot(t12, tr12.data, linewidth=.5, alpha=.7, label=f'{tr12.stats.sampling_rate} Hz')
    h21, = axis2.plot(t21, tr21.data, linewidth=.5, alpha=.7, label=f'{tr21.stats.sampling_rate} Hz')
    h22, = axis2.plot(t22, tr22.data, linewidth=.5, alpha=.7, label=f'{tr22.stats.sampling_rate} Hz')
    h31, = axis3.plot(t31, tr31.data, linewidth=.5, alpha=.7, label=f'{tr31.stats.sampling_rate} Hz')
    h32, = axis3.plot(t32, tr32.data, linewidth=.5, alpha=.7, label=f'{tr32.stats.sampling_rate} Hz')
    # event origin time
    axis1.plot([evt.preferred_origin().time.datetime, evt.preferred_origin().time.datetime], [axis1.get_ylim()[0], axis1.get_ylim()[1]], color='red')
    axis2.plot([evt.preferred_origin().time.datetime, evt.preferred_origin().time.datetime], [axis2.get_ylim()[0], axis2.get_ylim()[1]], color='red')
    axis3.plot([evt.preferred_origin().time.datetime, evt.preferred_origin().time.datetime], [axis3.get_ylim()[0], axis3.get_ylim()[1]], color='red')
    # legends
    axis1.legend()
    axis2.legend()
    axis3.legend()
    # figure title
    if not bp_freq:
        if evt.preferred_magnitude():
            fig.suptitle(f"M{evt.preferred_magnitude().mag:3.1f} {evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}", fontweight='bold')
        else:
            fig.suptitle(f"{evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}", fontweight='bold')
    else:
        if evt.preferred_magnitude():
            fig.suptitle(f"M{evt.preferred_magnitude().mag:3.1f} {evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}\n"
                         f"{bp_freq[0]} \u2013 {bp_freq[1]} Hz", fontweight='bold')
        else:
            fig.suptitle(f"{evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}\n"
                         f"{bp_freq[0]} \u2013 {bp_freq[1]} Hz", fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.07, right=.98, top=.92, wspace=.1)
    # visualise figure
    plt.show()
else:
    # create figure & axes
    fig, ((axis1, axis2), (axis3, axis4), (axis5, axis6)) = plt.subplots(nrows=3, ncols=2)
    # show axis grids
    axis1.grid(which='both', axis='both')
    axis2.grid(which='both', axis='both')
    axis3.grid(which='both', axis='both')
    axis4.grid(which='both', axis='both')
    axis5.grid(which='both', axis='both')
    axis6.grid(which='both', axis='both')
    # plot spectrograms
    tr11.spectrogram(axes=axis1, dbscale=True, wlen=2.0)
    axis1.images[0].set_clim([-100, 100])
    axis1.set_ylim([0, 25])
    tr12.spectrogram(axes=axis2, dbscale=True, wlen=2.0)
    axis2.images[0].set_clim([-100, 100])
    axis2.set_ylim([0, 25])
    tr21.spectrogram(axes=axis3, dbscale=True, wlen=2.0)
    axis3.images[0].set_clim([-100, 100])
    axis3.set_ylim([0, 25])
    tr22.spectrogram(axes=axis4, dbscale=True, wlen=2.0)
    axis4.images[0].set_clim([-100, 100])
    axis4.set_ylim([0, 25])
    tr31.spectrogram(axes=axis5, dbscale=True, wlen=2.0)
    axis5.images[0].set_clim([-100, 100])
    axis5.set_ylim([0, 25])
    tr32.spectrogram(axes=axis6, dbscale=True, wlen=2.0)
    axis6.images[0].set_clim([-100, 100])
    axis6.set_ylim([0, 25])
    # colour bars
    c1 = plt.colorbar(axis1.images[0], ax=axis1, orientation='vertical')
    c2 = plt.colorbar(axis2.images[0], ax=axis2, orientation='vertical')
    c3 = plt.colorbar(axis3.images[0], ax=axis3, orientation='vertical')
    c4 = plt.colorbar(axis4.images[0], ax=axis4, orientation='vertical')
    c5 = plt.colorbar(axis5.images[0], ax=axis5, orientation='vertical')
    c6 = plt.colorbar(axis6.images[0], ax=axis6, orientation='vertical')
    # display channels
    axis1.set_title(f'{tr11.stats.network}.{tr11.stats.station}.{tr11.stats.location}.{tr11.stats.channel} @ {tr11.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    axis2.set_title(f'{tr12.stats.network}.{tr12.stats.station}.{tr12.stats.location}.{tr12.stats.channel} @ {tr12.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    axis3.set_title(f'{tr21.stats.network}.{tr21.stats.station}.{tr21.stats.location}.{tr21.stats.channel} @ {tr21.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    axis4.set_title(f'{tr22.stats.network}.{tr22.stats.station}.{tr22.stats.location}.{tr22.stats.channel} @ {tr22.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    axis5.set_title(f'{tr31.stats.network}.{tr31.stats.station}.{tr31.stats.location}.{tr31.stats.channel} @ {tr31.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    axis6.set_title(f'{tr32.stats.network}.{tr32.stats.station}.{tr32.stats.location}.{tr32.stats.channel} @ {tr32.stats.sampling_rate} s', fontsize=10, fontweight='bold')
    # axes labels
    axis1.set_ylabel('Frequency [Hz]')
    axis3.set_ylabel('Frequency [Hz]')
    axis5.set_ylabel('Frequency [Hz]')
    axis5.set_xlabel('Time [s]')
    axis6.set_xlabel('Time [s]')
    # display event/time period
    if evt.preferred_magnitude():
        fig.suptitle(f"M{evt.preferred_magnitude().mag:3.1f} {evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}", fontweight='bold')
    else:
        fig.suptitle(f"{evt.preferred_origin().time.datetime.strftime('%d/%m/%Y %H:%M:%S.%f')}", fontweight='bold')
    # maximise figure
    plt.get_current_fig_manager().full_screen_toggle()
    # adjust plots
    fig.subplots_adjust(left=.07, right=.98, top=.92, wspace=.1)
    # visualise figure
    plt.show()
