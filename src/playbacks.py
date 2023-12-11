import pandas as pd
import numpy as np
import os
import re
from calendar import monthrange
from datetime import datetime, timedelta
from obspy import read_events
import xmltodict
import json

# experiment number
exp = 0
# year of interest
y = 2023
# month of interest
m = 9

# input parameters
networks = 'IS,GE'
channels = '(B|H|E|S)(H|N)Z'

# paths
wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/playback'
vdir = '/net/172.16.46.200/archive'
adir = '/net/172.16.46.122/mnt/archiveb'

# reading catalogue events
ctab = pd.read_csv(f'{wdir}/cat_evt_202309.csv', parse_dates=['DateTime'])\
    .rename(columns={'epiid': 'EventID', 'Depth(Km)': 'Depth'})
# remove quotation marks from event IDs and add 'gsi'
ctab.EventID = 'gsi' + ctab['EventID'].str.replace("'", "")

# read autopicker configuration
with open(f'{wdir}/scautopick.cfg') as f:
    txt = f.read()
deadTime = float(txt[re.search('triggerDeadTime', txt).span()[1]:
                 re.search('triggerDeadTime', txt).span()[1]+20].replace(' ', '').replace('s', ''))
minAmpOffs = float(txt[re.search('amplitudeMinOffset', txt).span()[1]:
                   re.search('amplitudeMinOffset', txt).span()[1]+20].replace(' ', '').replace('s', ''))
# autoloc configuration (no config dumping possible)
minNumPh = 6

# prepare output table
otab = pd.DataFrame({'EventID': pd.Series(dtype='string'), 'Network': pd.Series(dtype='string'),
                     'Station': pd.Series(dtype='string'), 'Location': pd.Series(dtype='string'),
                     'Channel': pd.Series(dtype='string'), 'TrigID': pd.Series(dtype='string'),
                     'TrigTime': pd.Series(dtype='datetime64[ms]'), 'TrigSNR': pd.Series(dtype='float64'),
                     'PickID': pd.Series(dtype='string'), 'PickTime': pd.Series(dtype='datetime64[ms]'),
                     'PickSNR': pd.Series(dtype='float64'), 'PickTres': pd.Series(dtype='float64'),
                     'filter': pd.Series(dtype='string'), 'triggerOn': pd.Series(dtype='float64'),
                     'triggerOff': pd.Series(dtype='float64'), 'minSNR': pd.Series(dtype='float64'),
                     'minPhCount': pd.Series(dtype='int64'), 'minAmplOffs': pd.Series(dtype='float64'),
                     'deadTime': pd.Series(dtype='float64')})
# initialise variables
nm = 0          # Missed events counter
nf = 0          # False events counter
avgnpick = []   # average number of picks per station
rmstres = []    # average pick time residual
avgsnr = []     # average pick SNR
# loop over days of the month
for d in range(0, monthrange(y, m)[1]+1):
    # prepare waveforms if needed
    if os.path.exists(f'{wdir}/{y:04d}{m:02d}{d:02d}.mseed') == 0:
        # retrieve velocity waveforms
        if os.path.exists(f'{wdir}/{y:04d}{m:02d}{d:02d}.vel.mseed') == 0:
            os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{y:04d}-{m:02d}-{d:02d} 00:00:00~'
                      f'{y:04d}-{m:02d}-{d+1:02d} 00:00:00" {vdir} > {wdir}/{y:04d}{m:02d}{d:02d}.vel.mseed')
        # retrieve acceleration waveforms
        if os.path.exists(f'{wdir}/{y:04d}{m:02d}{d:02d}.acc.mseed') == 0:
            os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{y:04d}-{m:02d}-{d:02d} 00:00:00~'
                      f'{y:04d}-{m:02d}-{d+1:02d} 00:00:00" {adir} > {wdir}/{y:04d}{m:02d}{d:02d}.acc.mseed')
        # merge the two mseed files
        os.system(f'cp -p {wdir}/{y:04d}{m:02d}{d:02d}.vel.mseed {wdir}/{y:04d}{m:02d}{d:02d}.mseed')
        os.system(f'cat {wdir}/{y:04d}{m:02d}{d:02d}.acc.mseed >> {wdir}/{y:04d}{m:02d}{d:02d}.mseed')
    # # check the playback was ran
    # if not os.path.exists(f'{wdir}/{y:04d}{m:02d}{d:02d}.{exp}.xml'):
    #     print(f'Run playback for {y:04d}{m:02d}{d:02d}')
    #     continue
    # # read event xml file (not with ObsPy to get SNR values and corresponding pick IDs) and convert to dictionary
    # with open(f'{wdir}/{y:04d}{m:02d}{d:02d}.{exp}.xml') as f:
    #     data = xmltodict.parse(f.read())
    # # read only relevant parts of dictionary
    # catalog = data['seiscomp']['EventParameters']
    # # convert to JSO > pd
    # P = pd.read_json(json.dumps(catalog['pick'])).rename(columns={'@publicID': 'publicID'})
    # A = pd.read_json(json.dumps(catalog['amplitude'])).rename(columns={'@publicID': 'publicID'})
    # # read configuration xml file and convert to dictionary
    # with open(f'{wdir}/config.{exp}.xml') as f:
    #     data = xmltodict.parse(f.read())
    # # read only relevant parts of dictionary
    # cfg = data['seiscomp']['Config']['parameterSet']
    # # convert to JSO > pd
    # config = pd.read_json(json.dumps(cfg))
    # config['network'] = config['@publicID'].apply(lambda x: x.split('/')[3])
    # config['station'] = config['@publicID'].apply(lambda x: x.split('/')[4])
    # config['paramtyp'] = config['@publicID'].apply(lambda x: x.split('/')[5])
    # config['parameters'] = config.loc[config.paramtyp == 'scautopick'].apply(lambda x: x['parameter'], axis=1)
    # # read xml file with ObsPy
    # xml = read_events(f'{wdir}/{y:04d}{m:02d}{d:02d}.{exp}.xml')
    # xevt = xml.events[0]
    # # loop over arrivals
    # for arr in xevt.preferred_origin().arrivals:
    #     # ignore S arrivals
    #     if arr.phase == 'S':
    #         continue
    #     # get arrival ID
    #     aid = arr.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '').split('_')[0]
    #     # getting AIC pick and SNR
    #     ain1 = A.publicID.str.match(aid + '.snr')
    #     pin1 = P.publicID.str.match(aid)
    #     apid = A.pickID[ain1].item()
    #     atim = datetime.strptime(P.time.values[pin1][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ')
    #     asnr = A.amplitude.values[ain1][0]['value']
    #     # print(f"  AIC: {apid}: {atim} ({asnr})")
    #     # looking for Trigger pick and SNR
    #     ain2 = A.pickID.str.match(aid.replace('-AIC', ''))
    #     # trying to find direct match
    #     pin2 = None
    #     if not A[ain2].empty:
    #         pin2 = P.publicID.str.match(aid.replace('-AIC', ''))
    #         tpid = A.pickID[ain1].item()
    #         ttim = datetime.strptime(P.time.values[pin2][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ')
    #         tsnr = A.amplitude.values[ain1][0]['value']
    #     else:
    #         ain2 = None
    #         # extract channel info
    #         chn = aid.replace('-AIC', '').split('-')[1]
    #         # look for Trigger closest to AIC picks
    #         if not A[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')].empty:
    #             tdif = [abs((atim-t).total_seconds()) for t in
    #                     pd.to_datetime(A[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')].
    #                                    pickID.str.split('-', n=1, expand=True)[0], format='%Y%m%d.%H%M%S.%f').to_list()]
    #             imin = tdif.index(min(tdif))
    #             tpid = A[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')].pickID.iloc[imin]
    #             ttim = datetime.strptime(
    #                 P.time.values[P.publicID.str.contains(chn) & ~P.publicID.str.contains('AIC')][imin]['value'],
    #                 '%Y-%m-%dT%H:%M:%S.%fZ')
    #             tsnr = A.amplitude.values[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')][imin]['value']
    #             ain2 = A[A.pickID.str.match(tpid)].index
    #             pin2 = P[P.publicID.str.match(tpid)].index
    #     # print(f"  Trig: {tpid}: {ttim} ({tsnr})")
    #     # print('')
    #     # check location exists
    #     if not P.waveformID[P.publicID.str.match(aid)].astype(str).str.contains('locationCode').item():
    #         loc = ''
    #     else:
    #         loc = P.waveformID[P.publicID.str.match(aid)].item()['@locationCode']
    #     # retrieve station configuration
    #     C = pd.read_json(json.dumps(
    #         config.parameters[config.station.str.match(P.waveformID[P.publicID.str.match(aid)].item()['@stationCode']) &
    #                           config.network.str.match(P.waveformID[P.publicID.str.match(aid)].item()['@networkCode']) &
    #                           config.paramtyp.str.match('scautopick')].item()))
    #     # fill output table
    #     otab.loc[otab.shape[0]] = [
    #         evt.EventID, P.waveformID[pin1].item()['@networkCode'], P.waveformID[pin1].item()['@stationCode'],
    #         loc, P.waveformID[pin1].item()['@channelCode'], (A.pickID[ain2].item() if ain2 else ''),
    #         (datetime.strptime(P.time.values[pin2][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ') if pin2 else ''),
    #         (float(A.amplitude.values[ain2][0]['value']) if ain2 else ''), A.pickID[ain1].item(),
    #         datetime.strptime(P.time.values[pin1][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ'),
    #         float(A.amplitude.values[ain1][0]['value']), arr.time_residual,
    #         C.value[C.name.str.match('picker.AIC.filter')].item(),
    #         float(C.value[C.name.str.match('trigOn')].item()), float(C.value[C.name.str.match('trigOff')].item()),
    #         float(C.value[C.name.str.match('picker.AIC.minSNR')].item()),
    #         minNumPh, minAmpOffs, deadTime]
    # # calculate average pick SNR
    # avgsnr.append(otab[otab.EventID == evt.EventID].PickSNR.mean())
    # # calculate RMS of pick time residual
    # rmstres.append(np.sqrt(otab[otab.EventID == evt.EventID].PickTres.pow(2).sum()))
    # # calculate average number of picks per station
    # avgnpick.append(otab[otab.EventID == evt.EventID].groupby('Station').count().EventID.mean())
    # # write output table to file
    # otab.to_csv(f'{wdir}/auto-evt_17-09-2023.{exp}.csv', index=False)
    # # print summary
    # print(f'{len(ctab)} Cat events')
    # print(f'{len(ctab)-nm} True events')
    # print(f'{nf} False events')
    # print(f'{nm} Missed events')
    # print(f'Min True mag: {ctab.Mag[ctab.EventID.isin(otab.EventID)].min()}')
    # print(f'Max Missed mag: {ctab.Mag[~ctab.EventID.isin(otab.EventID)].max()}')
    # print(f'Avg Pick SNR: {sum(avgsnr)/len(avgsnr) if avgsnr else 0}')
    # print(f'RMS Pick Tres: {sum(rmstres)/len(rmstres) if rmstres else 0}')
    # print(f'Avg No. Picks/Sta: {sum(avgnpick)/len(avgnpick) if avgnpick else 0}')
