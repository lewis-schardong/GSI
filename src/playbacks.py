import pandas as pd
import os
import re
from datetime import datetime, timedelta
from obspy import read_events
import xmltodict
import json


# input parameters
networks = 'IS,GE'
channels = '(B|H|E|S)(H|N)Z'

# paths
wdir = '/mnt/c/Users/lewiss/Documents/Research/Autopicker/playback'
vdir = '/net/172.16.46.200/archive'
adir = '/net/172.16.46.122/mnt/archiveb'

# reading catalogue events
ctab = pd.read_csv(f'{wdir}/cat-evt_17-09-2023.csv', parse_dates=['DateTime'])\
    .rename(columns={'epiid': 'EventID', 'Depth(Km)': 'Depth'})
# remove quotation marks from event IDs and add 'gsi'
ctab.EventID = 'gsi' + ctab['EventID'].str.replace("'", "")

# read autopicker configuration
with open(f'{wdir}/scautopick.cfg') as f:
    txt = f.read()
aMaxTWin = float(txt[re.search('amplitudeMaxTimeWindow', txt).span()[1]:
                 re.search('amplitudeMaxTimeWindow', txt).span()[1]+15].replace(' ', '').replace('s', ''))
aMinOffs = float(txt[re.search('amplitudeMinOffset', txt).span()[1]:
                 re.search('amplitudeMinOffset', txt).span()[1]+20].replace(' ', '').replace('s', ''))
# autoloc configuration (no config dumping possible)
minNumPh = 5

# prepare output table
otab = pd.DataFrame({'EventID': pd.Series(dtype='string'), 'Station': pd.Series(dtype='string'),
                     'Network': pd.Series(dtype='string'), 'Location': pd.Series(dtype='string'),
                     'Channel': pd.Series(dtype='string'), 'TrigID': pd.Series(dtype='string'),
                     'TrigTime': pd.Series(dtype='datetime64[ms]'), 'TrigSNR': pd.Series(dtype='float64'),
                     'PickID': pd.Series(dtype='string'), 'PickTime': pd.Series(dtype='datetime64[ms]'),
                     'PickSNR': pd.Series(dtype='float64'), 'filter': pd.Series(dtype='string'),
                     'minSNR': pd.Series(dtype='float64'), 'TriggerOn': pd.Series(dtype='float64'),
                     'TriggerOff': pd.Series(dtype='float64'), 'minNSta': pd.Series(dtype='int64'),
                     'AmplMinOffset': pd.Series(dtype='float64'), 'AmplMaxTimeWin': pd.Series(dtype='float64')})

# loop over events
for _, evt in ctab.iterrows():
    # prepare waveforms if needed
    if os.path.exists(f'{wdir}/{evt.EventID}.mseed') == 0:
        # retrieve velocity waveforms
        if os.path.exists(f'{wdir}/{evt.EventID}.vel.mseed') == 0:
            os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{evt.DateTime-timedelta(minutes=5.)}~'
                      f'{evt.DateTime+timedelta(minutes=10.)}" {vdir} > {wdir}/{evt.EventID}.vel.mseed')
        # retrieve acceleration waveforms
        if os.path.exists(f'{wdir}/{evt.EventID}.acc.mseed') == 0:
            os.system(f'scart -dsE -n "{networks}" -c "{channels}" -t "{evt.DateTime-timedelta(minutes=5.)}~'
                      f'{evt.DateTime+timedelta(minutes=10.)}" {adir} > {wdir}/{evt.EventID}.acc.mseed')
        # merge the two mseed files
        os.system(f'cp -p {wdir}/{evt.EventID}.vel.mseed {wdir}/{evt.EventID}.mseed')
        os.system(f'cat {wdir}/{evt.EventID}.acc.mseed >> {wdir}/{evt.EventID}.mseed')
    # check the playback was ran
    if not os.path.exists(f'{wdir}/{evt.EventID}.xml'):
        print(f'Run playback for {evt.EventID}')
        exit()
    # read event xml file (not with ObsPy to get SNR values and corresponding pick IDs) and convert to dictionary
    with open(f'{wdir}/{evt.EventID}.xml') as f:
        data = xmltodict.parse(f.read())
    # read only relevant parts of dictionary
    catalog = data['seiscomp']['EventParameters']
    # convert to JSO > pd
    P = pd.read_json(json.dumps(catalog['pick'])).rename(columns={'@publicID': 'publicID'})
    A = pd.read_json(json.dumps(catalog['amplitude'])).rename(columns={'@publicID': 'publicID'})
    # read configuration xml file and convert to dictionary
    with open(f'{wdir}/config.xml') as f:
        data = xmltodict.parse(f.read())
    # read only relevant parts of dictionary
    cfg = data['seiscomp']['Config']['parameterSet']
    # convert to JSO > pd
    config = pd.read_json(json.dumps(cfg))
    config['network'] = config['@publicID'].apply(lambda x: x.split('/')[3])
    config['station'] = config['@publicID'].apply(lambda x: x.split('/')[4])
    config['paramtyp'] = config['@publicID'].apply(lambda x: x.split('/')[5])
    config['parameters'] = config.loc[config.paramtyp == 'scautopick'].apply(lambda x: x['parameter'], axis=1)
    # read xml file with ObsPy
    xml = read_events(f'{wdir}/{evt.EventID}.xml')
    print(f"{evt.EventID}: {len(xml.events)} {'events' if len(xml.events) > 1 else 'event'}")
    if len(xml.events) == 0:
        print(f"M{evt.Mag} {evt.Type} in {evt.Region} ({evt.EventID}): undetected ({len(xml.events)} {'events' if len(xml.events) > 1 else 'event'})")
        continue
    if evt.EventID == 'gsi202309171239':
        xevt = xml.events[1]
    else:
        xevt = xml.events[0]
    print(f"M{evt.Mag} {evt.Type} in {evt.Region} ({evt.EventID}): detected ({len(xml.events)} {'events' if len(xml.events) > 1 else 'event'})")
    print(f" {xevt.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '')}:"
          f" {xevt.preferred_origin().time} ({len(xevt.origins)} {'origins' if len(xevt.origins) > 1 else 'origin'})")
    # loop over arrivals
    for arr in xevt.preferred_origin().arrivals:
        # get arrival ID
        aid = arr.resource_id.id.replace('smi:org.gfz-potsdam.de/geofon/', '').split('_')[0]
        print(arr)
        continue
        # getting AIC pick and SNR
        ain1 = A.publicID.str.match(aid + '.snr')
        pin1 = P.publicID.str.match(aid)
        apid = A.pickID[ain1].item()
        atim = datetime.strptime(P.time.values[pin1][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ')
        asnr = A.amplitude.values[ain1][0]['value']
        print(f"  AIC: {apid}: {atim} ({asnr})")
        # looking for Trigger pick and SNR
        ain2 = A.pickID.str.match(aid.replace('-AIC', ''))
        # trying to find direct match
        if not A[ain2].empty:
            pin2 = P.publicID.str.match(aid.replace('-AIC', ''))
            tpid = A.pickID[ain1].item()
            ttim = datetime.strptime(P.time.values[pin2][0]['value'], '%Y-%m-%dT%H:%M:%S.%fZ')
            tsnr = A.amplitude.values[ain1][0]['value']
        else:
            # extract channel info
            chn = aid.replace('-AIC', '').split('-')[1]
            # look for Trigger closest to AIC picks
            tdif = [abs((atim-t).total_seconds()) for t in
                    pd.to_datetime(A[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')].
                                   pickID.str.split('-', n=1, expand=True)[0], format='%Y%m%d.%H%M%S.%f').to_list()]
            imin = tdif.index(min(tdif))
            tpid = A[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')].pickID.iloc[imin]
            ttim = datetime.strptime(
                P.time.values[P.publicID.str.contains(chn) & ~P.publicID.str.contains('AIC')][imin]['value'],
                '%Y-%m-%dT%H:%M:%S.%fZ')
            tsnr = A.amplitude.values[A.pickID.str.contains(chn) & ~A.pickID.str.contains('AIC')][imin]['value']
            ain2 = A[A.pickID.str.match(tpid)].index
            pin2 = P[P.publicID.str.match(tpid)].index
        print(f"  Trig: {tpid}: {ttim} ({tsnr})")
        print('')
        # check location exists
        if not P.waveformID[P.publicID.str.match(aid)].astype(str).str.contains('locationCode').item():
            loc = ''
        else:
            loc = P.waveformID[P.publicID.str.match(aid)].item()['@locationCode']
        # retrieve station configuration
        C = pd.read_json(json.dumps(
            config.parameters[config.station.str.match(P.waveformID[P.publicID.str.match(aid)].item()['@stationCode']) &
                              config.network.str.match(P.waveformID[P.publicID.str.match(aid)].item()['@networkCode']) &
                              config.paramtyp.str.match('scautopick')].item()))
        # fill output table
        otab.loc[otab.shape[0]] = [
            evt.EventID, P.waveformID[pin1].item()['@networkCode'], P.waveformID[pin1].item()['@stationCode'],
            loc, P.waveformID[pin1].item()['@channelCode'], A.pickID[ain2].item(),
            P.time.values[pin2][0]['value'], A.amplitude.values[ain2][0]['value'],
            A.pickID[ain1].item(), P.time.values[pin1][0]['value'], A.amplitude.values[ain1][0]['value'],
            C.value[C.name.str.match('picker.AIC.filter')].item(), C.value[C.name.str.match('picker.AIC.minSNR')].item(),
            C.value[C.name.str.match('trigOn')].item(), C.value[C.name.str.match('trigOff')].item(), minNumPh, aMinOffs, aMaxTWin]
    exit()
# write output table to file
otab.to_csv(f'{wdir}/auto-evt_17-09-2023.csv')

