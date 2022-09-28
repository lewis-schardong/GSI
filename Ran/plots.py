#/usr/bin/env python
import sys, os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from adjustText import adjust_text
sys.path.extend(['/Users/ran/work/TRUAA/blindzone/TRUAA'])
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TRUAA.settings")
from log2db import *
m2r = pd.read_csv('m2r_5.txt', skiprows=[0, 1], names=['M', 'R'], sep=',')
sources = ['b-jer', 'b-lod', 'r-jer', 'r-lod']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_borders():
    border = pd.read_csv('israelborder.csv',names=['lon','lat','elev'], skiprows=[0])
    gborder = gpd.GeoDataFrame(geometry=[Polygon(border.values)])
    gborder = gborder.set_crs("epsg:4326")
    return gborder

def get_alert_radius(mag):
    if pd.isna(mag): return None
    M = round(mag,1)
    R = m2r.loc[m2r.M == M].R.values[0]
    return R


def alert_circle(data):
    # data should be a geopandas WGS84 points with Rkm field as radius in km.
    dataBuff = data.to_crs("epsg:3857")
    buffer = dataBuff.geometry.buffer(dataBuff.Rkm*1000)
    dataBuff.geometry = buffer
    dataBuff = dataBuff.to_crs("epsg:4326")
    return dataBuff

def get_alerts_max(eqs, source='b-lod'):
    header = ['eventid','ver','time','evlat','evlon','mag','alert_time','source']
    evts = EVENT.objects.filter(firstalert=True, source__contains=source)
    closest = eqs.DateTime.apply(lambda ot: get_closest_to_ot(evts, ot))
    data = pd.DataFrame(columns=header)
    for ev in closest:
        data.loc[data.shape[0]] = EVENT.objects.filter(Ast=True, eventid=ev.eventid, source__contains=source).order_by('-mag').values(*header).first() if ev else None
    data = gpd.GeoDataFrame(data, geometry=data.apply(lambda x: Point(x.evlon, x.evlat) if x.evlon else None,axis=1))
    data = data.set_crs("epsg:4326")
    return data

def get_alerts_first(eqs, source='b-lod'):
    header = ['eventid','ver','time','evlat','evlon','mag','alert_time','source']
    evts = EVENT.objects.filter(firstalert=True, source__contains=source)
    closest = eqs.DateTime.apply(lambda ot: get_closest_to_ot(evts, ot))
    data = pd.DataFrame(columns=header)
    for ev in closest:
        data.loc[data.shape[0]] = EVENT.objects.filter(Ast=True, eventid=ev.eventid, source__contains=source).order_by('alert_time').values(*header).first() if ev else None
    data = gpd.GeoDataFrame(data, geometry=data.apply(lambda x: Point(x.evlon, x.evlat) if x.evlon else None,axis=1))
    data = data.set_crs("epsg:4326")
    return data

def get_alerts_first_TRUAA(eqs, source='b-lod', mag=4.5):
    header = ['eventid','ver','time','evlat','evlon','mag','alert_time','source']
    evts = EVENT.objects.filter(firstalert=True, source__contains=source)
    closest = eqs.DateTime.apply(lambda ot: get_closest_to_ot(evts, ot))
    data = pd.DataFrame(columns=header)
    for ev in closest:
        data.loc[data.shape[0]] = EVENT.objects.filter(Ast=True, mag__gte=mag, eventid=ev.eventid, source__contains=source).order_by('alert_time').values(*header).first() if ev else None
    data = gpd.GeoDataFrame(data, geometry=data.apply(lambda x: Point(x.evlon, x.evlat) if x.evlon else None,axis=1))
    data = data.set_crs("epsg:4326")
    return data


def get_alerts_TRUAA(source='b-lod', mag=4.5):
    header = ['eventid','ver','time','evlat','evlon','mag','alert_time','source']
    evts = EVENT.objects.filter(Ast=True, source__contains=source, mag__gte=mag).values('eventid').distinct()
    data = pd.DataFrame(columns=header)
    for ev in evts:
        data.loc[data.shape[0]] = EVENT.objects.filter(Ast=True, mag__gte=mag, eventid=ev['eventid'], source__contains=source).order_by('alert_time').values(*header).first()
    data = gpd.GeoDataFrame(data, geometry=data.apply(lambda x: Point(x.evlon, x.evlat) if x.evlon else None, axis=1))
    data = data.set_crs("epsg:4326")
    return data


def get_eq_fdsn(eq, fdsn='https://seis.gsi.gov.il'):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client(fdsn)
    ot = UTCDateTime(eq.time)
    cat = client.read_events(ot-120, ot+120)


def get_catalog(filename = 'felt_20220101-20220801.csv'):
    data = pd.read_csv(filename)
    data.epiid = data.epiid.apply(lambda x: x[1:-1])
    data.DateTime = data.DateTime.apply(lambda x: pd.to_datetime(x+'Z'))
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Long, data.Lat))
    data = data.set_crs("epsg:4326")
    data.sort_values('DateTime', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    return data

def fig2(save=False):
    ax = plt.gca()
    land = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAND')
    land.plot(ax=ax, edgecolor='0.5', facecolor='None')
    lake = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAKE')
    lake.plot(ax=ax, edgecolor='0.5', facecolor='blue', alpha = 0.5)
    gborder = get_borders()
    gborder.plot(ax=ax, edgecolor='0.5', facecolor='None', label='Operation Area')
    eqs = get_catalog()
    eqs.plot(ax=ax, color='r', markersize=5, label='Earthquakes')
    x0, y0, x1, y1 = eqs.total_bounds
    dx = (x1-x0)*0.1
    dy = (y1-y0)*0.1
    ax.set_xlim(x0-dx, x1+dx)
    ax.set_ylim(y0-dy, y1+dy)
    t = eqs.apply(lambda x: ax.text(x.Long, x.Lat, x.name), axis=1).values
    adjust_text(t)
    if save:
        plt.savefig('felt_events.png')
        felts = eqs[['DateTime', 'Mw', 'Region','Lat','Long']]
        felts.DateTime = felts.DateTime.apply(lambda x: x.tz_localize(None))
        felts.to_excel('felt.xlsx')


def fig3(save=False):
    ax = plt.gca()
    land = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAND')
    land.plot(ax=ax, edgecolor='0.5', facecolor='None')
    lake = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAKE')
    lake.plot(ax=ax, edgecolor='0.5', facecolor='blue', alpha = 0.5)
    gborder = get_borders()
    gborder.plot(ax=ax, edgecolor='0.5', facecolor='None', label='Operation Area')
    eqs = get_catalog()
    datas = []
    for source in sources:
        datas.append(get_alerts_max(eqs, source=source))
    data = pd.concat(datas).sort_values('mag', ascending=False).groupby(level=0).first()
    data = data.set_crs("epsg:4326")
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    data.loc[data.mag >= 4.2].plot(ax=ax, color='k', markersize=5, label='Earthquakes')
    x0, y0, x1, y1 = data.total_bounds
    dx = (x1-x0)*0.4
    dy = (y1-y0)*0.2
    ax.set_xlim(x0-dx, x1+dx)
    ax.set_ylim(y0-dy, y1+dy)
    t = data.loc[data.mag >= 4.2].apply(lambda x: ax.text(x.evlon, x.evlat, x.name), axis=1).values
    adjust_text(t)
    data['Rkm'] = data.mag.apply(get_alert_radius)
    dataBuff = alert_circle(data)
    dataBuff.loc[dataBuff.mag >= 4.5].plot(ax=ax, edgecolor='k', lw=2, facecolor='None', label='Alert Zone')
    data.loc[data.mag >= 4.5].plot(ax=ax, color='r', markersize=6, label='Earthquakes')
    datas = []
    for source in sources:
        datas.append(get_alerts_first(eqs, source=source))
    data1 = pd.concat(datas).sort_values('alert_time').groupby(level=0).first()
    data1 = data1.set_crs("epsg:4326")
    data1.reset_index(drop=True, inplace=True)
    data1.index = data1.index + 1
    datas = []
    for source in sources:
        datas.append(get_alerts_first_TRUAA(eqs, source=source))
    data2 = pd.concat(datas).sort_values('alert_time').groupby(level=0).first()
    data2 = data2.set_crs("epsg:4326")
    data2.reset_index(drop=True, inplace=True)
    data2.index = data2.index + 1
    #dates = []
    #for source in sources:
    #    datas.append(get_alerts_TRUAA(source=source,mag=4.2))

    #data3 = data3.loc[data3.apply(lambda x: x.eventid not in data.eventid.values, axis=1)]
    #data3.plot(ax=ax, color='k', markersize=5, label='Earthquakes')

    if save:
        plt.savefig('felt_max_alerts.png')
        mxalerts = eqs[['DateTime', 'Mw', 'Region', 'Lat', 'Long']]
        mxalerts['M_epic_first'] = data1.mag.values
        mxalerts['M_epic_max'] = data.mag.values
        mxalerts['first_alert_dt'] = [abs(data1.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data1.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts['TRUAA_alert_dt'] = [abs(data2.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data2.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts['max_alert_dt'] = [abs(data.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts.DateTime = mxalerts.DateTime.apply(lambda x: x.tz_localize(None))
        mxalerts.to_excel('epic.xlsx')

def fig4(save=False, mag=4.5):
    datas = []
    for source in sources:
        datas.append(get_alerts_TRUAA(source=source, mag=mag))
    data = pd.concat(datas).sort_values('alert_time')
    data = data.assign(**{'time1': data.time.round('30min')})
    data = data.groupby('time1').first()
    data = data.loc[data.time >= pd.to_datetime(['2021-01-01T00:00:00+00:00'])[0]]
    data = data.set_crs("epsg:4326")
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    # plot
    ax = plt.gca()
    land = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAND')
    land.plot(ax=ax, edgecolor='0.5', facecolor='None')
    lake = gpd.read_file('/Users/ran/work/TRUAA/blindzone/STATUS/LAKE')
    lake.plot(ax=ax, edgecolor='0.5', facecolor='blue', alpha=0.5)
    gborder = get_borders()
    gborder.plot(ax=ax, edgecolor='0.5', facecolor='None', label='Operation Area')
    data.loc[data.mag >= 4.2].plot(ax=ax, color='k', markersize=5, label='Earthquakes')
    x0, y0, x1, y1 = data.total_bounds
    dx = (x1-x0)*0.4
    dy = (y1-y0)*0.2
    ax.set_xlim(x0-dx, x1+dx)
    ax.set_ylim(y0-dy, y1+dy)
    t = data.loc[data.mag >= 4.2].apply(lambda x: ax.text(x.evlon, x.evlat, x.name), axis=1).values
    adjust_text(t)
    data['Rkm'] = data.mag.apply(get_alert_radius)
    dataBuff = alert_circle(data)
    dataBuff.loc[dataBuff.mag >= 4.2].plot(ax=ax, edgecolor='0.5', lw=2, facecolor='None', label='M4.2 Alert Zone')
    dataBuff.loc[dataBuff.mag >= 4.5].plot(ax=ax, edgecolor='k', lw=2, facecolor='None', label='M4.5 Alert Zone')
    data.loc[data.mag >= 4.5].plot(ax=ax, color='r', markersize=6, label='Earthquakes')
    if save:
        plt.savefig(f'M{mag}_alerts.png')
        mxalerts = eqs[['DateTime', 'Mw', 'Region', 'Lat', 'Long']]
        mxalerts['M_epic_first'] = data1.mag.values
        mxalerts['M_epic_max'] = data.mag.values
        mxalerts['first_alert_dt'] = [abs(data1.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data1.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts['TRUAA_alert_dt'] = [abs(data2.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data2.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts['max_alert_dt'] = [abs(data.loc[i].alert_time - eqs.loc[i].DateTime).total_seconds() if not pd.isna(data.loc[i].alert_time) else None for i in range(1, eqs.shape[0]+1)]
        mxalerts.DateTime = mxalerts.DateTime.apply(lambda x: x.tz_localize(None))
        mxalerts.to_excel('epic.xlsx')
