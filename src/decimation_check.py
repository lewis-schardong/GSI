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

########################################################################################################################
# input parameters
ntw = 'IS'
chn = '(H|E)(H|N)Z'

# working directory
wdir = '/mnt/c/Users/lewiss/Documents/Research/Data'
print(f'Working directory: {wdir}\n')
mpl.rcParams['savefig.directory'] = f"{wdir}"
# data archive directory
adir = '/net/172.16.46.200/archive'

# FDSN database
isn_client = Client('http://172.16.46.102:8181/')

# retrieve station inventory
if path.exists(f"{wdir}/inventory.xml") != 0:
    print('ISN inventory file already exists:')
    os.system(f"ls -lh {wdir}/inventory.xml")
    print()
else:
    isn_inv = isn_client.get_stations(network=ntw, channel='ENZ, HHZ', level='response')
    isn_inv.write(f'{wdir}/inventory.xml', level='response', format='STATIONXML')
# read station inventory
isn_inv = read_inventory(f"{wdir}/inventory.xml", format='STATIONXML')

ori_time = datetime.strptime('2021-06-15 23:08:54', '%Y-%m-%d %H:%M:%S')
tbeg = str(datetime.strftime(ori_time - timedelta(minutes=5), '%Y-%m-%d %H:%M:%S'))
tend = str(datetime.strftime(ori_time + timedelta(minutes=5), '%Y-%m-%d %H:%M:%S'))
os.system(f'scart -dsE -n "{ntw}" -c "{chn}" -t "{tbeg}~{tend}" /net/172.16.46.200/archive > {wdir}/202106152308.ori.mseed')
os.system(f'scart -dsE -n "{ntw}" -c "{chn}" -t "{tbeg}~{tend}" /net/172.16.46.104/usr/local/autoloc/input > {wdir}/202106152308.dec.mseed')
