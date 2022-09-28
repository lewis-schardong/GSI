from datetime import datetime
from datetime import timedelta
from obspy.clients.fdsn import Client

"2020-04-10T16:28:00.000"

tbeg = datetime.strptime("2020-04-10T16:28:00.000", '%Y-%m-%dT%H:%M:%S.%f')
tend = tbeg + timedelta(minutes=5)

client_isn = Client("https://earthquake.co.il")
inventory_isn = client_isn.get_stations(network="IS", level="response", channel="*")
inventory_isn.write('my_inventory.xml', format='STATIONXML')

evt_list = client_isn.get_events(starttime=tbeg, endtime=tend, includearrivals=True,
                                 minlatitude=29.0, maxlatitude=34.0, minlongitude=33.0, maxlongitude=37.0)
print(evt_list)
exit()

data = []
for ii in range(len(evt_list)):
    for kk in range(len(evt_list.events[ii].picks)):
        data.append([evt_list.events[ii].preferred_origin().time, evt_list.events[ii].picks[kk].waveform_id.station_code,
                     evt_list.events[ii].picks[kk].phase_hint[0], evt_list.events[ii].picks[kk].time])
