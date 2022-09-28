########################################################################################################################
# seismology
from obspy import read_inventory
from obspy.clients.fdsn import Client

ntw = ['IS', 'GE']
chn = ['HHZ', 'BHZ', 'ENZ', 'SHZ']
# directories
wdir = '/mnt/c/Users/lewiss/Documents/Research'
sdir = '/home/sysop/seiscomp/etc/key/'
# FDSN database
client_isn = Client('http://172.16.46.102:8181/')

# read station isn_inv
# isn_inv = read_inventory('%s/Autopicker/inventory_autop.xml' % wdir) # , format='STATIONXML')
fdsn_inv = read_inventory('%s/Autopicker/inventory.xml' % wdir, format='STATIONXML')

# loop over networks
n = 0
for nt in fdsn_inv.networks:
    # check if network requested
    if sum([True if x == nt.code else False for x in ntw]) == 0:
        continue
    # loop over stations
    for st in nt.stations:
        # print(st.code)
        kk = None
        cc = None
        ll = None
        # loop over requested channels
        for k in range(len(chn)):
            if kk is not None:
                break
            # print(' ' + chn[k])
            for ch in st.channels:
                if kk is None and ch.code == chn[k]:
                    # print('  ' + ch.code)
                    kk = k
                    cc = ch.code
                    ll = ch.location_code
                    break
        if kk is not None:
            print(' Preferred channel: %s.%s.%s.%s' % (nt.code, st.code, ll, cc))
        else:
            print('No valid channel for %s' % st.code)
        # write station binding file
        f = open('%s/station_%s_%s' % (sdir, nt.code, st.code), 'w')
        f.write('# Binding references\n')
        f.write('global:%s_%s\n' % (ll, cc[:-1]))
        f.write('scautopick:%s_%s\n' % (ll, cc[:-1]))
        f.close()
        n += 1
print('Wrote %i station binding files' % n)
