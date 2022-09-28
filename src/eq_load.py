import os
import sys

evt = str(sys.argv[1])

# time window
whou = 0
wmin = 2
wsec = 0
# starting date & time
dat1 = evt[0:4] + '-' + evt[4:6] + '-' + evt[6:8] + ' ' + evt[8:10] + ':' + evt[10:12] + ':' + evt[12:14] + 'Z'
# ending date & time
dat2 = evt[0:4] + '-' + evt[4:6] + '-' + evt[6:8] + ' ' + str(int(evt[8:10])+whou) + ':' + str(int(evt[10:12])+wmin) + ':' + str(int(evt[12:14])+wsec) + 'Z'

print('Time window:', dat1, '~', dat2)

# import data
os.system('scart -dsEvt "' + dat1 + '~' + dat2 + '" /net/199.71.138.50/archive_all/ > data.mseed')
# repack data
os.system('/opt/passcal/bin/msrepack -R 512 -i -a -o r_data.mseed data.mseed')
# sort data
os.system('scmssort -uEv r_data.mseed > ' + evt + '.mseed')

# remove temporary files
os.system('rm r_data.mseed data.mseed')
