import os
import sys

evt = str(sys.argv[1])

# trace view
os.system('scrttv -u lewis1 &')
# origin locator view
os.system('scolv -u lewis2 &')
# delete old data
os.system('rm -rf /home/sysop/seiscomp3/var/lib/seedlink/buffer')
# playback
os.system('seiscomp start seedlink')
os.system('seiscomp restart')
# speed test
os.system('msrtsimul -v ' + evt + '.mseed')
