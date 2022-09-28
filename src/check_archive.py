# file manipulation
import os
from os import path
from pathlib import Path
import subprocess
import re
import stat
# maths & signal processing
import numpy as np

# archive directory
# adir = '/net/172.16.46.200/archive'
adir = '/net/172.16.41.48/backup/archive'

# maximum array dimensions allowed
if adir == '/net/172.16.46.200/archive' or adir == '/net/172.16.46.201/archive':
    yr = range(1979, 2023)
    nmax = 17
    smax = 247
    cmax = 78
elif adir == '/net/172.16.41.48/backup/archive':
    yr = [2000, range(2021, 2023)]
    nmax = 7
    smax = 150
    cmax = 78
else:
    nmax = 10
    smax = 100
    cmax = 10
dmax = 366

for i in yr:
    print(i)
exit()

# array allocations
n_ntw = np.empty((yr[1]-yr[0]+1, 1), dtype=int)
n_stn = np.empty((yr[1]-yr[0]+1, nmax+1), dtype=int)
n_chn = np.empty((yr[1]-yr[0]+1, nmax+1, smax+1), dtype=int)
n_day = np.empty((yr[1]-yr[0]+1, nmax+1, smax+1, cmax+1), dtype=int)
# index initialisation (most are to check we allocated enough space)
i = 0
n = 0
s = 0
c = 0
d = 0
spec_char = ['\@', '\!', '\#', '\$', '\%', '\^', '\&', '\*', '\{', '\}', '\[', '\]', '\<', '\>', '\?', '\:']
for y in range(yr[0], yr[1]+1):
    if not path.exists("{:s}/{:n}/".format(adir, y)):
        print(" Missing year: {:n}".format(y))
    else:
        print("Year {:n}:".format(y))
        out1 = subprocess.run(['ls', '{:s}/{:n}/'.format(adir, y)], stdout=subprocess.PIPE)
        res1 = out1.stdout.decode('utf-8')
        tab1 = re.findall("(.*?\n)", res1)
        n_ntw[i] = len(tab1)
        if n_ntw[i] > n:
            n = int(n_ntw[i])
        for j in range(n_ntw[i, 0]):
            ntw = re.sub('\n', '', tab1[j])
            # test for subdirectory that is not a network (typically same year subdirectory) (added exception for Z1 network)
            if (re.search("0", ntw) or re.search("1", ntw) or re.search("2", ntw)) and ntw != "Z1":
                print(" Subdirectory is not a network: {:s}/{:n}/{:s}".format(adir, y, ntw))
            # test for special characters in network name
            if re.findall(r"(?=("+'|'.join(spec_char)+r"))", ntw):
                print(" Subdirectory contains special character: {:s}/{:n}/{:s}".format(adir, y, ntw))
            # test for access to network directory
            st = os.stat("{:s}/{:n}/{:s}".format(adir, y, ntw))
            if not bool(st.st_mode & stat.S_IRWXG):
                print(" Network subdirectory is inaccessible: {:s}/{:n}/{:s}".format(adir, y, ntw))
                continue
            out2 = subprocess.run(['ls', '{:s}/{:n}/{:s}'.format(adir, y, ntw)], stdout=subprocess.PIPE)
            res2 = out2.stdout.decode('utf-8')
            tab2 = re.findall("(.*?\n)", res2)
            n_stn[i, j] = len(tab2)
            if n_stn[i, j] > s:
                s = n_stn[i, j]
            for k in range(n_stn[i, j]):
                stn = re.sub('\n', '', tab2[k])
                # test for special characters in station name
                if re.findall(r"(?=("+'|'.join(spec_char)+r"))", stn):
                    print(" Subdirectory contains special character: {:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                # test for unusual files (assuming station codes are <6 characters)
                if len(stn) > 5:
                    print(" Unusual file: {:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                    continue
                # test if channel directory is link instead of directory
                if Path("{:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn)).is_symlink():
                    print("  Station subdirectory is a link: {:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                    continue
                # test for access to station directory
                st = os.stat("{:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                if not bool(st.st_mode & stat.S_IRWXG):
                    print(" Station subdirectory is inaccessible: {:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                    continue
                # test for weird station name
                if re.search("xx", stn) or re.search("XX", stn):
                    print(" Unusual station name: {:s}/{:n}/{:s}/{:s}".format(adir, y, ntw, stn))
                out3 = subprocess.run(['ls', '{:s}/{:n}/{:s}/{:s}'.format(adir, y, ntw, stn)], stdout=subprocess.PIPE)
                res3 = out3.stdout.decode('utf-8')
                tab3 = re.findall("(.*?\n)", res3)
                n_chn[i, j, k] = len(tab3)
                if n_chn[i, j, k] > c:
                    c = n_chn[i, j, k]
                for ii in range(n_chn[i, j, k]):
                    chn = re.sub('\n', '', tab3[ii])
                    # test for special characters in channel name
                    if re.findall(r"(?=("+'|'.join(spec_char)+r"))", chn):
                        print(" Subdirectory contains special character: {:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                    # test for unusual files (assuming channel codes are <6 characters)
                    if len(chn) > 5:
                        print(" Unusual file: {:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                        continue
                    # test if channel directory is link instead of directory
                    if Path("{:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn)).is_symlink():
                        print("  Channel subdirectory is a link: {:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                        continue
                    # test for access to channel directory
                    st = os.stat("{:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                    if not bool(st.st_mode & stat.S_IRWXG):
                        print("  Channel subdirectory is inaccessible: {:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                        continue
                    out4 = subprocess.run(['ls', '{:s}/{:n}/{:s}/{:s}/{:s}'.format(adir, y, ntw, stn, chn)], stdout=subprocess.PIPE)
                    res4 = out4.stdout.decode('utf-8')
                    tab4 = re.findall("(.*?\n)", res4)
                    n_day[i, j, k, ii] = len(tab4)
                    # test for number of daily files (must be <366)
                    if n_day[i, j, k, ii] >= dmax:
                        print("  More daily files than days in a year: {:s}/{:n}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn))
                        break
                    if n_day[i, j, k, ii] > d:
                        d = n_day[i, j, k, ii]
                    for jj in range(n_day[i, j, k, ii]):
                        day = re.sub('\n', '', tab4[jj])
                        # test if channel directory is link instead of directory
                        if Path("{:s}/{:n}/{:s}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn, day)).is_symlink():
                            print("  Daily data file is a link: {:s}/{:n}/{:s}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn, day))
                            continue
                        # test for access to daily data file
                        st = os.stat("{:s}/{:n}/{:s}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn, day))
                        if not bool(st.st_mode & stat.S_IRWXG):
                            print("  Daily data file is inaccessible: {:s}/{:n}/{:s}/{:s}/{:s}/{:s}".format(adir, y, ntw, stn, chn, day))
                            continue
        i += 1

print("Maximum number of networks: {:n}".format(n))
print("Maximum number of stations: {:n}".format(s))
print("Maximum number of channels: {:n}".format(c))
print("Maximum number of days: {:n}".format(d))
