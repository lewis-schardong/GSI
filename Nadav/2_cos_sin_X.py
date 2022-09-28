""" created by tgoebel, Feb.  21 2017
- Create and plot cos( x)

licence: ?? etc.

"""

import matplotlib as mpl  # all modules and fct in package matplotlib
import numpy as np # vector and matrix operations, and much more ...
import matplotlib.pyplot as plt # basci plotting fcts., lines, axes, text, etc.
import pylab as plb # import pyplot and numpy to behave similar to matlab

# object oriented plotting
aX = np.arange(0, 2*np.pi, .1)
aCosX,aSinX = np.cos( aX), np.sin( aX)

plb.figure(1)
ax   = plb.subplot( 211)
ax2 = plb.subplot( 212)
ax.plot( aX/np.pi, aCosX, 'k-')
ax.set_title( 'cos')
ax2 = plb.subplot( 212)
ax2.plot( aX/np.pi, aSinX, 'k-')
ax2.set_title( 'sin')
ax2.set_xlabel( 'PI')
plb.show()

