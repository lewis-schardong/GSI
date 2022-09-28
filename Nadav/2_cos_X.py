""" created by tgoebel, Feb.  21 2017
- Create and plot cos( x)

licence: ?? etc.

"""

import matplotlib as mpl  # all modules and fct in package matplotlib
import numpy as np # vector and matrix operations, and much more …
from matplotlib.pyplot import pyplot # basci plotting fcts., lines, axes, text, etc.
import pylab as plb # import pyplot and numpy to behave similar to matlab

# object oriented plotting
aX        = numpy.arange( 0, 2*numpy .pi, .1)
aCosX,aSinX = numpy.cos( aX), numpy.sin( aX)

plb.figure(1)
ax   = plb.subplot( 211)
ax2 = plb.subplot( 212)
ax.plot( aX/numpy.pi, aCosX*180/numpy.pi, ‘k-’)
ax.set_title( ‘cos’)
ax2 = plb.subplot( 212)
ax.plot( aX/numpy.pi, aSinX*180/numpy.pi, ‘k-’)
ax2.set_title( ‘sin’)
ax2.set_xlabel( ‘PI’)
plb.show()

