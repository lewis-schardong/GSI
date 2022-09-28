# -*- coding: utf-8 -*-
"""
- Vector multiplication and dot product
[input] [output]
v  -   numpy vector (1x3)
[license]
"""

from numpy import *
v = arange(3)
print v
print (v**2).sum()
print dot(v, v)
print dot(v,v.T)
print dot(v,v.reshape(3,1))
print type( dot(v,v.reshape(3,1)))
print type( dot(v,v.T))