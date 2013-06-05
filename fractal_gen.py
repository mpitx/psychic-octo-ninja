#!/usr/bin/env python

import os
import sys
import shutil
import numpy as np
from tempfile import mkdtemp

I = 100
limits = (-2.0, 2.0)
size = 8192
N = size ** 2
irange = sum(abs(x) for x in limits)

def solve(z):
    def __bound(z):
        return abs(z) < 4

    n = z
    for i in range(I):
        if __bound(n):
            n = n * n + z
        else:
            return i / I
    return 1.0

def a(x):
    return limits[0] + (irange / size) * x

def b(y):
    return limits[1] - (irange / size) * y

tempdir = mkdtemp()
filename = os.path.join(tempdir, 'newimage.npy')
fp = np.memmap(filename, dtype='float32', mode='w+', shape=(N,))
for y in range(size):
    for x in range(size):
        fp[x + y * size] = solve(complex(a(x), b(y)))

def pixel(x):
    getP = lambda l,h: int(255 * (x - l) / (h - l))
    if 0 <= x < 0.15:
        p = getP(0, 0.15)
        return (0, p, p)
    elif 0.15 <= x < 0.30:
        p = getP(0.15, 0.30)
        return (0, p, 0)
    elif 0.30 <= x < 0.45:
        p = getP(0.30, 0.45)
        return (0, 0, p)
    elif 0.45 <= x < 0.60:
        p = getP(0.45, 0.60)
        return (p, 0, 0)
    elif 0.60 <= x < 0.75:
        p = getP(0.60, 0.75)
        return (p, p, 0)
    elif 0.75 <= x < 0.90:
        p = getP(0.75, 0.90)
        return (p, 0, p)
    elif 0.9 <= x < 1.0:
        p = getP(0.9, 1.0)
        return (p, p, p)
    return (0, 0, 0)

with open('fractal.ppm', 'w') as fimg:
    fimg.write('P3\n')
    fimg.write('#Big Fractal Image\n')
    fimg.write('%d %d\n' % (size, size))
    fimg.write('255\n')
    for x in fp:
        fimg.write('%d %d %d ' % pixel(x))

del fp
shutil.rmtree(tempdir)
