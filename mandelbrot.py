#!/usr/bin/env python

from mpi4py import MPI
import Image
import colorsys
from math import ceil

w = 600
h = 600
its = 80
d2 = 4.0
xmax = 1.5
xmin = -2.5
ymax = 2.0
ymin = 2.0

def step(z,c):
    return z**2+c

def point(c,n,d2):
    zo = 0.0
    zn = zo
    i = 0
    while abs(zn)**2 < d2 and i < n:
        zn = step(zo,c)
        zo = zn
        i = i +1
    return i

def colnorm(r,g,b):
    return (int(255*r)-1, int(255*g)-1, int(255*b)-1)

def col(n,max):
    if n == max:
        return (0,0,0)
    return colnorm(colorsys.hsv_to_rgb(1.0-float(n)/max,1.0,1.0))

def row(n,xmin,xmax,ymin,ymax):
    row = []
    for x in range(w):
        p = complex((xmin+x*(xmax-xmin)/w),(ymin+n*(ymax-ymin)/h))
        row.append(point(p,its,d2))
    return row

def __main__():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    rows = [ rank + size*i for i in range(int(float(h)/size)+1) if rank + size*i < h]

    pixels = []
    for y in rows:
        pixels.append(row(y,xmin,xmax,ymin,ymax))

    mandel = comm.Gather(pixels)

    if MPI.rank == 0:
        img = Image.new("RGB", (w, h), (0,0,0))
        rows = []

        for i in range(len(mandel[0])):
            for j in range(len(mandel)):
                r = mandel[j][i]
                rows.append([col(p,its) for p in r])

        for x in range(w):
            for y in range(h):
                r = rows[y]
                c = r[x]
                img.putpixel((x,y),c)

        img.save("mandel.png")

__main__()
