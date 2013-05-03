#!/usr/bin/env python

import os
import numpy as np
import time
from guppy import hpy

h = hpy()

a_filename = os.path.abspath('../../a.npy')
b_filename = os.path.abspath('../../b.npy')
c_filename = os.path.abspath('../../c.npy')
a = np.load(a_filename, mmap_mode='r')
b = np.load(b_filename, mmap_mode='r')
c = np.memmap(c_filename, dtype=np.float32, shape=a.shape, mode='w+')

N = len(a)

start = time.time()
c = a + b
end = time.time()

print("Time taken to add %d-Vector to %d-Vector: %s" % (N, N, str(end-start)))
print(h.heap())
