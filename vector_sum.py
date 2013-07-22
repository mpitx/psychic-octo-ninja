#!/usr/bin/env python

import os
import numpy as np
import time

data_in_dir = os.path.join(os.environ['HOME'], 'Data', 'in', 'vsum')
tmp_data_dir = os.path.join(os.environ['HOME'], 'tmp', 'vsum')
a_filename = os.path.join(data_in_dir, 'a.npy')
b_filename = os.path.join(data_in_dir, 'b.npy')
c_filename = os.path.join(tmp_data_dir, 'c.npy')
a = np.load(a_filename, mmap_mode='r')
b = np.load(b_filename, mmap_mode='r')
c = np.memmap(c_filename, dtype=np.float32, shape=a.shape, mode='w+')

N = len(a)

start = time.time()
c = a + b
end = time.time()

print("Time taken to add %d-Vector to %d-Vector: %s" % (N, N, str(end-start)))
