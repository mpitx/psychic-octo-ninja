#!/usr/bin/env python

import os
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a_file = os.path.join(os.environ['HOME'], 'a.npy')
b_file = os.path.join(os.environ['HOME'], 'b.npy')
c_file = os.path.join(os.environ['HOME'], 'slice-%d.npy' % rank)
a = np.load(a_file, mmap_mode='r')
b = np.load(b_file, mmap_mode='r')
N = len(a)
M = int((N + size - 1) / size)
def vector_add():
    def add(slice_a, slice_b):
       slice_c = np.empty_like(slice_a)
       start = time.time()
       slice_c = slice_a + slice_b
       end = time.time()
       return (slice_c, end - start)

    c = np.memmap(c_file, dtype=np.float32, shape=(M,), mode='w+')
    slice_low = M * rank
    slice_high = M * (rank + 1)
    #print("%d: %d(%d/%d)" % (rank, M, slice_low, slice_high))
    result = add(a[slice_low:slice_high], b[slice_low:slice_high])
    np.append(c, result[0])

    return (c, result[1])
    #return c

total_start = time.time()
node_calc_time = vector_add()[1]
#vector_add()
total_end = time.time()
print("Time taken to add %d-Vector to %d-Vector: %s" %
    (M, M, str(total_end-total_start)))
