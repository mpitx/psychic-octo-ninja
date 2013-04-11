#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = np.random.randn(4, 5).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)
a_doubled = (2 * a_gpu).get()
print('From %d of %d:' % (rank, comm.Get_size()))
print(a_gpu)
print(a_doubled)
