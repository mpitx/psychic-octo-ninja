#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = np.random.randn(4,4)
a = a.astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void map(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= %d;
    }
    """ % rank)

func = mod.get_function("map")

func(a_gpu, block=(4, 4, 1))
a_mapped = np.empty_like(a)
cuda.memcpy_dtoh(a_mapped, a_gpu)
print("From process %d:" % rank)
print(a_mapped)
print(a)
