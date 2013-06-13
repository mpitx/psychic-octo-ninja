#!/usr/bin/env python

import os
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import time

MAX_THREADS = \
    cuda.Device(0) \
        .get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
BLOCK_SIZE = int(math.sqrt(MAX_THREADS))

a_filename = os.path.join(os.environ['HOME'], 'a.npy')
b_filename = os.path.join(os.environ['HOME'], 'b.npy')
c_filename = os.path.join(os.environ['HOME'], 'c.npy')
a = np.load(a_filename, mmap_mode='r')
b = np.load(b_filename, mmap_mode='r')
N = len(a)

mod = SourceModule('''
    __global__ void vector_sum(float *a, float *b, float *c)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        c[idx] = a[idx] + b[idx];
    }
    ''')

def vector_add():
    func = mod.get_function('vector_sum')
    def add(slice_a, slice_b):
        slice_c = np.empty_like(slice_a)
        a_gpu = cuda.mem_alloc(slice_a.nbytes)
        cuda.memcpy_htod(a_gpu, slice_a)
        b_gpu = cuda.mem_alloc(slice_b.nbytes)
        cuda.memcpy_htod(b_gpu, slice_b)
        c_gpu = cuda.mem_alloc(slice_c.nbytes)
        start = time.time()
        func(a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE, BLOCK_SIZE, 1))
        end = time.time()
        cuda.memcpy_dtoh(slice_c, c_gpu)
        return (slice_c, end-start)

    c = np.memmap(c_filename, dtype=np.float32, shape=a.shape, mode='w+')
    total_time = 0
    max = 2 ** 24
    M = int((N + max - 1)/max)
    for i in range(0, M):
        slice_low = i * max
        slice_high = (i + 1) * max
        result = add(a[slice_low:slice_high], b[slice_low:slice_high])
        np.append(c, result[0])
        total_time = total_time + result[1]

    return (c, total_time)

total_start = time.time()
total_time = vector_add()[1]
total_end = time.time()
print("Time taken to add %d-Vector to %d-Vector: %s" %
    (N, N, str(total_end-total_start)))
print("\tTotal Computational Time: %s" % str(total_time))
