#!/usr/bin/env python

import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

N = 10 ** 8

a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.empty_like(a)

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
        func(a_gpu, b_gpu, c_gpu, block=(22, 22, 1))
        cuda.memcpy_dtoh(slice_c, c_gpu)
        return slice_c

    c = np.empty_like(a)
    max = 2 ** 26
    M = int((N + max)/max)
    for i in range(0, M):
        slice_low = i * max
        slice_high = (i + 1) * max
        c = np.append(c, add(a[slice_low:slice_high],
                             b[slice_low:slice_high]))

    return c

start = time.time()
vector_add()
end = time.time()
print("Time taken to add %d-Vector to %d-Vector: %s" % (N, N, str(end-start)))
