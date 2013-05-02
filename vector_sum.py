#!/usr/bin/env python

import numpy as np
import time
from guppy import hpy

h = hpy()

N = 10 ** 8

a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

start = time.time()
c = a + b
end = time.time()

print("Time taken to add N-Vector to N-Vector: %s" % str(end - start))
print(h.heap())
