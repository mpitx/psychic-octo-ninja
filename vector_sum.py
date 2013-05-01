#!/usr/bin/env python

import numpy as np
import time

N = 10 ** 8

a = np.random.randn(1, N)
b = np.random.randn(1, N)

start = time.time()
c = a + b
end = time.time()

print("Time taken to add N-Vector to N-Vector: %s" % str(end - start))
