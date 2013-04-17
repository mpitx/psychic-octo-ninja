#!/usr/bin/env python

import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("I'm %d and have the following settings::" % rank)
print("\tPath: %s" % os.getenv('PATH'))
print("\t" + os.readlink('/proc/%d/exe' % os.getppid()))
