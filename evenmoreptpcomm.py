#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import random

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

if rank == 0:
    for i in range(size - 1):
        to = i + 1
        l = random.randint(1, 10)
        data = np.ones(l)
        COMM.send(l, dest=to, tag=0)
        COMM.Send(data, dest=to, tag=to)
        print('sent', data, to)
else:
    count = COMM.recv(source=0, tag=0)
    data = np.zeros(count)
    COMM.Recv(data, source=0, tag=rank)
    print(rank, 'recieved', data)
