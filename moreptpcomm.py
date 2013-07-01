#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

if rank == 0:
    data = list([] for i in range(size - 1))
    print(data)
    for i in range(size - 1):
        to = i + 1
        data[i].append(list(x * to for x in range(4)))
        to_send = np.array(data[i])
        COMM.Send(to_send, dest=to, tag=to)
        print(to, 'sent')
else:
    data = np.empty((1, 4))
    COMM.Recv(data, source=0, tag=rank)
    print('recieved', data)
