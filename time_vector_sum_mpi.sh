#!/bin/sh

for i in $(seq 0 3)
do
    /usr/bin/time python2 vector_sum_mpi.py 2> ~/Data/out/vsum_mpi_${i}
done
