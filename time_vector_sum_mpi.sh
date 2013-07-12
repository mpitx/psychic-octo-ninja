#!/bin/sh

for i in $(seq 0 3)
do
    /usr/bin/time python2 vector_sum.py 2> vector_sum_mpi_results${i}
done
