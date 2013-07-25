#!/bin/sh

for i in $(seq 0 3)
do
    echo "Run ${i}"
    /usr/bin/time mpi_run.py /usr/bin/time python2 \
     vector_sum_cudampi.py 2> ~/Data/out/vsum_cudampi_${i}
done
