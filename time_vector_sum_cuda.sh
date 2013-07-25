#!/bin/sh

for i in $(seq 0 3)
do
    echo "Run ${i}"
    /usr/bin/time python2 vector_sum_cuda.py 2> ~/Data/out/vsum_cuda_${i}
done
