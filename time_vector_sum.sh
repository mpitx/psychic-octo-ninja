#!/bin/sh

for i in $(seq 0 3)
do
    /usr/bin/time python2 vector_sum.py 2> ~/Data/out/vsum_cpu_${i}
done
