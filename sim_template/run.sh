#!/bin/bash
rm -f out err OUT3* HIARCH ESC COLL COALL

export OMP_NUM_THREADS=8
export GPU_LIST="0"

../../nbody6df.gpu < cluster.input > out 2> err &

date > run.log
echo $HOSTNAME, PID $! >> run.log
