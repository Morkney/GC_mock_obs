#!/bin/bash
export GPU_LIST="0"
export OMP_NUM_THREADS=8
rm -f out err OUT3* HIARCH ESC COLL COALL

../nbody6df_evolve.gpu < GC_IC.input > out 2> err &

date > run.log
echo $HOSTNAME, PID $! >> run.log
