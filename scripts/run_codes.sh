#!/bin/bash

#-------------------------------------------------
#inputs=("1" "2" "3" "4" "5")
#-------------------------------------------------

#for input in "${inputs[@]}"
for input in {1..157}
do
  echo $input
  python ./simulation_scripts/GC_prepare_sim.py $input
done

echo Finished.
