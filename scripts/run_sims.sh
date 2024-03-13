#!/bin/bash

count=0

# Loop all available simulations:
for file in ../Nbody6_sims/Halo*; do

  # Skip if there is already a simulation:
  if [ ! -d "$file/run1" ]; then

    # If the directory is new, start the sim from the beginning:
    #--------------------------------------------------------
    if [ ! -f "$file/out" ]; then
      cd $file
      ./run.sh
      echo $file
      cd -
    #--------------------------------------------------------

    # If the first output has been made, adjust timestepping and restart:
    #--------------------------------------------------------
    elif [ ! -f "$file/initiated" ]; then
      # Get the new timestepping:
      params=$(python update_timestepping.py $file)

      # Skip if output hasn't run for long enough:
      if [ -z "${params}" ]; then
        echo $file
        echo ">    First output not finished yet."
        continue
      fi

      # Replace timestepping parameters in input file:
      cd $file
      sed -i "s/1.00000000\s1.00000000\s1.00000000/1.00000000 ${params}/g" GC_IC.input
      ./run.sh
      echo $file
      touch initiated
      cd -
    fi
    #--------------------------------------------------------

  else
    continue
  fi

  # End if I have started off four new jobs already:
  #--------------------------------------------------------
  count=$(( count+1 ))
  if (( $count > 3 )) ; then
    break
  fi
  #--------------------------------------------------------

done
