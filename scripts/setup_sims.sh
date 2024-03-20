#!/bin/bash

GPU=0
suite=$(python config.py)

# Uncomment for single mode:
#inputs=("../Nbody6_sims/files/Halo383_Massive_output_00014_123.txt")
#for file in "${inputs[@]}"; do

# List mode:
#--------------------------------------------------------------------------
for file in ../Nbody6_sims/${suite}_files/Halo*; do
  echo $file
  directory=${file%".txt"}
  directory=${directory/_files/}

  # Sanity checks:
  #--------------------------------------------------------------------------
  # Kill program if the metadata file doesn't exist:
  if [ ! -f "$file" ]; then
    echo "$file doesn't exist"
    continue
  fi

  # Kill program if the directory already exists:
  if [ -d "$directory" ]; then
    echo "$directory already exists"
    continue
  else
    mkdir -p $directory
  fi
  #--------------------------------------------------------------------------

  # Build the simulation directory:
  #--------------------------------------------------------------------------
  # Copy components into directory:
  cp ../sim_template/* $directory

  # Define the GPU number:
  sed -i "2s/.*/export GPU_LIST=\"${GPU}\"/" $directory/run.sh
  sed -i "2s/.*/export GPU_LIST=\"${GPU}\"/" $directory/restart.sh

  # Plummer profile, no segregation, Kroupa (2001) IMF, astrophysical output, stellar mass in range 0.08-100.
  SIM=$(sed -n 1p $file)  # Host profile name
  MASS=$(sed -n 2p $file) # Mass of star cluster [Msol]
  HMR=$(sed -n 3p $file)  # Half-mass radius [pc]
  ZMET=$(sed -n 4p $file) # Metallicity [Zsol]
  TOFF=$(sed -n 5p $file) # Nbody time offset [Myrs]
  BIN=$(sed -n 6p $file)  # Initial binary star fraction
  ../mcluster/mcluster -M $MASS -R $HMR -Z $ZMET -b $BIN -p 1 -T 1 -P 0 -S 0 -C 0 -G 1 -f 1 -u 1 -m 0.1 -m 100 -o GC_IC
  mv ./GC_IC* $directory

  # Rename the GC_IC.fort.10 file to a default name:
  mv $directory/GC_IC.fort.10 $directory/fort.10

  galaxy="$(sed -n 7p $file)$TOFF"
  phase=$(sed -n 8p $file)
  sed -i "12s/.*/${galaxy}/" $directory/GC_IC.input
  sed -i "13s/.*/${phase}/" $directory/GC_IC.input

  # Integrate with tidal tail too:
  sed -i "6s/.*/0 -1 0 0 1 2 0 1 0 1/" $directory/GC_IC.input
  sed -i "7s/.*/0 0 0 2 1 0 0 2 0 3/" $directory/GC_IC.input

  # Copy the profile fits to the simulation directory:
  cp ./files/$SIM.txt $directory/Host.txt

  echo Built $directory
  #--------------------------------------------------------------------------

  # Update GPU number iteratively, from between 0 and 3:
  #--------------------------------------------------------------------------
  # Reset GPU:
  GPU=$(( GPU+1 ))
  if [ $GPU -eq 4 ]; then GPU=0; fi
  #--------------------------------------------------------------------------

done

