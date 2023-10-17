#!/bin/bash

#for file in ./Nbody6_sims/files/Halo*; do
inputs=("./Nbody6_sims/files/Halo624_fiducial_hires_output_00030_27.txt")
for file in "${inputs[@]}"; do
  echo $file
  directory=${file%".txt"}
  directory=${directory/files\/}

  # Copy the new IC file into a fresh simulation directory:
  #directory=./Nbody6_sims/$sim_name

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
    mkdir $directory
  fi

  # Copy components into directory:
  cp ./sim_template/* $directory
  # Plummer profile, no segregation, Kroupa (2001) IMF, astrophysical output, stellar mass in range 0.08-100.
  MASS=$(sed -n 1p $file) # Mass of star cluster [Msol]
  HMR=$(sed -n 2p $file)  # Half-mass radius [pc]
  ZMET=$(sed -n 3p $file) # Metallicity [Zsol]
  TOFF=$(sed -n 4p $file) # Nbody time offset [Myrs]
  ./mcluster/mcluster -M $MASS -R $HMR -Z $ZMET -T 1 -P 0 -S 0 -C 0 -G 1 -f 1 -u 1 -m 0.1 -m 100 -o GC_IC
  mv ./GC_IC* $directory

  # Rename the GC_IC.fort.10 file to a default name:
  mv $directory/GC_IC.fort.10 $directory/fort.10

  galaxy="$(sed -n 5p $file) $TOFF"
  phase=$(sed -n 6p $file)
  sed -i "12s/.*/${galaxy}/" $directory/GC_IC.input
  sed -i "13s/.*/${phase}/" $directory/GC_IC.input

  # Integrate with tidal tail too:
  sed -i "6s/.*/0 -1 0 0 1 2 0 1 0 1/" $directory/GC_IC.input
  sed -i "7s/.*/0 0 0 2 1 0 0 2 0 3/" $directory/GC_IC.input

  echo Built $directory

done
