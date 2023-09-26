#!/bin/bash

#===================================#
EDGE_sim_name=Halo624_fiducial_hires
EDGE_output=output_00017
GC_ID=1
sim_name=GC_${GC_ID}
file=./Nbody6_sims/files/${sim_name}.txt
#===================================#

# Copy the new IC file into a fresh simulation directory:
directory=./Nbody6_sims/$sim_name

# Kill program if the metadata file doesn't exist:
if [ ! -f "$file" ]; then
  echo "$file doesn't exist"
  exit 1
fi

# Kill program if the directory already exists:
if [ -d "$directory" ]; then
  echo "$directory already exists"
  exit 1
else
  mkdir $directory
fi

# Copy components into directory:
cp ./sim_template/* $directory
# Plummer profile, no segregation, Kroupa (2001) IMF, astrophysical output, stellar mass in range 0.08-100.
MASS=$(sed -n 1p $file) # Mass of star cluster [Msol]
HMR=$(sed -n 2p $file)  # Half-mass radius [pc]
ZMET=$(sed -n 3p $file) # Metallicity [Zsol]
T_NB=$(sed -n 4p $file) # Nbody time unit [Myrs]
./mcluster/mcluster -M $MASS -R $HMR -Z $ZMET -T 1 -P 0 -S 0 -C 0 -G 1 -f 1 -u 1 -m 0.1 -m 100 -o GC_IC
mv ./GC_IC* ./Nbody6_sims/$sim_name

# Rename the GC_IC.fort.10 file to a default name:
mv ./Nbody6_sims/$sim_name/GC_IC.fort.10 ./Nbody6_sims/$sim_name/fort.10

# Overwrite the galaxy potential and GC orbit:
#line_counter=12
#while read -r line; do
#  sed -i "${line_counter}s/.*/${line}/" $directory/GC_IC.input
#  line_counter=$(($line_counter + 1))
#done < $file
#sed -i "12s/.*/0.0 0.0 0.0 0.0 0.0 0.0 479028865.500940 0.876989 0.00/" $directory/GC_IC.input

phase=$(sed -n 5p $file)
galaxy=$(sed -n 6p $file)
sed -i "12s/.*/${phase}/" $directory/GC_IC.input
sed -i "13s/.*/${galaxy}/" $directory/GC_IC.input

# Integrate with tidal tail too:
sed -i "6s/.*/0 -1 0 0 1 2 0 1 0 1/" $directory/GC_IC.input
sed -i "7s/.*/0 0 0 2 1 0 0 2 0 3/" $directory/GC_IC.input

echo Built $directory
