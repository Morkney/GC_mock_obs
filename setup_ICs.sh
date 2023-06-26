#!/bin/bash

#===================================#
# Name of the simulation:
sim_name=TEST

# Step 1: generate a McLuster output
MASS=1000 # Mass of star cluster [Msol]
HMR=1 # Half-mass radius [pc]
ZMET=0.0001 # Metallicity [Zsol}
# Position and velocity coordinates [pc, km/s]:
X=1
Y=2
Z=3
VX=4
VY=5
VZ=6
#===================================#

# Copy the new IC file into a fresh simulation directory:
directory=./Nbody6_sims/$sim_name

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
./mcluster/mcluster -M $MASS -R $HMR -Z $ZMET -X $X -X $Y -X $Z -V $VX -V $VY -V $VZ -T 1 -P 0 -S 0 -C 0 -G 1 -f 1 -u 1 -m 0.08 -m 100 -o GC_IC
mv ./GC_IC* ./Nbody6_sims/$sim_name

# Rename the GC_IC.fort.10 file to a default name:
mv ./Nbody6_sims/$sim_name/GC_IC.fort.10 ./Nbody6_sims/$sim_name/fort.10

# Overwrite the galaxy potential:
sed -i "12s/.*/0.0 0.0 0.0 0.0 0.0 0.0 479028865.500940 0.876989 0.00/" $directory/GC_IC.input

echo Built $directory
