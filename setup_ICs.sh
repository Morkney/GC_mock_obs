#!/bin/bash

sim_name=TEST

# Step 1: generate a McLuster output
MASS=1000 # Mass of star cluster
HMR=5 # Half-mass radius
ZMET=0.0001 # Metallicity
# Position and velocity coordinates:
X=0
Y=0
Z=0
VX=0
VY=0
VZ=0

# Copy the new IC file into a fresh simulation directory:
directory=./Nbody6_sims/$sim_name

# Kill program if the directory exists:
if [ -d "$directory" ]; then
  echo "$directory already exists"
  exit 1
fi

mkdir $directory

# Copy components into directory:
cp ./sim_template $directory
./mcluster/mcluster -M $MASS -R $HMR -Z $ZMET -X $X -X $Y -X $Z -V $VX -V $VY -V $VZ -T $T -P 0 -S 0 -C 0 -G 1 -f 1 -u 1 -o GC_IC
cp ./GC_IC* ././Nbody6_sims/$sim_name

# Overwrite the galaxy potential:
update_template() {
    sed -i "$(grep -n "${1}" $directory/GC_IC.input | cut -d: -f1)s|.*|${2}|" $directory/GC_IC.input
}
blah blah

# Make sure the KZ(J) options are correct

echo Built $directory
