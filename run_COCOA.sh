#!/bin/bash

# Inputs:
#======================================================================
sim_name=TEST
directory=/user/HS301/m18366/EDGE_codes/GC_mock_obs/Nbody6_sims/$sim_name/
COCOA=/user/HS301/m18366/EDGE_codes/GC_mock_obs/COCOA/
#declare -A Bands
#declare -A Rows
Rows=("9" "10" "11")
#Rows=("11" "12" "13")
Bands=("U" "V" "B")
#======================================================================

# Function to replace lines in files:
update() {
    sed -i "$(grep -n "${2}" ${1} | cut -d: -f1)s|.*|${3}|" ${1}
}

function run_band ()
{

cd $COCOA
cp ${directory}/YBC_${sim_name}.dat ${COCOA}/YBC_${sim_name}.dat

# Clean the old output directories:
#----------------------------------------------------------------------
if [ -d param-files ]; then
  rm -r param-files
fi
if [ -d fits-files ]; then
  rm -r fits-files
fi
#----------------------------------------------------------------------

# Update parameter files:
#----------------------------------------------------------------------
# Update the param_rtt.py file:
update "param_rtt.py" "rtt=" "rtt=2.0"
update "param_rtt.py" "snapname=" "snapname='YBC_${sim_name}.dat'"

# Update the input_observation.py file:
# [Any other choices will need to be pre-set]
update "input_observation.py" "posx =" "posx = 1"
update "input_observation.py" "posy =" "posy = 2"
update "input_observation.py" "dmag =" "dmag = ${Rows[$1]}"
update "input_observation.py" "filt =" "filt = '${Bands[$1]}'"
#----------------------------------------------------------------------

# Run the observation script:
#----------------------------------------------------------------------
python2 observation.py
#----------------------------------------------------------------------

# Copy the fits files to a secure location:
#----------------------------------------------------------------------
mv param-files $directory/fits-files-${Bands[$1]}
mv fits-files $directory/fits-files-${Bands[$1]}
#----------------------------------------------------------------------

echo "#"    Produced fits files for Band ${Bands[$i]}
}

export -f run_band

# Cannot perform this job in parallel because the fits-files and param-files
# directories may be overwritten.
cycle=("0" "1" "2")
cycle=("0")
for i in "${cycle[@]}"
do
  echo Running COCOA/observation.py on $sim_name for Band ${Bands[$i]}
  run_band $i
done

rm ${COCOA}/YBC_${sim_name}.dat 
