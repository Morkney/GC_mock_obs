#!/bin/bash
#./sim_size_profile_GC_r_v.sh 8k Halo1459GM3 10 0.01 0

size=$1
profile=$2
GC_size=$3
orbit=$4
velocity=$(printf "%.2f\n" $5)
if [[ $GC_size == "0" ]]; then
  GC_size="default"
fi

# Read in the profile parameters:
gamma=$(sed -n '1p' < ./ICs/$profile.txt)
Mg=$(sed -n '2p' < ./ICs/$profile.txt)
rs=$(sed -n '3p' < ./ICs/$profile.txt)
if [[ $profile == *"GM"* ]]; then
  rs_2=$(sed -n '4p' < ./ICs/$profile.txt)
fi

# Initialise the directory:
directory=./"$size"/"$profile"_"$velocity"_"$orbit"_"$GC_size"

# Kill program if the directory exists:
if [ -d "$directory" ]; then
  echo "$directory already exists"
  exit 1
fi

mkdir $directory

# Get the orbital speed:
vel=$(python circular_velocity.py $orbit $rs $Mg $gamma $velocity)

# Copy components into directory:
cp ./ICs/run.sh $directory/
cp ./ICs/restart.sh $directory/
cp ./ICs/restart_run_fort.sh $directory/
cp ./ICs/rs $directory/

# Copy the GC ICs into directory and find number of timesteps:
if [[ $GC_size == "default" ]]
then
  cp ./ICs/"$size".10 $directory/fort.10
  cp ./ICs/"$size".input $directory/cluster.input
  step=$(sed -n '1p' ./ICs/"$size".info)
  total=$(sed -n '2p' ./ICs/"$size".info)
else
  cp ./ICs/"$size"_"$GC_size"pc.10 $directory/fort.10
  cp ./ICs/"$size"_"$GC_size"pc.input $directory/cluster.input
  step=$(sed -n '1p' ./ICs/"$size"_"$GC_size"pc.info)
  total=$(sed -n '2p' ./ICs/"$size"_"$GC_size"pc.info)
fi

if [[ $profile == *"GM"* ]]
then
  echo 0.0 0.0 0.0 0.0 0.0 0.0 $Mg $rs $rs_2 $gamma >> $directory/cluster.input
  sed -i "7s|.*|../../nbody6df_morkney.gpu < cluster.input > out 2> err \&|" $directory/run.sh
  sed -i "7s|.*|../../nbody6df_morkney.gpu < rs > out 2> err \&|" $directory/restart.sh
else
  echo 0.0 0.0 0.0 0.0 0.0 0.0 $Mg $rs $gamma >> $directory/cluster.input
fi
echo 0.0 $orbit 0.0 $vel 0.0 0.0 >> $directory/cluster.input
echo 0.000000 0.000000 0.000000 0.000000 >> $directory/cluster.input

# Overwrite the timesteps in the input file:
line=($(sed -n '3p' < $directory/cluster.input))
line[4]=$step
line[5]=$total
line=$(echo ${line[*]// / })
sed -i "3s/.*/$line/" $directory/cluster.input

echo Built $directory
