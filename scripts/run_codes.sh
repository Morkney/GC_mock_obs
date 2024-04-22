#!/bin/bash

#-------------------------------------------------
# EDGE
# Halo1445_fiducial_hires: None
# Halo1459_fiducial_hires: 0-2
# Halo600_fiducial_hires:  0-1
# Halo605_fiducial_hires:  0-15
# Halo624_fiducial_hires:  0-4

# CHIMERA
# Halo383_fiducial_early:  0-24
# Halo383_Massive:         0-150
#-------------------------------------------------

#for input in {0..2}
#do
#  echo $input
#  python ./simulation_scripts/construct_GC_ICs.py Halo1459_fiducial_hires $input
#done

#for input in {0..1}
#do
#  echo $input
#  python ./simulation_scripts/construct_GC_ICs.py Halo600_fiducial_hires $input
#done

#for input in {0..15}
#do
#  echo $input
#  python ./simulation_scripts/construct_GC_ICs.py Halo605_fiducial_hires $input
#done

#for input in {0..4}
#do
#  echo $input
#  python ./simulation_scripts/construct_GC_ICs.py Halo624_fiducial_hires $input
#done

#for input in {0..24}
#do
#  echo $input
#  python ./simulation_scripts/construct_GC_ICs.py Halo383_fiducial_early $input
#done

for input in {0..150}
do
  echo $input
  python ./simulation_scripts/construct_GC_ICs.py Halo383_Massive $input
done

echo Finished.
