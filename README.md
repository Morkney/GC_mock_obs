    ________  __                ___     __  ___      __  __ 
   / ____/ /_/ /_  ____ _____  ( _ )   /  |/  /___ _/ /_/ /_
  / __/ / __/ __ \/ __ `/ __ \/ __ \/|/ /|_/ / __ `/ __/ __/
 / /___/ /_/ / / / /_/ / / / / /_/  </ /  / / /_/ / /_/ /_  
/_____/\__/_/ /_/\__,_/_/ /_/\____/\/_/  /_/\__,_/\__/\__/  

============================================================

Directory to handle Nbody6df simulations fit to EDGE ICs.

Things you will need:

Python2.7
EDGE1 python distribution

Astropy python module

ArtPop python module (the edited version that allows passing of Nbody6 data)

lmfit python module

agama python module

McLuster:
git clone https://github.com/ahwkuepper/mcluster.git
git checkout 8ef0880b21e88537b78cfa3789169d33e4151455
Located in the base directory

To compile Nbody6df:
git clone https://github.com/JamesAPetts/NBODY6df.git
cd NBODY6df/GPU2
change sm_20 to sm_30 in the makefile
mkdir Build
mkdir run
make gpu
make a synbolic link in GPU2 to Ncode/common6.h
Do the same for Ncode
in the lib directory: ln -s cnbint5.cpp cnbint.cpp
> The executable will be in the /run directory
