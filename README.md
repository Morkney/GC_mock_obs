    ________  __                ___     __  ___      __  __ 
   / ____/ /_/ /_  ____ _____  ( _ )   /  |/  /___ _/ /_/ /_
  / __/ / __/ __ \/ __ `/ __ \/ __ \/|/ /|_/ / __ `/ __/ __/
 / /___/ /_/ / / / /_/ / / / / /_/  </ /  / / /_/ / /_/ /_  
/_____/\__/_/ /_/\__,_/_/ /_/\____/\/_/  /_/\__,_/\__/\__/  

============================================================                                                           

This directory includes scripts to automate mock GC image
production from EDGE simulation outputs.

Pipeline description:
1 - a) Retrieve the star cluster properties (Nstar, half-mass-rad, Z)
  - b) Retrieve the host galaxy properties (Dehnen profile potential fit)
  - c) Find the "birth" position and velocity of the star cluster (t, x/y/z, vx/vy/vz)
    - This is complicated, need to arrange this with Ethan.

2 - a) Construct GC ICs using McLuster
  - b) Set up an Nbody6 run directory with the GC ICs and galactic host potential
    - Prime and activate the "setup_ICs.sh" script for both a) and b)

3 - Run the GC Nbody6 simulation for set amount of time.
    - Needs to be done manually. Restart instructions are included in the simulation directories.

4 - Produce visual using the ArtPop package: https://artpop.readthedocs.io/en/latest/index.html
  - a) Prepare the Nbody6 data snapshot (i.e. find the right snapshot, centre correctly, etc.)
  - b) Pass the Nbody6 mass and x,y coordinates into ArtPop.
  - c) Plot the ArtPop using the desired filters and observational parameters.

Done!

===============================================================

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
