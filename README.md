    ________  __                ___     __  ___      __  __ 
   / ____/ /_/ /_  ____ _____  ( _ )   /  |/  /___ _/ /_/ /_
  / __/ / __/ __ \/ __ `/ __ \/ __ \/|/ /|_/ / __ `/ __/ __/
 / /___/ /_/ / / / /_/ / / / / /_/  </ /  / / /_/ / /_/ /_  
/_____/\__/_/ /_/\__,_/_/ /_/\____/\/_/  /_/\__,_/\__/\__/  
                                                           

This directory includes scripts to automate mock GC image
production from EDGE simulation outputs.

Pipeline description:
1 - a) Retrieve the star cluster properties (Nstar, half-mass-rad, Z)
  - b) Retrieve the host galaxy properties (Dehnen profile potential fit)
  - c) Find the "birth" position and velocity of the star cluster (t, x/y/z, vx/vy/vz)

2 - a) Construct GC ICs using McLuster
  - b) Set up an Nbody6 run directory with the GC ICs and galactic host potential

3 - Run the GC Nbody6 simulation for set amount of time.

4 - a) Convert the Nbody6 output into a format readable by LBC: http://stev.oapd.inaf.it/YBC/
  - b) Run LBC on the data
  - c) Convert LBC into a format readable by COCOA

5 - Run COCOA on the data

6 - Convert the COCOA output fits files into a GC mock image



===============================================================

Things you will need:

Python2.7
EDGE1 python distribution

LBC:
http://stev.oapd.inaf.it/YBC/

COCOA:
https://github.com/abs2k12/COCOA.git

GSL:
wget "ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz"
tar -xf gsl-latest.tar.gz
cd gsl_folder
export GSL_DIR=$HOME/GSL
mkdir $GSL_DIR
./configure --prefix=$GSL_DIR && make && make install
# GSL
export GSL_DIR=$HOME/GSL
export PATH=$PATH:$GSL_DIR/bin
export C_INCLUDE_PATH=$GSL_DIR/include
export LIBRARY_PATH=$LIBRARY_PATH:$GSL_DIR/lib
export LD_LIBRARY_PATH="$GSL_DIR/lib:$LD_LIBRARY_PATH"

McLuster:
git clone https://github.com/ahwkuepper/mcluster.git
git checkout 8ef0880b21e88537b78cfa3789169d33e4151455
