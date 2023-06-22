#!/bin/bash

echo restart_run$1_fort.2$2

mkdir run$1
mv fort.* run$1
mv OUT* run$1
mv err run$1
mv ESC run$1
mv out.* run$1
mv out run$1
mv COAL run$1
mv COLL run$1
mv ROCHE run$1

cp run$1/fort.2 fort.1

./restart.sh
