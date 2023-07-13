---- Accomplished
The DMC potential needs to be included in xtrnlf.f (external force on regular integration step)
Normally the galactic force is calculated for each star position/velocity in galacocentric units,
  and then the force from the galacocentric orbit is substracted.
In this new case, the DMC force must be calculated for each star position/velocity in SC-centric units.

---- Accomplished
The reading of units (i.e. the DMC profile parameters) are read in via xtrnl0.f.
They must also have their units converted into Nbody units.

----
The dynamical friction mass must be updated, this is MASSCL.

----
The Nbody6 star cluster mass must be updated, this is ZMASS. See also ZMTOT = ZMASS*ZMBAR

---
How are stars considered unbound? Do I need to include the new mass profile there somehow?
