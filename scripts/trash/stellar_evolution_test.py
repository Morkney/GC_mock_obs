import numpy as np
import stellar_evolution_functions as StellarEvolution
import stellar_devolution_functions as StellarDevolution
import importlib
import matplotlib.pyplot as plt
plt.ion()

param = StellarEvolution.Parameters('EDGE1')
star = StellarEvolution.Stars()
star.add_stars(500,0.02)

fig = plt.figure()
plt.xlabel('Time')
plt.ylabel('Mass')

for i in range(1000):
    star.evolve(1,param)
    plt.plot(star.age[0],star.mass[0],'.k')
plt.show()

param = StellarDevolution.Parameters('EDGE1')
star2 = StellarDevolution.Stars()
star2.add_stars(star.mass[0],star.metal[0],star.age[0])

for i in range(1000):
    star2.evolve(1,param)
    plt.plot(star2.age[0],star2.mass[0],'rx')

# It is the same!
