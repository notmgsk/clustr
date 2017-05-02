import numpy as np
import matplotlib
import matplotlib.pyplot as plt

length = 40000

RA = np.random.random_sample(length) + 68
DEC = np.random.random_sample(length) - 48
galaxy_data = [RA, DEC]
np.savetxt('simulation_field.txt', galaxy_data, delimiter = ' ')

#save the random data for use as a control field
#run the cluster finding algorithms on the field
#then add clusters at random locations
#subtract the known clusters from those found by the algorithms
#now from the remaining catalog we know completeness and false
#positives
#repeat n times
#repeat varying parameters such as redshift, p/bin width, bandwidth

plt.figure(figsize = (6,6))
plt.scatter(galaxy_data[0], galaxy_data[1], marker='.', s=0.5, color='cornflowerblue')
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.xlim([68,69])
plt.ylim([-48,-47])
plt.show()
