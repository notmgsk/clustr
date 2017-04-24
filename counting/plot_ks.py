import numpy as np
import matplotlib.pyplot as plt

minz = [5, 4, 3, 2.75]
vals = []

#load the data
for i in minz:
    vals.append(np.loadtxt('data/minz_sim/avgs{}.txt'.format(i), delimiter = ' ', skiprows = 1))

n_found = np.array([x[4] for x in vals])
n_found_s = np.array([x[5] for x in vals])

n_match = np.array([x[2] for x in vals])
n_match_s = np.array([x[3] for x in vals])

n_natural = np.array([x[0] for x in vals])
n_natural_s = np.array([x[1] for x in vals])

plt.figure()
#plt.step(minz, (n_match)/(n_found))
plt.plot(minz, (n_match)/(n_found-n_natural))
plt.xlim([2.75,5])
plt.ylim([0,1])
plt.xlabel('$\sigma$')
plt.ylabel('$N_{m}/N_{f}$')
plt.show()
    
