import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib as mpl

#distance measures
Om = 0.308 #matter density parameter
Ol = 0.692
Dh = 3e5/(67.8) #hubble distance in Mpc

n = 125 #number of redshift values to use
Da = np.zeros(n)
z = np.linspace(0,2,n)

for i in range(0,n):
    Da[i] = (Dh/(1+z[i]))*(integrate.quad(lambda x: 1/np.sqrt(Om*(1+z[i])**3 +
                                                    Ol), 0, z[i])[0])

mpl.rc('font', family='sans-serif')
mpl.rc('lines', linewidth = 2)
plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20

plt.figure(figsize = (8,6))
plt.plot(z, Da, color = 'k')
plt.xlabel('$z$')
plt.ylabel('$D_a\ \mathrm{[Mpc/rad]}$')
plt.savefig('angdiam_redshift.png', dpi = 600, transparent = True)
