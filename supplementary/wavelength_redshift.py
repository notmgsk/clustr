import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib as mpl

#distance measures
lambda_emit = 4000

n = 125 #number of redshift values to use
lambda_obs = np.zeros(n)
z = np.linspace(0,2,n)

for i in range(0,n):
    lambda_obs[i] = lambda_emit*(1 + z[i])

mpl.rc('font', family='sans-serif')
mpl.rc('lines', linewidth = 2)
plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['text.fontsize'] = 18

plt.figure(figsize = (8,6))
ax = plt.gca()
plt.plot(z, lambda_obs, color = 'k')
plt.axhline(y = 4770, color = '#A8383B', linestyle='-') #g
ax.text(1.8, 4900, '$g$', color = '#A8383B')
plt.axhline(y = 6231, color = '#338A2E', linestyle='-') #r
ax.text(1.8, 6311, '$r$', color = '#338A2E')
plt.axhline(y = 7625, color = '#226764', linestyle='-') #i
ax.text(1.8, 7705, '$i$', color = '#226764') 
plt.axhline(y = 9134, color = '#AA6B39', linestyle='-') #z
ax.text(1.8, 9184, '$z$', color = '#AA6B39') 
plt.xlabel(r'$z$')
plt.ylabel(r'$\lambda_{\mathrm{obs}}\ \mathrm{[\AA]}$')
plt.savefig('wavelength_redshift.png', dpi = 600, transparent = True)