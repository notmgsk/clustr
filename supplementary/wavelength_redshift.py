import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib as mpl
from matplotlib import rc

#distance measures
lambda_emit = 400

n = 125 #number of redshift values to use
lambda_obs = np.zeros(n)
z = np.linspace(0,2,n)

for i in range(0,n):
    lambda_obs[i] = lambda_emit*(1 + z[i])

data = np.loadtxt('dec_filters.txt')

wl = data[:,0]
dec_g = data[:,2]
dec_r = data[:,3]
dec_i = data[:,4]
dec_z = data[:,5]

peak_g = wl[np.argmax(dec_g)]
peak_r = wl[np.argmax(dec_r)]
peak_i = wl[np.argmax(dec_i)]
peak_z = wl[np.argmax(dec_z)]

mpl.rc('lines', linewidth = 2)
plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['text.fontsize'] = 18
mpl.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

fig1 = plt.figure(1, figsize = (8,6))
ax = plt.gca()
plt.plot(z, lambda_obs, color = 'k')
plt.axhline(y = peak_g, color = '#A8383B', linestyle='-') #g
ax.text(1.8, peak_g + 10, '$g$', color = '#A8383B')
plt.axhline(y = peak_r, color = '#338A2E', linestyle='-') #r
ax.text(1.8, peak_r + 5, '$r$', color = '#338A2E')
plt.axhline(y = peak_i, color = '#226764', linestyle='-') #i
ax.text(1.8, peak_i + 5, '$i$', color = '#226764') 
plt.axhline(y = peak_z, color = '#AA6B39', linestyle='-') #z
ax.text(1.8, peak_z + 5, '$z$', color = '#AA6B39') 
plt.xlabel('$z$')
plt.ylabel('$\lambda_{\mathrm{obs}}\ \mathrm{[nm]}$')
#plt.savefig('wavelength_redshift.png', dpi = 600, transparent = True)
fig1.show()

fig2 = plt.figure(2, figsize = (8,6))
plt.plot(wl, dec_g, color = '#A8383B')
plt.hold(True)
plt.plot(wl, dec_r, color = '#338A2E')
plt.plot(wl, dec_i, color = '#226764')
plt.plot(wl, dec_z, color = '#AA6B39')
plt.ylim(0,100)
plt.xlabel('$\lambda \ [\mathrm{nm}]$')
plt.ylabel('Transmission (\%)')

fig2.show()
