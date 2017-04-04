import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
import matplotlib as mpl
from matplotlib import rc

#import the data
hdulist = fits.open("sva1_gold_r1.0_catalog.fits")

data = hdulist[1].data
galmask = data['MODEST_CLASS'] == 1
galdata = data[galmask]

RA = galdata['RA']
DEC = galdata['DEC']

galaxy_data = [RA, DEC]

#these must correspond to the indices of the spt-e survey since no other data points lie in this RA range
spte_arg = np.argwhere((galaxy_data[0] < 99) & (galaxy_data[0] > 60))

#spte_ra = [np.array(galaxy_data[0][arg]) for arg in spte_arg]
#spte_ra = np.array([item for sublist in spte_ra for item in sublist])
#spte_dec = [np.array(galaxy_data[1][arg]) for arg in spte_arg]
#spte_dec = np.array([item for sublist in spte_dec for item in sublist])
#spte_data = [spte_ra, spte_dec]

spte_ra = data['RA'][spte_arg]
spte_dec = data['DEC'][spte_arg]
magdata_G = data['MAG_AUTO_G'][spte_arg]
magdata_R = data['MAG_AUTO_R'][spte_arg]
magdata_I = data['MAG_AUTO_I'][spte_arg]
magdata_Z = data['MAG_AUTO_Z'][spte_arg]


#save the data in a new, condensed fits file


tbhdu = fits.BinTableHDU.from_columns( [fits.Column(name='RA', format='E', array=spte_ra),         fits.Column(name='DEC', format='E', array=spte_dec), fits.Column(name = 'MAG_AUTO_G', format = 'E', array = magdata_G), fits.Column(name = 'MAG_AUTO_R', format = 'E', array = magdata_R), fits.Column(name = 'MAG_AUTO_I', format = 'E', array = magdata_I), fits.Column(name = 'MAG_AUTO_Z', format = 'E', array = magdata_Z), fits.Column(name = 'INDEX', format = 'D', array = spte_arg)])
tbhdu.writeto('spte_sva1.fits')
#save the indices just in case we need them later
np.savetxt('spte_indices.txt', spte_arg, '%d')

mpl.rc('lines', linewidth = 2)
plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['font.size'] = 18
mpl.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

plt.figure()
plt.scatter(spte_data[0], spte_data[1], marker = '.', s=2)
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.savefig('spte_footprint.png', dpi=600, transparent = True)