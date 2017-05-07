import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
import matplotlib as mpl
from matplotlib import rc

#import the data
hdulist = fits.open("redmapper_sva1_public_v6.3_catalog.fits")

data = hdulist[1].data
RA = data['RA']
DEC = data['DEC']
hdulist.close()

#these must correspond to the indices of the spt-e survey since no other data points lie in this RA range
red_arg = np.argwhere((RA > 68) & (RA < 69) & (DEC < -47) & (DEC > -48))

#spte_ra = [np.array(galaxy_data[0][arg]) for arg in spte_arg]
#spte_ra = np.array([item for sublist in spte_ra for item in sublist])
#spte_dec = [np.array(galaxy_data[1][arg]) for arg in spte_arg]
#spte_dec = np.array([item for sublist in spte_dec for item in sublist])
#spte_data = [spte_ra, spte_dec]

red_ra = data['RA'][red_arg]
red_dec = data['DEC'][red_arg]
# magdata_G = data['MAG_AUTO_G'][spte_arg]
# magdata_R = data['MAG_AUTO_R'][spte_arg]
# magdata_I = data['MAG_AUTO_I'][spte_arg]
# magdata_Z = data['MAG_AUTO_Z'][spte_arg]


#save the data in a new, condensed fits file

tbhdu = fits.BinTableHDU.from_columns( [fits.Column(name='RA',
                                                    format='E',
                                                    array=red_ra),
                                        fits.Column(name='DEC',
                                                    format='E',
                                                    array=red_dec)])
tbhdu.writeto('redmapper_red.fits')



plt.figure(figsize = (6,6))
plt.scatter(red_ra, red_dec, marker = '.', s=5, c = 'cornflowerblue')
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
#plt.savefig('redmapper.png', dpi=600, transparent = True)
plt.show()
