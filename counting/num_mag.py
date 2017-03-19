import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits

#import the data
hdulist = fits.open("sva1_gold_r1.0_catalog.fits")

data = hdulist[1].data[0::2000]
galmask = data['MODEST_CLASS'] == 1
magdata_G = data['MAG_AUTO_G'][galmask]
magdata_R = data['MAG_AUTO_R'][galmask]
magdata_I = data['MAG_AUTO_I'][galmask]
magdata_Z = data['MAG_AUTO_Z'][galmask]

plt.figure(1, figsize = (6,6), dpi = 200)
plt.subplot(2,2,1)
plt.hist(magdata_G, 100, normed = 1, facecolor = 'green')
plt.xlabel('Mag')
plt.ylabel('Fraction')
plt.title('$g$ band')
plt.subplot(2,2,2)
plt.hist(magdata_R, 100, normed = 1, facecolor = 'red')
plt.xlabel('Mag')
plt.ylabel('Fraction')
plt.title('$r$ band')
plt.subplot(2,2,3)
plt.hist(magdata_I, 100, normed = 1, facecolor = 'blue')
plt.xlabel('Mag')
plt.ylabel('Fraction')
plt.title('$i$ band')
plt.subplot(2,2,4)
plt.hist(magdata_Z, 100, normed = 1, facecolor = 'black')
plt.xlabel('Mag')
plt.ylabel('Fraction')
plt.title('$z$ band')

plt.tight_layout()
plt.show()
