import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

#why is this returning only one match even when the tolerance is
#greater than the data range??


#import redmapper cluster catalog
hdulist = fits.open("redmapper_red.fits")
data = hdulist[1].data
red_RA = data['RA']
red_DEC = data['DEC']
hdulist.close()

#convert redmapper catalog to SkyCoord for matching
red_cat = SkyCoord(red_RA*u.degree, red_DEC*u.degree)

#import algorithm cluster catalog
alg_data = np.loadtxt('data/clusterlocs_minz4.5_z0.5_p8.txt', delimiter = ' ')
alg_RA = [x[0] for x in alg_data]
alg_DEC = [x[1] for x in alg_data]
#convert to SkyCoord
alg_cat = SkyCoord(alg_RA*u.degree, alg_DEC*u.degree)

#match catalogs
n = 0
tol = Angle(0.1*u.degree)

for i in range(0, len(red_RA) - 1):
    ind, sep, _ = red_cat[i].match_to_catalog_sky(alg_cat,
                                                     nthneighbor = 1)
    n += int(sep < tol)

print(n)

plt.figure(figsize = (6,6))
plt.scatter(alg_RA, alg_DEC, s = 25, c = 'darkgreen')
plt.hold(True)
plt.scatter(red_RA, red_DEC, s = 25, c = 'r')
plt.show()
