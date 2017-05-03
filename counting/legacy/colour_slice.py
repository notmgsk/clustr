import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits

#import the data
hdulist = fits.open("sva1_gold_r1.0_catalog.fits")

data = hdulist[1].data
galmask = data['MODEST_CLASS'] == 1

#mask out the galaxies with magnitude 99
magmask_G = (data['MAG_AUTO_G'] < 99)*galmask
magmask_R = (data['MAG_AUTO_R'] < 99)*galmask
#magmask_I = (data['MAG_AUTO_I'] < 99)*galmask
#magmask_Z = (data['MAG_AUTO_Z'] < 99)*galmask
            
magdata_G = data['MAG_AUTO_G']
magdata_R = data['MAG_AUTO_R'] 
#magdata_I = data['MAG_AUTO_I']
#magdata_Z = data['MAG_AUTO_Z']

z = 0.5

def colour_index(filter1, filter2, mask1, mask2):
    
    if len(mask1) < len(mask2):
        
        #first cut out the false magnitudes using the larger mask
        filter1 = filter1[mask1]
        filter2 = filter2[mask1]
        
    else:
        
        filter1 = filter1[mask2]
        filter2 = filter2[mask2]
        
    ci = (filter1 - filter2)
    
    mask = ci < 45
    
    ci = ci[mask]
    filter2 = filter2[mask]
    
    return ci, filter2

ci, filter2 = colour_index(magdata_G, magdata_R, magmask_G, magmask_R)

#only use these parameters on a uni computer
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['text.fontsize'] = 18
mpl.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

plt.figure(figsize = (8,6))
plt.plot(filter2, ci, 'k.', markersize =1)
plt.xlabel('$r$')
plt.ylabel('$g-r$')
plt.savefig('gr_colourmag_alldata.png', dpi = 600, transparent = True)