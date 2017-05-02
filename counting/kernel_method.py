import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from matplotlib import patches
import scipy.stats as stats
from matplotlib import gridspec
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import abell as abell

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

#import the data
hdulist = fits.open("spte_sva1_red.fits")
data = hdulist[1].data
# #galmask = data['MODEST_CLASS'] == 1
# #galdata = data[galmask]

magmask = data['MAG_AUTO_R'] < 24.4

RA = data['RA'][magmask]
DEC = data['DEC'][magmask]

galaxy_data = [RA, DEC]

p = 8 #2^p bins for Fourier transform
z = 0.5 #redshift

h = abell.kernel_bandwidth(galaxy_data, p, z)

density, xgr, ygr, xedges, yedges, bw_width, bw_height, binnums = abell.gaussian_estimator(galaxy_data, p, h)

minz = 2.8  #min density for cluster detection

clusters = abell.find_clusters(density, minz)

groups = abell.find_groups(clusters, 4)
groups_field = [np.array(list(map(list, (zip(xedges[group[:,0]],
                                             yedges[group[:,1]])))))
                for group in groups]
weights = [density[group[:,0], group[:,1]] for group in groups]
sigs = [np.mean(vals) for vals in weights]

#groups_avg = np.array([average_position(group, weight) for (group, weight) in
#                       list(map(list, zip(groups_field, weights)))])

rick = [abell.points_in_group(group, binnums) for group in groups]
#removing the nans by the following line means that the significances don't correspond to their physical positions
#rick = [y for y in rick if 0 not in y.shape]

groups_avg_pos = np.array([abell.average_position(morty) for morty in rick])

#these control the plotting to produce handsome latex plots
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True

#plotting the actual figure
fig = plt.figure(figsize=(12,6))
plt.suptitle('$z = {}$, $h = {:04f}$, $\sigma >{}$'.format(z, h,
                                                           minz))
#defining the ratio of the two subplots - the empirical width ratios
#just work to make the density map and scatter plot the same size
gs=gridspec.GridSpec(1, 2, width_ratios=[1,1.067])
#assign the axes to each subplot
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
#scatter plot
ax1.scatter(galaxy_data[0], galaxy_data[1], marker='.', s=0.5, color='cornflowerblue')
ax1.scatter(groups_avg_pos[:,0] - bw_width/2, groups_avg_pos[:,1] - bw_height/2,
            marker='x', s=80, c='r') 
ax1.set_aspect('equal')
ax1.set_xlabel('$\mathrm{RA\ (deg)}$')
ax1.set_ylabel('$\mathrm{Dec\ (deg)}$')
ax1.set_xlim([min(galaxy_data[0]), max(galaxy_data[0])])
ax1.set_ylim([min(galaxy_data[1]), max(galaxy_data[1])])
#density map
#cmap = plt.cm.jet for original
#cmap = plt.cm.YlOrRd for fyah
cb = ax2.imshow(density.transpose()[::-1], cmap = plt.cm.YlGnBu)
#grim stuff for colorbar
divider = make_axes_locatable(ax2)
cax1 = divider.append_axes("right", size="5%", pad=0.08)
cax2 = plt.colorbar(cb, cax = cax1)
cax2.set_label('$\sigma$')
plt.tight_layout()
#save and show
#plt.savefig('kernel_minz{}_z{}.png'.format(minz, z), dpi = 600, transparent = True)
plt.show()

#RA and Dec positions
RA_pos = groups_avg_pos[:,0] - bw_width/2
DEC_pos = groups_avg_pos[:,1] - bw_height/2
#sig = np.array([density[cluster[0], cluster[1]] for cluster in
#clusters])

#save cluster locations as a text file
np.savetxt('clusterlocs_minz{}_z{}_p{}.txt'.format(minz, z, p),
           np.transpose([RA_pos, DEC_pos,sigs]), fmt = '%4.8f',
           delimiter = ' ', header = 'RA (deg)  Dec (Deg)  Sigma')
#save density map as a text file
np.savetxt('kd_minz{}_z{}_p{}.txt'.format(minz, z, p), density,
           delimiter = ' ')
