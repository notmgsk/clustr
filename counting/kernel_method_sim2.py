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
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import ICRS
from astropy.coordinates import SkyCoord
from astroML.crossmatch import crossmatch_angular
from abell import find_clusters
from abell import find_groups
from abell import average_position
from abell import indices_in_bin
from abell import points_in_bin
from abell import gaussian_estimator
from abell import kernel_bandwidth
from abell import get_clusters
from abell import add_cluster

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

#these control the plotting to produce handsome latex plots
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True

################################################################################
p = 8 #2^p bins for Fourier transform
z = 0.5 #redshift
minz = 4 #min density for cluster detection
linking = 2 #linking length between groups

#first find the clusters that occur naturally due to the distribution
#of the galaxies in the field
galaxy_data = list(np.loadtxt('data/simulation_field.txt', delimiter = ' '))
h, ang_cluster = kernel_bandwidth(galaxy_data, p, z)
natural_clusters = get_clusters(galaxy_data, p, z, h, minz, linking,
             'natural_clusters.txt')


N = 1 #number of runs
run = 0
n_matches = np.empty(N)
l_matches = np.empty(N)
fc = np.empty(N)

rad = 1

while run < N:
    #create the new galaxy field with placed clusters
    nclust = 10 #number of clusters to place in the field
    ng_min = 100 #minimum number of galaxies in a cluster
    ng_max = 1000 #maximu number of galaxies in a cluster
    ngal = np.random.randint(ng_min, ng_max, nclust) #richness of the clusters (array)
    cluster_locs = np.array([(np.random.random_sample(nclust) + 68),
                             (np.random.random_sample(nclust) - 48)]) #locations of
                                                            #the centre of
                                                            #each cluster
                                                            #(array of
                                                            #length
                                                            #2nclust)
    #add_cluster uses a normal distribution for the galaxies in the
    #cluster, this could probably be improved
    for i in range(0, nclust-1):
        add_cluster(galaxy_data, (cluster_locs[0][i], cluster_locs[1][i]),
                    (rad*ang_cluster), ngal[i])

    minz = 3.75

    found_clusters = get_clusters(galaxy_data, p, z, h, minz, linking,
                                  'found_clusters.txt')

    f_ra = [x[0] for x in found_clusters if str(x) != 'nan']
    f_dec = [x[1] for x in found_clusters if str(x) != 'nan']
    f_list = np.transpose([f_ra, f_dec])

    n_ra = [x[0] for x in natural_clusters]
    n_dec = [x[1] for x in natural_clusters]
    n_list = np.transpose([n_ra, n_dec])

    c_list = np.transpose(cluster_locs)

    tol = 0.1 #tolerance value for matches (degrees)

    l_dist, l_ind = crossmatch_angular(f_list, c_list, tol)
    l_matches[run] = len(l_dist[~np.isinf(l_dist)])

    n_dist, n_ind = crossmatch_angular(f_list, n_list, tol)
    n_matches[run] = len(n_dist[~np.isinf(n_dist)])

    fc[run] = len(found_clusters)

    print('done' + str(run+1))

    run += 1
    

nm_avg = np.mean(n_matches)
nm_std = np.std(n_matches)
lm_avg = np.mean(l_matches)
lm_std = np.std(l_matches)
fc_avg = np.mean(fc)
fc_std = np.std(fc)

avgs = np.array([nm_avg, nm_std, lm_avg, lm_std, fc_avg, fc_std])

#make sure to change .format() between rad and minz depending on which
#is being tested
# np.savetxt('avgs{}.txt'.format(minz), avgs, delimiter = ' ',
#            header = 'n_mu n_sig l_mu l_sig fc_mu fc_sig')

print('rly done')





