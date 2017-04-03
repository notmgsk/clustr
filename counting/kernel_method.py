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

def gaussian_estimator(galaxy_data,p,h):
    """ 
    estimates the density function using the FFT method
    the raw data is finely discretised into 2^p bins
    the fourier transform of the raw data is taken
    a gaussian kernel is generated
    the fourier transform of the gaussian kernel is taken
    the kernel and the data are convolved by multiplying their fourier
    transforms
    an inverse fourier transform is performed on the data to yield the
    density map

    data: raw data to be analysed
    p: number of bins to discretise the data into (2^p) bins
    h: window width (smoothing parameter)
        
    """
    
    #need to account for the fact that the bin widths won't be equal in height
    #and width
    data_width = max(galaxy_data[0]) - min(galaxy_data[0])
    data_height = max(galaxy_data[1]) - min(galaxy_data[1])
    data_ratio = data_height/data_width
    
    M = 2**p #number of bins (width)
    
    n = np.size(galaxy_data)
    length = len(galaxy_data)
    a = 0 #limits
    b = length
    
    bw_width = (b-a)/M #the width of each discrete cell
    bw_height = bw_width*data_ratio #the height of each discrete cell
           
    #data_discrete, xedges, yedges = np.histogram2d(galaxy_data[0], galaxy_data[1], bins = M)
    
    # xedges and yedges are the corners of the bins used by np.histogram2d. Useful
    # for when we need to convert from "bin" coordinates into "real" coordinates.
    data_discrete, xedges, yedges, binnums = stats.binned_statistic_2d(galaxy_data[0],
                                                       galaxy_data[1],
                                                       values=None,
                                                       statistic='count',
                                                       bins=M,
                                                       expand_binnumbers=True)
    

    data_fourier = np.fft.fft2(data_discrete) #fourier transform the
    #discrete data

    #create the gaussian kernel
    x = np.linspace(0, M, M)
    y = np.linspace(0, M, M)
    xgr, ygr = np.meshgrid(x,y)
    kernel = np.exp(-0.5*(xgr**2 + ygr**2)/h)/np.sqrt(2*np.pi)

    kernel_fourier = np.fft.fft2(kernel) #take the fourier transform
    #of the gaussian kernel

    density_fourier = kernel_fourier*data_fourier #convolve the kernel
    #and the data

    density = np.real(np.fft.ifft2(density_fourier)) #perform the inverse
    #fourier transform to get the density in real space
    
    mu = np.mean(density)
    
    sigma = np.std(density)
    
    density_normed = (density - mu)/sigma
                     
    binnums -= 1
    
    return density_normed, xgr, ygr, xedges, yedges, bw_width, bw_height, binnums

def find_clusters(data, cond):
    """Returns the coordinates of overdensities (i.e. where cond is
    satisfied)"""
    # For now, data is expected to be data_normed. It might be more useful to
    # have the input data be the galaxy field, and then this particular
    # function would call count_in_cells, and return the relevent coordinates
    # pointing into the galaxy field, rather than into the density map (i.e.
    # data_normed). I say this would be more useful because, currently, after
    # calling this function, we have to manually convert the coordinates from
    # data_normed coords into galaxy field coords.
    return np.argwhere(data > cond)

def kernel_bandwidth(galaxy_data, p, z):
    """Finds the appropriate bandwidth for the kernel using the angular
    diameter size at a given redshift z.

    Dependent upon the power of two used for the Fourier bins."""

    import scipy.integrate as integrate
    
    RA_range = max(galaxy_data[0]) - min(galaxy_data[0])
    DEC_range = max(galaxy_data[1]) - min(galaxy_data[1])
    ang_range = 0.5*(RA_range + DEC_range)

    #distance measures
    Om = 0.308 #matter density parameter
    Ol = 0.692
    Dh = 3e5/(67.8) #hubble distance in Mpc

    Da = (Dh/(1+z))*(integrate.quad(lambda x: 1/np.sqrt(Om*(1+z)**3 +
                                                        Ol), 0, z)[0])
    #angular diameter distance as a function of redshit l/theta [Mpc/rad]

    ang_cluster = (1/Da)*(180/np.pi) #typical cluster radius 1 Mpc, yields typical
    #angular size in degrees
    
    p = 9
    
    ang_bins = ang_range/2**p #angle per bin

    h = ang_cluster/ang_bins #the window width for the gaussian estimator
    #should be the number of bins that a cluster would typically
    #occupy

    return h


def find_groups(data, r=1):
    """Returns the cluster groups; i.e., if there are neighbouring
    overdensities from the galaxy histogram, they are considered to be part of
    the same cluster (group).

    Note: data should be the overdensities in the histogram picked out by find_clusters."""
    def dist(v1, v2):
        return (data[v1][0] - data[v2][0])**2 + (data[v1][1] - data[v2][1])**2
    
    # Vertices are unique labels for each point in data
    verts = np.arange(len(data))
    # The problem now is to construct edges between vertices. An edge between two vertices
    # exists if the distance between the two corresponding points <= r.
    # These edges include duplicates but that's ok... doesn't affect the end goal.
    edges = [[v1, v2] for v1 in verts for v2 in verts if dist(v1, v2) <= r]
    # Groups is a list of sets, each set containing the connected vertices of
    # that group.
    groups = list(map(list, nx.connected_components(nx.Graph(edges))))
    # Now we want to go back from vertices to data points.
    # So, where we started with something like
    #    [A, B, C, D, E]    (with A, B, etc. of shape (1,2))
    # we return something like
    #    [[A, B, C], [D], [E]].
    return np.array([data[group] for group in groups])

def average_position(points):
    #totw = np.sum(weights)
    #avgx = np.sum([x * w for (x,w) in zip(group[:,0], weights)])/totw
    #avgy = np.sum([y * w for (y,w) in zip(group[:,1], weights)])/totw
    
    if points.shape == 0:
        avg_pos = []
    
    else:
        avg_pos = [np.sum(points[:,0])/len(points),
            np.sum(points[:,1])/len(points)]
    
    return avg_pos
    
def indices_in_bin(binn, binnums):
    """Returns the indices of elements of binnums for which the bin is equal to
    binn""" 
    return np.argwhere([(row == binn).all() for row in binnums.T]).ravel()

def points_in_bin(binn, binnums):
    """Returns an array of [RA, DEC] points that are within binn"""
    global galaxy_data
    indices = indices_in_bin(binn, binnums)

    return np.column_stack((galaxy_data[0][indices], galaxy_data[1][indices]))

def points_in_group(group, binnums):
    """Returns an array of [RA, DEC] points that are within group's bins"""
    return np.concatenate(([points_in_bin(binn, binnums) for binn in
                            group]))

################################################################################

#import the data
hdulist = fits.open("sva1_gold_r1.0_catalog.fits")

data = hdulist[1].data[0:5000]
galmask = data['MODEST_CLASS'] == 1
galdata = data[galmask]

RA = galdata['RA']
DEC = galdata['DEC']

galaxy_data = [RA, DEC]

p = 8 #2^p bins for Fourier transform
z = 0.5 #redshift

h = kernel_bandwidth(galaxy_data, p, z)

density, xgr, ygr, xedges, yedges, bw_width, bw_height, binnums = gaussian_estimator(galaxy_data, p, h)

minz = 3  #min density for cluster detection

clusters = find_clusters(density, minz)

groups = find_groups(clusters, 4)
groups_field = [np.array(list(map(list, (zip(xedges[group[:,0]],
                                             yedges[group[:,1]])))))
                for group in groups]
weights = [density[group[:,0], group[:,1]] for group in groups]

#groups_avg = np.array([average_position(group, weight) for (group, weight) in
#                       list(map(list, zip(groups_field, weights)))])

rick = [points_in_group(group, binnums) for group in groups]
rick = [y for y in rick if 0 not in y.shape]

groups_avg_pos = np.array([average_position(morty) for morty in rick])

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(12,6))
plt.suptitle('Gaussian kernel estimation with $h = {:04f}$, $\sigma >{}$'.format(h, minz))
gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax1.scatter(galaxy_data[0], galaxy_data[1], marker='.', s=2)
ax1.scatter(groups_avg_pos[:,0] - bw_width/2, groups_avg_pos[:,1] - bw_height/2,
            marker='x', s=80, c='r') 
#ax1.set_aspect('equal')
ax1.set_xlabel('RA (deg)')
ax1.set_ylabel('Dec (deg)')
ax1.set_xlim([min(galaxy_data[0]), max(galaxy_data[0])])
ax1.set_ylim([min(galaxy_data[1]), max(galaxy_data[1])])
SC = ax2.imshow(density.transpose()[::-1])
cax1 = plt.colorbar(SC, cax=ax3)
cax1.set_label('$\sigma$')
plt.savefig('graphing_kernel.png')
plt.show()

#write some code to save the cluster positions to a text file inc. significance
RA_pos = groups_avg_pos[:,0] - bw_width/2
DEC_pos = groups_avg_pos[:,1] - bw_height/2
#sig = np.array([density[cluster[0], cluster[1]] for cluster in clusters])                       
np.savetxt('cluster_locs.txt', np.transpose([RA_pos, DEC_pos]), delimiter = ' ')