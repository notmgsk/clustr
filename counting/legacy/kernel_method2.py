import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.path import Path

def insert_around(data, datum, loc):
    drow = datum.shape[0]//2
    dcol = datum.shape[1]//2
    row = loc[0]
    col = loc[1]
    data[(row-drow):(row+drow), (col-dcol):(col+dcol)] = datum

    return None

def add_cluster(loc, scale, size):
    global galaxy_data

    x_cluster = np.random.normal(loc[0], scale, size)
    y_cluster = np.random.normal(loc[1], scale, size)

    galaxy_data[0] = np.append(galaxy_data[0], x_cluster)
    galaxy_data[1] = np.append(galaxy_data[1], y_cluster)

    return (loc, scale, size)

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


# Set up the galaxy field
galaxy_width = 400
N_field_gals = 30
galaxy_data = [np.random.rand(N_field_gals) * galaxy_width,
               np.random.rand(N_field_gals) * galaxy_width]

add_cluster([200, 125], 10, 20)
add_cluster([300, 300], 10, 15)


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

    M = 2**p #number of bins
    n = np.size(galaxy_data)
    length = len(galaxy_data)
    a = 0 #limits
    b = length

    bin_width = (b-a)/M #the width of each discrete cell

    data_discrete, xedges, yedges = np.histogram2d(galaxy_data[0],
    galaxy_data[1], bins = M) 
    
    data_fourier = np.fft.fft2(data_discrete) #fourier transform the
    #discrete data

    #create the gaussian kernel
    x = np.linspace(0, M, M)
    y = np.linspace(0, M, M)
    xgr,ygr = np.meshgrid(x,y)
    kernel = np.exp(-0.5*(xgr**2 + ygr**2)/h)/np.sqrt(2*np.pi)

    kernel_fourier = np.fft.fft2(kernel) #take the fourier transform
    #of the gaussian kernel

    density_fourier = kernel_fourier*data_fourier #convolve the kernel
    #and the data

    density = np.fft.ifft2(density_fourier) #perform the inverse
    #fourier transform to get the density in real space
    
    return density, xgr, ygr, xedges, yedges, bin_width

# #calculate the optimal bandwidth as detailed in Silverman
# h = (4/3)**(1/5)*np.std(data)*np.sum(data)**(-1/5)

h = 15

density, xgr, ygr, xedges, yedges, bin_width = gaussian_estimator(galaxy_data, 7, h)

mind = 1.5 #min density for cluster detection

clusters = find_clusters(density, mind)
clusters_field = [[xedges[binx], yedges[biny]] for (binx, biny) in
clusters]
cluster_patches = []

# line_to = [Path.LINETO]*(len(clusters_field)-1)
# cluster_codes = [Path.MOVETO] + line_to 
# cluster_verts = np.array(map(list, (clusters_field)))

# path = Path(cluster_verts, cluster_codes)

# cluster_patch = patches.PathPatch(path, facecolor='r', lw=1)


for cluster in clusters_field:
    
    cluster_patches.append(patches.Rectangle(cluster, bin_width,
                                             bin_width, edgecolor =
                                             'r', facecolor = 'none'))


fig = plt.figure(figsize=(12,6))
plt.suptitle('Gaussian kernel estimation with $h = {}$'.format(h))
gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax1.scatter(galaxy_data[0], galaxy_data[1], marker='.')
for cluster_patch in cluster_patches:
    ax1.add_patch(cluster_patch)
SC = ax2.imshow(np.real(density).transpose()[::-1])
cax1 = plt.colorbar(SC, cax=ax3)
plt.tight_layout()
plt.show()


