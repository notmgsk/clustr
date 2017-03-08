import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches

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
    # For now, data is expected to be data_normed. It might be more useful to,
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

# Histogram configuration
N_bins = 15
bin_width = galaxy_width // N_bins
H, xedges, yedges = np.histogram2d(galaxy_data[0], galaxy_data[1], bins=N_bins)
# xedges and yedges are the corners of the bins used by np.histogram2d. Useful for when we
# need to convert from "bin" coordinates into "real" coordinates. 
edges = np.array(list(zip(xedges, yedges)))
# In order to give some significance to the clusters, we calculate the number of standard
# deviations (i.e. z-value) away from the mean. Then, we only select those that are
# _above_ a minimum z-value.
#
# Note: this calculation of the standard deviation is biased towards a large value because
# it includes the clusters. In the SVA catalog this may not be an issue because clusters
# will be few and far between, but it's something to consider.
sigma = np.std(H)
mu = np.mean(H)
data_normed = (H - mu)/sigma # z-values for each bin
minz = 2
# These are the "bin" coordinates and need translating. A lil bit of list comprehension
# never hurt nobody.
clusters_normed = find_clusters(data_normed, minz)
clusters_field = [[xedges[binx], yedges[biny]] for (binx, biny) in clusters_normed]
# Would've used a list comprehension here but couldn't figure it out.
cluster_patches = []
for cluster in clusters_field:
    cluster_patches.append(patches.Rectangle(cluster, bin_width, bin_width,
                                             edgecolor='r', facecolor='none'))

fig = plt.figure(figsize=(12,6))
fig.suptitle('''Finding clusters with bin width = {} and min z-value = {}'''
             .format(bin_width, minz))
gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax1.scatter(galaxy_data[0], galaxy_data[1], marker='.')
ax1.set_xlim(0, galaxy_width)
ax1.set_ylim(0, galaxy_width)
ax1.set_aspect('equal')
for cluster_patch in cluster_patches:
    ax1.add_patch(cluster_patch)
# The transposing stuff here is weird. Haven't figured out why I need to just
# this quite yet but will update when it makes some sense.
SC = ax2.imshow(data_normed.transpose()[::-1])
cax1 = plt.colorbar(SC, cax=ax3)
cax1.set_label('$\sigma$', size = 20)
plt.tight_layout()
plt.show()


# #the normal distribution of this data set
# x = np.linspace(np.min(data_summed), np.max(data_summed), 1000)
# normal_dist = (1/sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/2*sigma**2)
# plt.figure(3)
# plt.plot(x, normal_dist)
# plt.show()
