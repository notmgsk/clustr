import abell as abell
import numpy as np
import scipy.stats as stats
from astropy.io import fits

sva_catalog_path = "./spte_sva1_red.fits"
clusters_catalog_path = "clusters_cic.txt"
# If "a", will append to the cluster catalog instead of overwriting it.
clusters_catalog_open = "ab"
# This is obviously an important configuration option. Needs to be refined
# by-hand unfortunately. 
N_bins = 50
# Instead of running the histogram over the entire dataset in one go, I suspect
# it may be better to approach it in steps. This is useful in a couple ways:
#   1. Quicker to test;
#   2. Can get a better idea of what N_bins needs to be. If we tried to work on
#   the full data, we'd need N_bins to be something huge. This way it's a
#   "reasonable" size.
interval = 10000

# Gives an array of 2 elements, the second of which has the data.
hdulist = fits.open(sva_catalog_path, mmap=True)
# sva_catalog_size = interval*10
sva_catalog_size = hdulist[1].data.shape[0]

clusters_fh = open(clusters_catalog_path, clusters_catalog_open)

for i in np.arange(sva_catalog_size//interval):
    data = hdulist[1].data[(i*interval):((i+1)*interval)]
    galdata = data

    RA = galdata['RA'] 
    DEC = galdata['DEC']

    RAmin,RAmax = (np.min(RA), np.max(RA))
    DECmin,DECmax = (np.min(DEC), np.max(DEC))

    galaxy_data = [RA, DEC]

    # bin_width = galaxy_width // N_bins xedges and yedges are the corners of
    # the bins used by np.histogram2d. Useful for when we need to convert from
    # "bin" coordinates into "real" coordinates.
    H, xedges, yedges, binnums = stats.binned_statistic_2d(
        galaxy_data[0],
        galaxy_data[1],
        values=None,
        statistic='count',
        bins=N_bins,
        expand_binnumbers=True)
    # binnums seems to start counting from 1, not 0.
    binnums -= 1
    # In order to give some significance to the clusters, we calculate the
    # number of standard deviations (i.e. z-value) away from the mean. Then, we
    # only select those that are _above_ a minimum z-value. Note: this
    # calculation of the standard deviation is biased towards a large value
    # because it includes the clusters. In the SVA catalog this may not be an
    # issue because clusters will be few and far between, but it's something to
    # consider.
    sigma = np.std(H)
    mu = np.mean(H)
    data_normed = (H - mu)/sigma # z-values for each bin
    minz = 3
    # These are the "bin" coordinates and need translating. A lil bit of list
    # comprehension never hurt nobody.
    clusters_normed = abell.find_clusters(data_normed, minz)
    clusters_field = [[xedges[binx], yedges[biny]] for (binx, biny) in
                      clusters_normed] 
    # Would've used a list comprehension here but couldn't figure it out.
    cluster_patches = []
    # for cluster in clusters_field:
    #     cluster_patches.append(patches.Rectangle(cluster, bin_width,
    #     bin_width, edgecolor='b', facecolor='none', alpha=0.25))

    groups = abell.find_groups(clusters_normed, 4)
    # There's gotta be a nicer way to do this. Ugly as f right now. All it's
    # doing is converting from bin coordinates to real coordinates, but man is
    # it ugly.
    groups_field = [np.array(list(map(list, (zip(xedges[group[:,0]],
                                                 yedges[group[:,1]]))))) 
                    for group in groups] 
    # weights = [H[group[:,0], group[:,1]] for group in groups] # This is
    # converted to a numpy array so we can use array slicing on it later.
    # groups_avg = np.array([average_position(group, weight) for (group,
    # weight) in map(list, zip(groups_field, weights))])
    groups_avg_pos = np.array([
        abell.average_position(abell.points_in_group(group,
                                                     binnums,
                                                     galaxy_data))
        for group in groups])

    # Finally, write our clusters to the file
    np.savetxt(clusters_fh,
               groups_avg_pos,
               fmt='%4.8f')

clusters_fh.close()
hdulist.close()
