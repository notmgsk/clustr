import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import gridspec
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import rc
import networkx as nx
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

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

def find_groups(data, r=1):
    """Returns the cluster groups; i.e., if there are neighbouring
    overdensities from the galaxy histogram, they are considered to be part of
    the same cluster (group).

    Note: data should be the overdensities in the histogram picked out by
    find_clusters.""" 
    def dist(v1, v2):
        return (data[v1][0] - data[v2][0])**2 + (data[v1][1] - data[v2][1])**2
    
    # Vertices are unique labels for each point in data
    verts = np.arange(len(data))
    # The problem now is to construct edges between vertices. An edge between
    # two vertices exists if the distance between the two corresponding points
    # <= r. These edges include duplicates but that's ok... doesn't affect the
    # end goal.
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
    """Returns the average position of points; shape of points must be (N,2),
    and average is calculated along columns"""  
    # totw = np.sum(weights)

    # avgx = np.sum([x * w for (x,w) in zip(group[:,0], weights)])/totw
    # avgy = np.sum([y * w for (y,w) in zip(group[:,1], weights)])/totw
    return [np.sum(points[:,0])/len(points),
            np.sum(points[:,1])/len(points)]

def indices_in_bin(binn, binnums):
    """Returns the indices of elements of binnums for which the bin is equal to
    binn""" 
    return np.argwhere([(row == binn).all() for row in binnums.T]).ravel()

def points_in_bin(binn, binnums, galaxy_data):
    """Returns an array of [RA, DEC] points that are within binn"""
    indices = indices_in_bin(binn, binnums)

    return np.column_stack((galaxy_data[0][indices], galaxy_data[1][indices]))

def points_in_group(group, binnums, galaxy_data):
    """Returns an array of [RA, DEC] points that are within group's bins"""
    return np.concatenate(([points_in_bin(binn, binnums, galaxy_data) for binn
                            in group]))
