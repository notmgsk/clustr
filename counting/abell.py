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

def add_cluster(galaxy_data, loc, scale, size):
    x_cluster = np.random.normal(loc[0], scale, size)
    y_cluster = np.random.normal(loc[1], scale, size)

    galaxy_data[0] = np.append(galaxy_data[0], x_cluster)
    galaxy_data[1] = np.append(galaxy_data[1], y_cluster)

    return galaxy_data, (loc, scale, size)

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
           
    #data_discrete, xedges, yedges = np.histogram2d(galaxy_data[0],
    #galaxy_data[1], bins = M) 
    
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
    
    ang_bins = ang_range/2**p #angle per bin

    h = ang_cluster/ang_bins #the window width for the gaussian estimator
    #should be the number of bins that a cluster would typically
    #occupy

    return h, ang_cluster

def get_clusters(galaxy_data, p, z, h, minz, linking, fname):
    density, xgr, ygr, xedges, yedges, bw_width, bw_height, binnums = gaussian_estimator(galaxy_data, p, h)
    clusters = find_clusters(density, minz)
    groups = find_groups(clusters, linking)
    groups_field = [np.array(list(map(list, (zip(xedges[group[:,0]],
                                                 yedges[group[:,1]])))))
                    for group in groups]
    weights = [density[group[:,0], group[:,1]] for group in groups]
    sigs = [np.mean(vals) for vals in weights]
    rick = [points_in_group(group, binnums, galaxy_data) for group in groups]
    groups_avg_pos = np.array([average_position(morty) for morty in
    rick])

    if len(groups_avg_pos) < 1:
        RA_pos = 'nan'
        DEC_pos = 'nan'
        
    else:
        RA_pos = groups_avg_pos[:,0] - bw_width/2
        DEC_pos = groups_avg_pos[:,1] - bw_height/2
        np.savetxt(fname,
               np.transpose([RA_pos, DEC_pos,sigs]), fmt = '%4.8f',
               delimiter = ' ', header = 'RA (deg)  Dec (Deg)  Sigma')
        
    return np.transpose([RA_pos, DEC_pos])

def parula():
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
     [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
     [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
      0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
     [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
      0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
     [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
      0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
     [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
      0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
     [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
      0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
     [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
      0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
      0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
     [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
      0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
     [0.0589714286, 0.6837571429, 0.7253857143], 
     [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
     [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
      0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
     [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
      0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
     [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
      0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
     [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
      0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
     [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
     [0.7184095238, 0.7411333333, 0.3904761905], 
     [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
      0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
     [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
     [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
      0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
     [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
      0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
     [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
     [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
     [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
      0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
     [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    return parula_map
