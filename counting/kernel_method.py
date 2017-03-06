import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import gridspec

xrange = 400
yrange = xrange
data = np.random.rand(xrange, yrange) > 0.999

def insert_around(data, datum, loc):
    drow = datum.shape[0]//2
    dcol = datum.shape[1]//2
    row = loc[0]
    col = loc[1]
    data[(row-drow):(row+drow), (col-dcol):(col+dcol)] = datum

    return None

def add_cluster(data, pos, loc=0.0, scale=1.0, size=[20,20], sel=0.01):
    clstr = abs(np.random.normal(loc=loc, scale=scale, size=size)) < sel
    insert_around(data, clstr, pos)

    return None

# def find_clusters(data, cond):
#     """Returns the coordinates of overdensities (i.e. where cond is
#     satisfied)"""
#     # For now, data is expected to be data_normed. It might be more useful to
#     # have the input data be the galaxy field, and then this particular
#     # function would call count_in_cells, and return the relevent coordinates
#     # pointing into the galaxy field, rather than into the density map (i.e.
#     # data_normed). I say this would be more useful because, currently, after
#     # calling this function, we have to manually convert the coordinates from
#     # data_normed coords into galaxy field coords.
#     return np.argwhere(data > cond) 

add_cluster(data, pos=[300, 300], scale=0.1, size=[20, 20])
add_cluster(data, pos=[100, 100], scale=0.5, size=[40, 40])


def gaussian_estimator(data,p,h):
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

    M = 2**p #number of points to sample the data at
    n = np.size(data)
    length = data.shape
    a = 0 #limits
    b = length[0]

    width = (b-a)/M #the width of each discrete cell
    
    data_discrete = np.zeros(shape = (M, M)) #preallocating the array
    #for the discretised data
    
    #loop over a grid and sum the data in each 'cell', to discretise
    #the data so it can be fourier transformed
    for i in range(0, M-1):

        for j in range(0, M-1):

            data_discrete[i,j] = np.sum(data[i*width:(i+1)*width, j*width:(j+1)*width])
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
    
    return density, xgr, ygr

# #calculate the optimal bandwidth as detailed in Silverman
# h = (4/3)**(1/5)*np.std(data)*np.sum(data)**(-1/5)

h = 10

density, xgr, ygr = gaussian_estimator(data, 7, h)


fig = plt.figure(figsize=(12,6))
plt.suptitle('Gaussian kernel estimation with $h = ' + str(h) +'$')
gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax1.imshow(data, cmap='binary')
SC = ax2.imshow(np.real(density))
cax1 = plt.colorbar(SC, cax=ax3)
plt.tight_layout()
plt.show()


