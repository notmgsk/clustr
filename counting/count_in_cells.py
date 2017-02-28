import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

xrange = 400
yrange = xrange
data = np.random.rand(xrange, yrange) > 0.997

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

add_cluster(data, pos=[300, 300], scale=0.3, size=[20, 20])
add_cluster(data, pos=[100, 100], scale=0.5, size=[40, 40])

def count_in_cells(data, width):
    """Splits data up into width x width cell and then counts the data points within those cells. The shape of the return value is (data/width) x (data/width)."""
    #find galaxy clusters by determining the density of galaxies in
    #small cells and identifying overdensities

    n_cells = int(xrange/width) #number of cells

    data_summed = np.zeros(shape = (n_cells, n_cells))
    
    #loop over a grid and sum the data in each 'cell'
    for i in range(0,n_cells-1):

        for j in range(0, n_cells-1):

            data_summed[i,j] = np.sum(data[i*width:(i+1)*width, j*width:(j+1)*width])
    
    return data_summed

width = 40 #the width of each cell

data_summed = count_in_cells(data,width)

# # The picks out the average points per cell, to be used to cut out background
# vmin = np.sum(data_summed)/(xrange/width)**2
# # etc...
# vmax = np.max(data_summed)

# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.imshow(data, cmap='binary')
# plt.subplot(1, 2, 2)
# plt.imshow(data_summed, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.grid()
# plt.show()

# subtract from each data point the mean of the distribution
# divide this new data set by the standard deviation
# this data set now tells us how mmany standard deviations larger than
# the mean each point is

sigma = np.std(data_summed)
mu = np.mean(data_summed)
data_normed = abs(data_summed - mu)/sigma

#defining the red rectangles that show where the clusters are located 
clustr1 = patches.Rectangle((290, 290), 20, 20, edgecolor = 'r', facecolor = 'none')
clustr2 = patches.Rectangle((80, 80), 40, 40, edgecolor = 'r', facecolor = 'none')

fig = plt.figure(figsize=(12,6))
gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax1.imshow(data, cmap='binary')
ax1.add_patch(clustr1)
ax1.add_patch(clustr2)
SC = ax2.imshow(data_normed)
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
