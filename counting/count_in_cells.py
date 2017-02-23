import numpy as np
import matplotlib.pyplot as plt

xrange = 400
yrange = xrange
data = np.random.rand(xrange, yrange) > 0.99

def insert_around(data, datum, loc):
    drow = datum.shape[0]//2
    dcol = datum.shape[1]//2
    row = loc[0]
    col = loc[1]
    data[(row-drow):(row+drow), (col-dcol):(col+dcol)] = datum

    return None

def add_cluster(data, pos, loc=0.0, scale=1.0, size=[20,20], sel=0.05):
    clstr = abs(np.random.normal(loc=loc, scale=scale, size=size)) < sel
    insert_around(data, clstr, pos)

    return None

add_cluster(data, pos=[300, 300], scale=0.3, size=[20,20])
add_cluster(data, pos=[100, 100], scale=0.5, size=[40, 40])

def count_in_cells(data, width):
    #find galaxy clusters by determining the density of galaxies in
    #small cells and identifying overdensities

    n_cells = int(xrange/width) #number of cells

    data_summed = np.zeros(shape = (n_cells, n_cells))
    
    #loop over a grid and sum the data in each 'cell'
    for i in range(0,n_cells-1):

        for j in range(0, n_cells-1):

            data_summed[i,j] = np.sum(data[i*width:(i+1)*width, j*width:(j+1)*width])
    
    return data_summed

width = 80 #the width of each cell

data_summed = count_in_cells(data,width)

# The picks out the average points per cell, to be used to cut out background
vmin = np.sum(data_summed)/(xrange/width)**2
# etc...
vmax = np.max(data_summed)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='binary')
plt.subplot(1, 2, 2)
plt.imshow(data_summed, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.grid()

#alternative way to display the counts in cells

#use the standard deviation to give the colourmap some quantiative
#significance
sigma = np.std(data_summed)
data_normed = data_summed/sigma

plt.figure(2)
plt.imshow(data_normed)
cax = plt.colorbar()
cax.set_label('$\sigma$')
plt.grid()
plt.show()
