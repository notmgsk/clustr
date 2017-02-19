import numpy as np
import matplotlib.pyplot as plt

xrange = 400
yrange = xrange
data = np.random.rand(xrange, yrange) > 0.99

cluster_1_loc = [300, 300]
cluster_1 = abs(np.random.normal(loc=0.0, scale=0.3, size=(20,20))) < 0.05
cluster_2_loc = [100, 100]
cluster_2 = abs(np.random.normal(loc=0.0, scale=0.5, size=(40,40))) < 0.05

data[290:310, 290:310] = cluster_1
data[80:120, 80:120] = cluster_2

plt.imshow(data, cmap='binary')
plt.grid()
plt.show()
