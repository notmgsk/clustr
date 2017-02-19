import numpy as np
import matplotlib.pyplot as plt

### could add a random distribution of the brightnesses
### will the radii be normally distributed?

N = 500 #size of the image
n = 45 #number of field galaxies
nclust = 30

field = np.zeros(shape=(N,N)) #generate the empty fields
cluster = np.zeros(shape=(N,N))

gal_pos = 500*np.random.random_sample((2,n)) #generate n pairs of random positions, which will define the galaxy positions

lmean = 6 #mean radius 
lsigma = 2 #standard deviation of radius
l = np.random.normal(lmean, lsigma, n) #a normally distributed range of radii for the galaxies

for i in range(0,n):
        
        #coordinates of the positions of the centre of each galaxy
        a = gal_pos[0,i]
        b = gal_pos[1,i]

        #define the indices of the region to place the galaxy
        pmin = int(max(0, a - 1.25*l[i]))
        pmax = int(min(N, a + 1.25*l[i]))
        qmin = int(max(0, b - 1.25*l[i]))
        qmax = int(min(N, b + 1.25*l[i]))
        
        for p in range(pmin,pmax):
                for q in range(qmin,qmax):
                    if ((p-a)**2 + (q-b)**2 - l[i]**2) < 1:
                        field[p,q] = 1*np.exp(-np.sqrt((p-a)**2+(q-b)**2)/l[i]) #the exponential factor is the surface profile

#define a box for the cluster from clustmin to clustmax in x and y
clustmin = 200
clustmax = 230
clust_pos = clustmax*np.random.random_sample((2,n)) - clustmin
lclust = np.random.normal(lmean, lsigma, nclust) #a normally distributed range of radii for the galaxies


for i in range(0,nclust):

        #coordinates of the positions of the centre of each galaxy
        a = clust_pos[0,i]
        b = clust_pos[1,i]
        
        #define the indices of the region to place the galaxy
        pmin = int(max(0, a - 1.25*lclust[i]))
        pmax = int(min(N, a + 1.25*lclust[i]))
        qmin = int(max(0, b - 1.25*lclust[i]))
        qmax = int(min(N, b + 1.25*lclust[i]))
        
        for p in range(pmin,pmax):
                for q in range(qmin,qmax):
                    if ((p-a)**2 + (q-b)**2 - lclust[i]**2) < 1:
                        cluster[p,q] = 1*np.exp(-np.sqrt((p-a)**2+(q-b)**2)/lclust[i]) #the exponential factor is the surface profile

img = field + cluster

plt.imshow(img, cmap = 'binary')
plt.show()
