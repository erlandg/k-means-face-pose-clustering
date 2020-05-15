import numpy as np
import matplotlib.pyplot as plt
import random

A = np.genfromtxt('data/frey-faces.csv')
testdata = A.copy()
N = 3

def k_means(data, clusters):
    dim = data.shape[1]
    clst = np.zeros([3, 2])
    avg = np.mean(data, axis = 0)
    sd = np.std(data, axis = 0)
    clst = avg + np.random.randn(clusters, dim) * sd
    d  = np.zeros([data.shape[0], clusters])
    a = 0
    y = np.zeros([dim, 2])
    change = 1
    while change != 0:
        a += 1
        for i in range(clusters):
            d[:,i] = np.linalg.norm(data - clst[i], axis = 1)
        change = clst.copy()
        for i in range(clusters):
            clst[i] = np.mean(data[np.argmin(d, axis=1) == i], axis = 0)
        change = np.linalg.norm(clst - change)
    y = np.c_[np.arange(0, data.shape[0], 1), np.argmin(d, axis=1), np.min(d, axis=1)]
    return clst, y, a


k_m = k_means(testdata, N)
skr = k_m[0].copy()

for row in range(N):
    plt.subplot(1,N,row+1)
    image = skr[row]
    reshaped_image = image.reshape(28,20)
    plt.imshow(reshaped_image, cmap='gray', interpolation='bilinear')
    plt.title('Cluster '+str(row))
plt.show()


pics = k_m[1].copy()
pics = pics[pics[:,2].argsort()] # Small to far
close = pics[:50]
closecop = close.copy()
far = pics[-50:]
farcop = far.copy()
np.random.shuffle(close)
np.random.shuffle(far)

for row in range(3):
    plt.subplot(1,3,row+1)
    image = testdata[int(closecop[row,0])]
    reshaped_image = image.reshape(28,20)
    plt.imshow(reshaped_image, label='hei', cmap='gray', interpolation='bilinear')
    plt.title('Close to centroid\n'+'Picture '+str(closecop[row,0])+',\nClass '+str(closecop[row,1]))
    closecop = close[close[:,1] != closecop[row,1]]
plt.show()

for row in range(3):
    plt.subplot(1,3,row+1)
    image = testdata[int(farcop[row,0])]
    reshaped_image = image.reshape(28,20)
    plt.imshow(reshaped_image, cmap='gray', interpolation='bilinear')
    plt.title('Far from centroid\n'+'Picture '+str(farcop[row,0])+',\nClass '+str(farcop[row,1]))
    farcop = far[far[:,1] != farcop[row,1]]
plt.show()
