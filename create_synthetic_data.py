# SPDX-License-Identifier: 3-Clause-BSD
# Copyright (c) 2020 Trevor Vannoy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from sklearn.datasets import make_blobs, make_moons, make_classification


"""
Dataset with unbalanced cluster sizes and random noise
"""
data, labels = make_blobs(n_samples = [20, 100, 500, 1500], centers=[[-2,8], [-2,-1], [5,5], [3,-2]], cluster_std=0.5, random_state=1)

data = np.vstack((data, 4 * np.random.standard_normal(size=(50, 2))))
labels = np.concatenate([labels, -1 * np.ones(50)])

# plt.scatter(data[:,0], data[:,1], c=labels)
# plt.show()

savemat('unbalanced_blobs_2d_noise.mat', {'data': data, 'labels': labels})

"""
Moons with noise
"""
data, labels = make_moons(n_samples=500, noise=0.1, random_state=1)

data = np.vstack((data, np.random.standard_normal(size=(20, 2))))
labels = np.concatenate([labels, -1 * np.ones(20)])

# plt.scatter(data[:,0], data[:,1], c=labels)
# plt.show()

savemat('moons_with_noise.mat', {'data': data, 'labels': labels})

"""
Gaussian blobs with noise features and random noise
"""
data, labels = make_classification(n_samples=1000, n_classes=4, n_clusters_per_class=1, weights=[0.02, 0.18, 0.3, 0.5], n_features=20, shuffle=False, class_sep=3, random_state=5)

data = np.vstack((data, 5 * np.random.standard_normal(size=(50, 20))))
labels = np.concatenate([labels, -1 * np.ones(50)])

# plt.scatter(data[:,0], data[:,1], c=labels)
# plt.show()

savemat('unbalanced_blobs_20d_noise.mat', {'data': data, 'labels': labels})

"""
Grid dataset with varying densities and noise
"""
grid1 = np.mgrid[-1:1:5j, -1:1:5j]
grid1 = np.reshape(grid1, (2,25)).T + [-2.5, 0]
grid2 = np.mgrid[-1:1:10j, -1:1:10j]
grid2 = np.reshape(grid2, (2,100)).T
grid3 = np.mgrid[-1:1:20j, -1:1:20j]
grid3 = np.reshape(grid3, (2,400)).T + [-1, 2.5]
noise = np.random.standard_normal(size=(20,2))

data = np.vstack((grid1, grid2, grid3, noise))

labels = np.concatenate([np.repeat(1, 25), np.repeat(2, 100), np.repeat(3, 400), np.repeat(-1, 20)])

# plt.scatter(data[:,0], data[:,1], c=labels)
# plt.show()

savemat('multiple_grids_noise.mat', {'data': data, 'labels': labels})
