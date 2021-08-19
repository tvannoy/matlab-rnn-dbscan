# matlab-rnn-dbscan
This repo contains a MATLAB implementation of the [RNN-DBSCAN algorithm by Bryant and Cios](https://doi.org/10.1109/TKDE.2017.2787640). This implementation is based upon the graph-based interpretation presented in their paper.


RNN DBSCAN is a density-based clustering algorithm that uses reverse nearest neighbor counts as an estimate of observation density. It is based upon traversals of the directed k-nearest neighbor graph, and can handle clusters of different densities, unlike DBSCAN. For more details about the algorithm, see the [original paper](https://doi.org/10.1109/TKDE.2017.2787640).

## Dependencies
- My [knn-graphs](https://github.com/tvannoy/knn-graphs) MATLAB library
- Statistics and Machine Learning Toolbox

To run the tests contained in the Jupyter notebook, you will need to install the Jupyter [matlab kernel](https://github.com/calysto/matlab_kernel).

To use the NN Descent algorithm to construct the KNN graph used by RNN DBSCAN, you need [pynndescent](https://github.com/lmcinnes/pynndescent) and [MATLAB's Python language interface](https://www.mathworks.com/help/matlab/call-python-libraries.html). I recommend using Conda to set up an environment, as MATLAB is picky about which Python versions it supports. 

## Usage
`RnnDbscan` is a class with a single public method, `cluster`. The results of the clustering operation are stored in read-only public properties. 

Creating an RnnDbscan object:
```matlab
% Create an RnnDbscan object using a 5-nearest-neighbor graph.
% nNeighborsIndex is how many neighbors used to create the knn index, and must be >= nNeighbors + 1
% because the index includes self-edges (each point is it's own nearest neighbor).
nNeighors = 5;
nNeighborsIndex = 6;
rnndbscan = RnnDbscan(data, nNeighbors, nNeighborsIndex);

% Use the NN Descent algorithm to create the knn index; this is much faster than an exhaustive search
rnndbscan = RnnDbscan(data, nNeighbors, nNeighborsIndex, 'Method', 'nndescent');

% Explicitly use an exhaustive search, which is the default
rnndbscan = RnnDbscan(data, nNeighbors, nNeighborsIndex, 'Method', 'knnsearch');

% Use a precomputed knn index
knnidx = knnindex(data, nNeighborsIndex);
rnndbscan = RnnDbscan(data, nNeighbors, knnidx);
```

Clustering:
```matlab
rnndbscan.cluster();
% Or
cluster(rnndbscan);

% Inspect clusters, outliers, and labels
rnndbscan.Clusters
rnndbscan.Outliers
rnndbscan.Labels
```

For more details, see the help text: `help RnnDbscan`. `RNN-DBSCAN tests.ipynb` also contains many tests, which can be used as usage examples. 

## Contributing
All contributions are welcome! Just submit a pull request or open an issue.