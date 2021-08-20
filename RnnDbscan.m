classdef RnnDbscan < handle
%RnnDbscan RNN DBSCAN clustering 
%   RNN DBSCAN is a density-based clustering algorithm that uses reverse nearest
%   neighbor counts as an estimate of observation density. It is based upon
%   traversals of the directed k-nearest neighbor graph, and can handle clusters
%   of different densities, unlike DBSCAN.
%
%   RnnDbscan constructor:
%       rnnDbscan = RnnDbscan(X, k, indexNeighbors) creates an RNN DBSCAN
%       object for input data X, with k-nearest neighbors used in
%       constructing the knn graph. indexNeighbors is the number of
%       neighbors used to build the knn k-nearest neighbors index.
%       indexNeighbors must be >= k + 1
%
%       rnnDbscan = RnnDbscan(X, k, indexNeighbors, 'Method', method)
%       creates an RNN DBSCAN object for input data X, with k nearest neighbors
%       used in constructing the knn graph, using the specified knn method.
%
%       Optional parameters:
%       'Method'        - The method to use when building the knn index
%           "knnsearch" - Use KNNSEARCH from Mathwork's Statistics and
%                         Machine Learning Toolbox. This is the default
%                         method. 
%           "nndescent" - Use the nearest neighbor descent algorithm to
%                         build an approximate knn index. For large
%                         matrices, this is much faster than KNNSEARCH, at
%                         the cost of slightly less accuracy. This method
%                         requires the pynndescent python package to be
%                         installed and accessible through MATLAB's Python
%                         external language interface
%
%   Rows of X correspond to observations, and columns correspond to variables.
%
%   RnnDbscan properties:
%       K                 - number of nearest neighbors 
%       Data              - input data X
%       KnnIndex          - k-nearest neighbor index
%       KnnGraph          - knn graph used for clustering
%       Clusters          - Clusters{i} contains indices for points in cluster i
%       Outliers          - array of outliers that do not belong to a cluster
%       CorePoints        - list of core points
%       ClusterDensities  - density of each cluster
%       Labels            - cluster labels; -1 corresponds to outliers  
%
%   RnnDbscan methods:
%       cluster           - cluster the input data
%
%   This algorithm was presented in https://dx.doi.org/10.1109/TKDE.2017.2787640,
%   by Bryant and Cios but no implementation was given. This implementation is
%   based on the graph-based interpretation presented in the original paper.   
%
%   See also knnindex, knngraph

% SPDX-License-Identifier: BSD-3-Clause 
% Copyright (c) 2020 Trevor Vannoy

    properties (SetAccess = public, AbortSet)
        % Number of nearest neighbors used to construct the knn graph
        K (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    properties (SetAccess = private)
        % Input data; rows are observations, and columns are variables
        Data (:,:) double
        % k-nearest neighbor index
        KnnIndex (:,:) int32
        % Directed k-nearest neighbor graph
        KnnGraph (1,1) digraph
        % Cell array of clusters; Clusters{i} contains indices for all points
        % that are in cluster i
        Clusters (:,1) cell
        % Outliers that do not belong to any cluster
        Outliers (1,:) int32
        % List of core points, which are points with >= k reverse nearest
        % neighbors. 
        CorePoints (1,:) int32
        % Cluster densities, defined as the max distance between 
        % connected core points
        ClusterDensities (1,:) double
        % Cluster labels. Outliers have a label value of -1; all other 
        % clusters have labels starting at 1
        Labels (:,1) int32 = []
    end

    methods (Access = public)
        function obj = RnnDbscan(X, k, index, options)
        %RnnDbscan Construct an RNN DBSCAN object
        %   rnnDbscan = RnnDbscan(X, k, nIndexNeighbors) creates an RNN DBSCAN
        %   object for input data X, with k-nearest neighbors used in
        %   constructing the knn graph. indexNeighbors is the number of
        %   neighbors used to build the k-nearest neighbors index.
        %   indexNeighbors must be >= k + 1
        %
        %   rnnDbscan = RnnDbscan(X, k, knnIndex) creates an RNN DBSCAN object
        %   using a precomputed knn index, knnIndex. knnIndex must have the same
        %   number of rows as X.
        %
        %   rnnDbscan = RnnDbscan(X, k, nIndexNeighbors, 'Method', method)
        %   creates an RNN DBSCAN object for input data X, with k nearest
        %   neighbors used in constructing the knn graph, using the specified
        %   knn method.
        %
        %   Optional parameters:
        %   'Method'        - The method to use when building the knn index
        %       "knnsearch" - Use KNNSEARCH from Mathwork's Statistics and
        %                     Machine Learning Toolbox. This is the default
        %                     method. 
        %       "nndescent" - Use the nearest neighbor descent algorithm to
        %                     build an approximate knn index. For large
        %                     matrices, this is much faster than KNNSEARCH, at
        %                     the cost of slightly less accuracy. This method
        %                     requires the pynndescent python package to be
        %                     installed and accessible through MATLAB's Python
        %                     external language interface

        % TODO: I could add another constructor that doesn't take indexNeighbors and uses k instead
            arguments
                X (:,:) double
                k (1, 1) {mustBePositive, mustBeInteger}
                index
                options.Method (1,1) string {mustBeMember(options.Method, ["knnsearch", "nndescent"])} = "knnsearch"
            end

            obj.Data = X;

            if numel(index) == 1
                indexNeighbors = index;

                if indexNeighbors < k + 1
                    error("indexNeighbors must be >= k + 1")
                end

                obj.KnnIndex = knnindex(obj.Data, indexNeighbors, ...
                    'Method', options.Method);
            elseif size(index, 1) == size(X, 1)
                if size(index, 2) < k + 1
                    error("knnIndex must have # columns >= k + 1")
                end

                obj.KnnIndex = index;
            else
                % TODO: better error message
                error("argument 3 is incorrect")
            end

            obj.Labels = zeros(size(X, 1), 1, 'int32');
            obj.K = k;

            % the setter for k won't execute for k = 1 because 1 is the
            % default value for k, so AbortSet will stop set.K from running
            if k == 1
                obj.KnnGraph = knngraph(obj.KnnIndex, obj.K, ...
                    'Precomputed', true);
            end
        end

        function cluster(obj)
        %CLUSTER Cluster the input data used to create the RnnDbscan object
        %   The resulting cluster assignments and labels are saved to the 
        %   RnnDbscan object

            obj.findInitialClusters();
            obj.computeClusterDensities();
            obj.addDirectlyDensityRechablePoints();
            obj.createExtendedClusters();
        end
    end

    methods
        function obj = set.K(obj, k)
            % this method only gets called if k != obj.K, due to AbortSet
            if k >= size(obj.KnnIndex, 2)
                error("Error setting property 'K' of  class 'RnnDbsan':\n" + ...
                    "'K' must be less than knn index size %d", ...
                    size(obj.KnnIndex, 2));
            end

            obj.K = k;

            % XXX: Mathwork's says I shouldn't set other properties in a setter
            % method, which I do conceptually agree with, but I want to be able
            % to "reset" the object to cluster using a different k without
            % having to make a new object and recompute the knn index
            obj.KnnGraph = knngraph(obj.KnnIndex, obj.K, ...
                'Precomputed', true);
            obj.Labels = zeros(1, size(obj.Data, 1), 1, 'int32');
            obj.Outliers = int32([]);
            obj.CorePoints = int32([]);
            obj.Clusters = {};
            obj.ClusterDensities = [];
        end
    end

    methods (Access = private)
        function findInitialClusters(obj)
        % Find initial clusters from the core point subgraph
        %
        % In the graph-based interpretation of the RNN DBSCAN algorithm,
        % an initial clustering of core points is defined by the weakly
        % connected components in the core point subgraph, which is a subgraph
        % of the knn graph consisting of only core points. Core points are
        % defined as points with a reverse nearest neighbor count >= k, which
        % corresponds to a node having an indegree >= k in the knn graph

            obj.CorePoints = int32(find(indegree(obj.KnnGraph) >= obj.K));

            % creating a subgraph renumbers the node IDs. This is problematic
            % because I am expecting the node IDs to correspond to the data
            % points' indices in the data matrix. To remedy this, the subgraph's
            % node IDs have to be converted back to the original node IDs.
            corePointSubgraph = subgraph(obj.KnnGraph, obj.CorePoints);     

            % get the initial clusters from the connected components.
            % 'OutputForm' = 'cell' returns a cell array where the i-th cell
            % contains all of the node IDs belonging to component i; this 
            % results in a cell array where each cell is a cluster.
            clusters = conncomp(corePointSubgraph, 'Type', 'weak', ...      
                'OutputForm', 'cell');

            % convert the subgraph's node IDs back to the original node IDs that
            % correspond to the data points' indices. 
            obj.Clusters = cellfun(@(c) obj.CorePoints(c), clusters, ...
                'UniformOutput', false);

            for i = 1:length(obj.Clusters)
                obj.Labels(obj.Clusters{i}) = int32(i);
            end
        end

        function addDirectlyDensityRechablePoints(obj)
        % Add directly density-reachable border points to clusters
        %
        % Directly density-reachable points are border points that are in the
        % k-neighborhood of a core point.             
        % 
        % This function expands the clusters to include some border points, 
        % but the resulting clustering is still considered to be the initial
        % set of clusters in the RNN DBSCAN paper, rather than the extended
        % clusters. This would be part of the "ExpandCluster" algorithm in the
        % paper, but I am leveraging the graph interpretation rather than the
        % algorithm pseudocode

            for i = 1:length(obj.Clusters)
                borderPoints = [];

                % find k-nearest neighbors of every point in cluster i
                for j = 1:length(obj.Clusters{i})

                    % k-nearest neighbors of a point j are the successors of 
                    % node j in the knn graph, i.e. there is an edge originating
                    % from node j 
                    neighbors = int32(successors(obj.KnnGraph, obj.Clusters{i}(j)));

                    % only add points to a cluster if they do not already belong
                    % to a cluster
                    unclusteredPoints = neighbors(obj.Labels(neighbors) == 0);

                    % points in a cluster are likely to share neighbors, but we 
                    % don't want these neighbors to be added to the cluster
                    % more than once, so we take the union of all
                    % previously-found neighbors and the neighbors of point j
                    borderPoints = union(borderPoints, unclusteredPoints);
                end
                
                % force borderPoints to be a row vector, because the
                % clusters are also row vectors
                borderPoints = borderPoints(:)';
                
                % add neighbors of core points in cluster i to cluster i
                obj.Clusters{i} = horzcat(obj.Clusters{i}, borderPoints(:)');
                obj.Labels(borderPoints) = int32(i);
            end
        end

        function createExtendedClusters(obj)
        % Cluster remaining points that satisify the extended cluster definition
        %
        % After creating the initial clusters, which comprise core points and
        % their directly density-reachable border points, remaining points are
        % assigned to a cluster C if they have a core point in their
        % k-neighborhood that is < den(C) away. Such points are added to the 
        % cluster of their closest core point. Points that do not satisfy the
        % extended cluster definition are labeled as outliers.

            clusteredPoints = [obj.Clusters{:}];
            pointIndices = int32([1:size(obj.Data, 1)]);
            unclusteredPoints = setdiff(pointIndices, clusteredPoints);

            for unclusteredPoint = unclusteredPoints
                pointIsNoise = true;

                % find k-nearest neighbors of the unclustered point
                neighbors = int32(successors(obj.KnnGraph, unclusteredPoint));

                % determine which neighbors are core points; we do not want to
                % cluster the unclustered point if it is not connected to any 
                % core points
                corePointNeighbors = intersect(neighbors, obj.CorePoints);

                if corePointNeighbors
                    corePointNeighborsData = obj.Data(corePointNeighbors, :);
                    unclusteredPointData = obj.Data(unclusteredPoint, :);

                    % sort core point neighbors by distance from the
                    % unclustered point
                    distances = pairwiseDist(unclusteredPointData, ...
                        corePointNeighborsData);
                    [distances, sortIdx] = sort(distances);
                    corePointNeighbors = corePointNeighbors(sortIdx);

                    for i = 1:length(corePointNeighbors)
                        % find which cluster the core point belongs to
                        clusterIdx = obj.Labels(corePointNeighbors(i));

                        if distances(i) <= obj.ClusterDensities(clusterIdx)
                            % add unclustered point to cluster
                            obj.Clusters{clusterIdx} = horzcat(obj.Clusters{clusterIdx}, unclusteredPoint); 
                            obj.Labels(unclusteredPoint) = int32(clusterIdx);

                            pointIsNoise = false;

                            % point can't be assigned to more than one cluster, 
                            % so stop once it is assigned
                            break;
                        end
                    end
                end
                if pointIsNoise
                    % the unclustered point does not have any neighbors that are core points, so it is considered an outlier
                    obj.Outliers = horzcat(obj.Outliers, unclusteredPoint);
                    obj.Labels(unclusteredPoint) = int32(-1);
                end
            end
        end

        function computeClusterDensities(obj)            
        % Compute the densities of all the clusters.
        %
        % den(C) max(dist(x,y)) s.t. x,y are both core points in cluster C, 
        % and y is directly density-reachable from x (i.e. y is a k-nearest 
        % neighbor of x); equivalently, den(C) is the maximum distance between
        % points that have an edge between them in the core points subgraph.
        % 
        % Since the cluster density is defined as the maximum distance between 
        % nearest neighbor core points, the cluster density does not change as 
        % non-core points are added to the cluster. Thus we can calculate the 
        % cluster densities after finding the initial clusters of core points,
        % which are defined by the weakly connected components of the core
        % points subgraph.

            obj.ClusterDensities = zeros(length(obj.Clusters), 1);

            for i = 1:length(obj.Clusters)
                % find all edges between core points in the cluster
                clusterCorePoints = intersect(obj.Clusters{i}, obj.CorePoints);
                G = subgraph(obj.KnnGraph, clusterCorePoints);
                [sourceNodesTmp, targetNodesTmp] = findedge(G);
                
                % If a cluster only has one core point, we can't compute it's
                % density; consequently, we must check if there is an edge in
                % the cluster before trying to compute it's density
                if sourceNodesTmp
                    % findedge returns a source and target node for each edge;
                    % to reduce the number of iterations in the following for 
                    % loop, we transform the source and target nodes such that
                    % there is one cell of target nodes for each source node
                    sourceNodes = unique(sourceNodesTmp);
                    targetNodes = mat2cell(targetNodesTmp, ...
                        histcounts(sourceNodesTmp, [sourceNodes; inf]));
                
                    for nodeId = 1:length(sourceNodes)
                        % convert node IDs back to their original indices
                        origNodeId = clusterCorePoints(nodeId);
                        origNeighborIds = clusterCorePoints(targetNodes{nodeId});

                        distances = pairwiseDist(obj.Data(origNodeId, :), ...
                            obj.Data(origNeighborIds, :));  
                        maxDist = max(distances);

                        if maxDist > obj.ClusterDensities(i)
                            obj.ClusterDensities(i) = maxDist;
                        end
                    end
                else
                    obj.ClusterDensities(i) = nan;
                end
            end
        end
    end
end

function d = pairwiseDist(X, Y)
    d = sqrt(bsxfun(@plus, sum(X.^2, 2), sum(Y.^2, 2)') - 2*(X * Y'));
end