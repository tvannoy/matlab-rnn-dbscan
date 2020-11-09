classdef RnnDbscan < handle
%RnnDbscan RNN DBSCAN clustering 
%   RNN DBSCAN is a density-based clustering algorithm that uses reverse nearest
%   neighbor counts as an estimate of observation density. It is based upon
%   traversals of the directed k-nearest neighbor graph, and can handle clusters
%   of different densities, unlike DBSCAN.
%
%   RnnDbscan constructor:
%       rnnDbscan = RnnDbscan(X, k)
%
%   Rows of X correspond to observations, and columns correspond to variables.
%   k is the number of nearest neighbors to use when constructing the mutual 
%   k-nearest neighbor graph.
%
%   RnnDbscan properties:
%       Data              - input data X 
%       K                 - number of nearest neighbors 
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

    properties (SetAccess = private)
        % Input data; rows are observations, and columns are variables
        Data (:,:) double
        % Number of nearest neighbors used to construct the knn graph
        K (1,1) double {mustBePositive, mustBeInteger} = 1
        % Directed k-nearest neighbor graph
        KnnGraph (1,1) digraph
        % Cell array of clusters; Clusters{i} contains indices for all points
        % that are in cluster i
        Clusters (:,1) cell
        % Outliers that do not belong to any cluster
        Outliers (1,:) double
        % List of core points, which are points with >= k reverse nearest
        % neighbors. 
        CorePoints (1,:) double
        % Cluster densities, defined as the max distance between 
        % connected core points
        ClusterDensities (1,:) double
        % Cluster labels. Outliers have a label value of -1; all other 
        % clusters have labels starting at 1
        Labels (:,1) double {mustBeInteger} = []
    end

    methods (Access = public)
        function obj = RnnDbscan(X, k)
        %RnnDbscan Construct an RNN DBSCAN object
        %   rnnDbscan = RnnDbscan(X, k) creates an RNN DBSCAN object for input
        %   data X, with k nearest neighbors used in constructing the knn graph.

            obj.Data = X;
            obj.K = k;
            obj.Labels = zeros(size(X, 1), 1);
            obj.KnnGraph = knngraph(X, k);
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

            obj.CorePoints = find(obj.KnnGraph.indegree >= obj.K);

            % creating a subgraph renumbers the node IDs. This is problematic
            % because I am expecting the node IDs to correspond to the data
            % points' indices in the data matrix. To remedy this, the subgraph's
            % node IDs have to be converted back to the original node IDs.
            corePointSubgraph = obj.KnnGraph.subgraph(obj.CorePoints);     

            % get the initial clusters from the connected components.
            % 'OutputForm' = 'cell' returns a cell array where the i-th cell
            % contains all of the node IDs belonging to component i; this 
            % results in a cell array where each cell is a cluster.
            clusters = corePointSubgraph.conncomp('Type', 'weak', 'OutputForm', 'cell');

            % convert the subgraph's node IDs back to the original node IDs that
            % correspond to the data points' indices. 
            obj.Clusters = cellfun(@(c) obj.CorePoints(c), clusters, 'UniformOutput', false);

            for i = 1:length(obj.Clusters)
                obj.Labels(obj.Clusters{i}) = i;
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
                    corePointNeighbors = obj.KnnGraph.successors(obj.Clusters{i}(j));

                    % only add points to a cluster if they do not already belong
                    % to a cluster
                    unclusteredPoints = corePointNeighbors(obj.Labels(corePointNeighbors) == 0);

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
                obj.Labels(borderPoints) = i;
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
            pointIndices = 1:size(obj.Data, 1);
            unclusteredPoints = setdiff(pointIndices, clusteredPoints);

            for unclusteredPoint = unclusteredPoints
                % find k-nearest neighbors of the unclustered point
                neighbors = obj.KnnGraph.successors(unclusteredPoint);

                % determine which neighbors are core points; we do not want to
                % cluster the unclustered point if it is not connected to any 
                % core points
                corePointNeighbors = intersect(neighbors, obj.CorePoints);

                if corePointNeighbors
                    % sort core point neighbors by distance from the
                    % unclustered point
                    [distances, sortIdx] = sort(pdist2(unclusteredPoint, corePointNeighbors));
                    corePointNeighbors = corePointNeighbors(sortIdx);

                    for i = 1:length(corePointNeighbors)
                        % find which cluster the core point belongs to
                        clusterIdx = obj.Labels(corePointNeighbors(i));

                        if distances(i) <= obj.ClusterDensities(clusterIdx)
                            % add unclustered point to cluster
                            obj.Clusters{clusterIdx} = horzcat(obj.Clusters{clusterIdx}, unclusteredPoint); 

                            % point can't be assigned to more than one cluster, 
                            % so stop once it is assigned
                           break;
                        end
                    end
                else
                    % the unclustered point does not have any neighbors that are core points, so it is considered an outlier
                    obj.Outliers = horzcat(obj.Outliers, unclusteredPoint);
                    obj.Labels(unclusteredPoint) = -1;
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
                subgraph = obj.KnnGraph.subgraph(clusterCorePoints);
                [sourceNodes, targetNodes] = subgraph.findedge();

                % since the subgraph renumbers the node IDs, we convert back to
                % the original node IDs so we can properly index into the data
                % matrix. The subgraph's node IDs are the indices of the vector
                % (clusterCorePoints) used to create the subgraph
                sourceNodes = clusterCorePoints(sourceNodes);
                targetNodes = clusterCorePoints(targetNodes);

                % get the data instances for the source and target nodes of 
                % each edge
                sourcePoints = obj.Data(sourceNodes, :);
                targetPoints = obj.Data(targetNodes, :);

                % compute the euclidean distance between all pairs of core 
                % points that have edges between them; since observations are 
                % in rows, the third argument (DIM) of vecnorm is set to 2 to
                % compute the norm along rows, not columns 
                distances = vecnorm(sourcePoints - targetPoints, 2, 2);

                obj.ClusterDensities(i) = max(distances);
            end
        end
    end
end