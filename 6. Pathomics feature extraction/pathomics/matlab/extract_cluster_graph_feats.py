# coding: utf-8
# from msilib.schema import Component
from re import X
from turtle import distance
from uuid import RESERVED_FUTURE
import numpy as np
import random
import os
# from scipy import io
import scipy
from scipy import spatial
import scipy.io as sio
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sklearn
from sklearn import decomposition
from operator import itemgetter


def cluster_graph_features_optimized(bounds, edges):
    x = [bounds[i][0] for i in range(len(bounds))]
    y = [bounds[i][1] for i in range(len(bounds))]
    x_coord = [bounds[i][4] for i in range(len(bounds))]
    y_coord = [bounds[i][5] for i in range(len(bounds))]

    pathlengths = []
    pathlengths_nonempty = []

    if len(x_coord) == 1:
        print('\nNot enough bounds to compute Cluster Graphs!')
        N = 1
        pathlengths.append(0)
        pathlengths_nonempty.append(0)
        eccentricity = 0
    else:
        ## %1) Number of Nodes

        N = len(x_coord)
        # feats.append(N)
        # feature_list.append('Number of Nodes')

        ## %%% Eccentricity calculation
        ## % generate distance matrix
        X = list(map(list, zip(x_coord, y_coord)))
        D = scipy.spatial.distance.pdist(np.array(X), metric='euclidean')
        D = scipy.spatial.distance.squareform(D) + np.eye(len(X))

        ## % create sparse distance-weighted edge matrix
        edges = np.triu(edges)  ##, 1   ###% force edges to be upper triangular
        sym_edges = edges | edges.T  ## % full symmetric matrix
        weighted = sym_edges * D
        weighted = scipy.sparse.csr_matrix(weighted)  ##sparse

        for i in range(N):
            distance = scipy.sparse.csgraph.shortest_path(weighted,
                                                          indices=i,
                                                          method='D')
            distance_ = distance[np.isfinite(distance)]
            distance_indices = np.nonzero(distance_)
            pathlengths.append(distance_[distance_indices])
        pathlengths_nonempty = [pa.tolist() for pa in pathlengths]
        pathlengths_nonempty = [pa for pa in pathlengths_nonempty if pa]
        ## % all non-zero pathlengths
        eccentricity = [max(pa) for pa in pathlengths_nonempty]

    feats = []
    feature_list = []
    ## %1) Number of Nodes
    ## %N = length(edges);
    N = len(x_coord)
    feats.append(N)
    feature_list.append('Number of Nodes')

    ## %2) Number of edges
    E = np.sum(edges * 1)
    feats.append(E.item())
    feature_list.append('Number of Edges')

    ## %3) Average Degree
    feats.append(E / N)
    feature_list.append('Average Degree')

    ##%4) Average eccentricity
    feats.append(sum(eccentricity) / N)
    feature_list.append('Average Eccentricity')

    if len(pathlengths[0]) < 1:
        pathlengths_ = [pa.tolist() for pa in pathlengths]
        pathlengths_ = [pa for pa in pathlengths_ if pa]
        if len(pathlengths_) < 1:
            pathlengths = [
                0,
            ] * N
            print(
                '\n Warning: No edges found! All isolated nodes. Consider not using these features.'
            )
            # pathlengths = []
            # for i in range(N):
            #     pathlengths.append(0)
            #     print('\n Warning: No edges found! All isolated nodes. Consider not using these features.')
    ## %5) Diameter
    diameter = max(eccentricity)
    feats.append(diameter)
    feature_list.append('Diameter')

    ## %6) Radius
    radius = min(eccentricity)
    feats.append(radius)
    feature_list.append('Radius')

    ## % eccentricity for largest 90% of path lengths
    ## %%% CHOOSE ONE OF THE FOLLOWING %%%
    ## %%%%% George's definition of 90% %%%%%
    pathlengths_nonempty90 = [
        pa[:round(0.9 * len(pa))] for pa in pathlengths_nonempty
    ]
    eccentricity90 = [max(pa) for pa in pathlengths_nonempty90 if pa]

    ## %7) Average Eccentricity 90%
    feats.append(sum(eccentricity90) / N)
    feature_list.append('Average Eccentricity 90\%')

    if len(eccentricity90) < 1: eccentricity90 = [0]

    ## %8) Diameter 90%
    diameter90 = max(eccentricity90)
    feats.append(diameter90)
    feature_list.append('Diameter 90\%')

    ## %9) Radius 90%
    radius90 = min(eccentricity90)
    feats.append(radius90)
    feature_list.append('Radius 90\%')

    ## %10) Average Path Length
    feats.append(
        sum([sum(pathle) for pathle in pathlengths_nonempty]) /
        sum([len(pathle) for pathle in pathlengths_nonempty]))
    feature_list.append('Average Path Length')

    ## %% clustering coefficients
    # En = []
    # kn = []
    if 'sym_edges' not in vars():
        En = 0
        kn = 1
    else:
        n_components, network = scipy.sparse.csgraph.connected_components(
            csgraph=sym_edges, directed=False, return_labels=True)
        En = np.zeros(shape=[len(network)])
        kn = np.zeros(shape=[len(network)])
        for n in range(N):
            nodes = np.where(network == n)
            nodes = nodes[0].tolist()
            En[nodes] = np.sum([
                edges[nodes[i], nodes].tolist() for i in range(len(nodes))
            ])  ### equal to MATLAB edges(nodes, nodes))
            kn[nodes] = len(nodes)
            # En.append( np.sum( [edges[nodes[i], nodes].tolist() for i in range(len(nodes))] ) )
            # kn.append( len(nodes))

    ## %11) Clustering Coefficient C
    ## % ratio beteween A: the number of edges between neighbors of node n and B:
    ## % the number of possible edges between the neighbors of node n
    np.seterr(divide='ignore', invalid='ignore'
              )  ###for 'divide by zero' of 'Cn = 2 * En / (kn * (kn-1) ) '
    Cn = 2 * En / (kn * (kn - 1))
    where_are_NaNs = np.isnan(Cn)
    Cn[where_are_NaNs] = 0
    feats.append(np.sum(Cn) / N)
    feature_list.append('Clustering Coefficient C')

    ## %12) Clustering Coefficient E
    ## %Dn(n) = 2*(kn(n) + En(n)) / ( kn(n)* (kn(n)+1) )
    Dn = 2 * (kn + En) / (kn * (kn + 1))
    where_are_NaNs = np.isnan(Dn)
    Dn[where_are_NaNs] = 0
    # Dn( isnan(Dn) ) = 0
    feats.append(sum(Dn) / N)
    feature_list.append('Clustering Coefficient D')

    ## %13) Clustering Coefficient E
    ## % count isolated nodes
    iso_nodes = sum(kn == 1)
    if N == iso_nodes:
        feats.append(np.nan)
        print('All isolated nodes')
    else:
        feats.append(sum(Cn[kn > 1]) / (N - iso_nodes))
    feature_list.append('Clustering Coefficient E')

    ## %14) Number of connected components
    feats.append(len(kn[kn > 1]))
    feature_list.append('Number of connected components')

    ## %15) Giant connected component ratio
    feats.append(np.max(kn) / N)
    feature_list.append('giant connected component ratio')

    ## %16) Average Connected Component Size
    if N == iso_nodes:
        feats.append(1)
    else:
        feats.append(np.mean(kn[kn > 1]))
        feature_list.append('average connected component size')

    ## %17 and 18) Number / Percentage of Isolated Nodes
    feats.append(np.asarray(iso_nodes).item())
    feature_list.append('number isolated nodes')

    feats.append(iso_nodes / N)
    feature_list.append('percentage isolated nodes')

    ## %19 and 20) Number / Percentage of End points
    feats.append(np.sum(kn == 2).item())
    feature_list.append('number end nodes')
    feats.append(sum(kn == 2) / N)
    feature_list.append('percentage end nodes')

    ## % 21 and 22) Number / Percentage of Central points
    feats.append(np.sum(np.array(eccentricity) == radius).item())
    feature_list.append('number central nodes')
    feats.append(np.sum(np.array(eccentricity) == radius) / N)
    feature_list.append('percentage central nodes')

    ## % 23 - 26) Edge length statistics
    if N == iso_nodes:
        feats = feats + [0, 0, 0]
    else:
        edge_lengths = weighted.toarray().flatten(
        )  ###np.toarray is to dense from sparse matrix
        edge_lengths = edge_lengths[edge_lengths !=
                                    0]  ## % remove zero edge lengths

        feats.append(np.sum(edge_lengths) /
                     len(edge_lengths))  ##% mean edge-length
        feats.append(np.std(edge_lengths, ddof=1))  ##% standard deviation
        feats.append(scipy.stats.skew(edge_lengths))  ##% skewness
        feats.append(
            scipy.stats.kurtosis(edge_lengths, axis=0, fisher=False)
        )  ##% kurtosis  ##default using Fisher's definition.. intend to use Pearson’s definition of kurtosis. MATLAB
    feature_list.append('mean edge length')
    feature_list.append('standard deviation edge length')
    feature_list.append('skewness edge length')
    feature_list.append('kurtosis edge length')
    # print()

    return feats, feature_list


def extract_cluster_graph_feats(bounds):
    info = {}
    info['alpha'] = 0.5
    info['radius'] = 0.2
    ##% build graph
    alpha = info['alpha']
    r = info['radius']
    from .extract_CGT_feats import construct_ccgs_optimized
    [VX, VY, x, y, edges, _] = construct_ccgs_optimized(bounds, alpha, r)

    [CCGfeats, feature_list] = cluster_graph_features_optimized(bounds, edges)
    # print()
    return CCGfeats, feature_list
