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
import pandas as pd
import sklearn
from sklearn import decomposition
from operator import itemgetter

from .utils import list_flatten


def fitEllipseToBoundary(bounds):
    x = [bounds[i][0] for i in range(len(bounds))]
    y = [bounds[i][1] for i in range(len(bounds))]
    x_coord = [bounds[i][4] for i in range(len(bounds))]
    y_coord = [bounds[i][5] for i in range(len(bounds))]

    # major_axis = np.zeros([len(x), 2])
    # minor_axis = np.zeros([len(x), 2])
    major_axis = []
    minor_axis = []
    for i in range(len(x)):
        xy_i = list(map(list, zip(x[i], y[i])))
        pca = sklearn.decomposition.PCA()
        pca.fit(np.array(xy_i))
        # print(pca.components_)
        comp = pca.components_  ### +/- different with MATLAB +????
        major_axis.append(comp[:, 0])
        minor_axis.append(comp[:, 1])
    major_axis = np.array(major_axis)
    minor_axis = np.array(minor_axis)
    # print()

    return major_axis, minor_axis


def construct_ccgs_optimized(bounds, alpha, r):
    # % constructs subgraphs using parameters alpha and r
    # % option to select ccg edge connection using object boundary or centroid
    # % eg. (option = 'boundary')
    params = {}
    params['dist'] = 'centroid'

    x = [bounds[i][0] for i in range(len(bounds))]
    y = [bounds[i][1] for i in range(len(bounds))]
    x_coord = [bounds[i][4] for i in range(len(bounds))]
    y_coord = [bounds[i][5] for i in range(len(bounds))]

    X = list(map(list, zip(x_coord, y_coord)))
    ## cal the distance matrix
    D = scipy.spatial.distance.pdist(np.array(X), metric='euclidean')
    # eucl(X, X)
    params['dist'] = 'centroidbased'

    ## % probability matrix
    P = D**(-alpha)
    mat_prob = P > r
    mat_prob = scipy.spatial.distance.squareform(mat_prob)

    ## %define edges
    mat_True = np.full([len(X), len(X)], True)
    edges = np.triu(
        mat_True, 1
    ) & mat_prob  ###here & means np.logical_and(A, B), A,B np.array. BUT & didn't work if A and B are the scaler.

    ## % get edge locations
    xx, yy = np.nonzero(edges * 1)  ###sort along with xx
    xxyy = np.array([xx, yy]).transpose().tolist()
    xxyy = sorted(xxyy,
                  key=lambda l: l[1])  ## , reverse=True ###sort alng with yy
    xx_, yy_ = np.array(xxyy).transpose().tolist()
    VX = [[x_coord[xx_[i]] for i in range(len(xx_))],
          [x_coord[yy_[i]] for i in range(len(yy_))]]
    VY = [[y_coord[xx_[i]] for i in range(len(xx_))],
          [y_coord[yy_[i]] for i in range(len(yy_))]]
    # VX = [ [ x_coord[xx[i]], x_coord[yy[i] ] ] for i in range(len(xx))]
    # VY = [ [ y_coord[xx[i]], y_coord[yy[i] ] ] for i in range(len(xx))]

    # ## % get node locations
    # idx = np.array([xx, yy])
    # idx = idx.transpose()
    x = [x_coord[xx[i]]
         for i in range(len(xx))] + [x_coord[yy[i]] for i in range(len(yy))]
    y = [y_coord[xx[i]]
         for i in range(len(xx))] + [y_coord[yy[i]] for i in range(len(yy))]

    params['r'] = r
    params['alpha'] = alpha
    # print()
    return VX, VY, x, y, edges, params


def haralick_no_img(SGLD):
    feats = [
        'contrast_energy', 'contrast_inverse_moment', 'contrast_ave',
        'contrast_var', 'contrast_entropy', 'intensity_ave',
        'intensity_variance', 'intensity_entropy', 'entropy', 'energy',
        'correlation', 'information_measure1', 'information_measure2'
    ]
    ### %%% Calculate Statistics %%%%
    ###      MATLAB       [pi,pj,p] = find(SGLD>(eps^2));  ###bj ??? bug?? p is not the value of the position [pi, pj], but 1. Since SGLD is {0, 1} matrix.
    pi, pj = np.where(SGLD > (np.finfo(float).eps)**2)
    # p = [SGLD[pi[i], pj[i]] for i in range(len(pi))]   ###????
    p = SGLD[pi, pj]
    p = p / np.sum(p)
    # p = [1/len(pi)]*len(pi)

    if len(p) <= 1:
        return feats
    ### %marginal of x
    px_all = np.sum(SGLD, axis=1)
    pxi = np.where(px_all > (np.finfo(float).eps)**2)
    # px = [px_all[pxi[i]] for i in range(len(pxi))] ##?????
    px = px_all[pxi]
    px = px / np.sum(px)
    # px = [1/len(pxi[0])]*len(pxi[0])

    ### %marginal of y
    py_all = np.sum(SGLD, axis=0)
    pyi = np.where(py_all > (np.finfo(float).eps)**
                   2)  ###same function [pyi,junk,py] = find(py_all>(eps^2));
    # py = [px_all[pyi[i]] for i in range(len(pyi))]   ##????
    py = py_all[pyi]
    py = py / np.sum(py)
    # py = [1/len(pyi[0])]*len(pyi[0])

    ### %%% Calculate Contrast Features %%%%
    all_contrast = np.abs(pi - pj).tolist()
    # sorted_contrast = np.sort(all_contrast)
    sind, sorted_contrast = zip(
        *sorted(enumerate(all_contrast), key=itemgetter(1)))
    # ind = [find(diff(sorted_contrast)); length(all_contrast)]
    # contrast = sorted_contrast(ind)
    # pcontrast = cumsum(p(sind))
    # pcontrast = diff([0; pcontrast(ind)])
    sorted_contrast = np.array(sorted_contrast)
    diff = sorted_contrast[
        1:] - sorted_contrast[:-1]  ###same function with diff in MATLAB
    ind1 = np.where(diff > (np.finfo(float).eps)**2)
    ind2 = len(all_contrast) - 1
    ind = ind1[0].tolist() + [
        ind2
    ]  ### MATLAB ind = [find(diff(sorted_contrast)); length(all_contrast)];
    contrast = [sorted_contrast[ind[i]] for i in range(len(ind))]
    pcontrast = np.cumsum([p[sind_i] for sind_i in sind])
    pcontrast_ = np.array([0] + [pcontrast[ind_i] for ind_i in ind])
    pcontrast = pcontrast_[1:] - pcontrast_[:-1]

    # contrast = [0, 8]
    # pcontrast = [2/3, 1/3]
    contrast = np.array(contrast)
    pcontrast = np.array(pcontrast)
    contrast_energy = np.sum(contrast**2 * pcontrast)
    contrast_inverse_moment = np.sum((1 / (1 + contrast**2)) * pcontrast)
    contrast_ave = np.sum(contrast * pcontrast)
    contrast_var = np.sum((contrast - contrast_ave)**2 * pcontrast)
    contrast_entropy = -np.sum(pcontrast * np.log(pcontrast))

    ##%%% Calculate Intensity Features %%%%
    # pi = np.array([0,0,8])
    # pj = np.array([0,8,8])
    all_intensity = (pi + pj) / 2
    sind, sorted_intensity = zip(
        *sorted(enumerate(all_intensity), key=itemgetter(1)))
    sorted_intensity = np.array(sorted_intensity)
    diff = sorted_intensity[
        1:] - sorted_intensity[:-1]  ###same function with diff in MATLAB
    ind1 = np.where(diff > (np.finfo(float).eps)**2)
    ind2 = len(all_intensity) - 1
    ind = ind1[0].tolist() + [
        ind2
    ]  ### MATLAB ind = [find(diff(sorted_intensity)); length(all_contrast)];
    intensity = [sorted_intensity[ind[i]] for i in range(len(ind))]
    pintensity = np.cumsum([p[sind_i] for sind_i in sind])
    pintensity_ = np.array([0] + [pintensity[ind_i] for ind_i in ind])
    pintensity = pintensity_[1:] - pintensity_[:-1]

    intensity_ave = np.sum(intensity * pintensity)
    intensity_variance = np.sum((intensity - intensity_ave)**2 * pintensity)
    intensity_entropy = -np.sum(pintensity * np.log(pintensity))

    ###%%% Calculate Probability Features %%%%
    # p = np.array([1/3,1/3,1/3])
    p_ = np.array(p)
    entropy = -np.sum(p_ * np.log(p_))
    energy = np.sum(p_ * p_)

    ### %%% Calculate Correlation Features %%%%
    # px = np.array([1/2, 1/2])
    # pxi = np.array([0, 8])
    # py = np.array([1/2, 1/2])
    # pyi = np.array([0, 8])
    # pi = np.array([0,0,8])
    # pj = np.array([0,8,8])

    px_ = np.array(px)
    py_ = np.array(py)
    mu_x = np.sum(pxi * px_)
    sigma_x = np.sqrt(np.sum((pxi - mu_x)**2 * px))
    mu_y = np.sum(pyi * py_)
    sigma_y = np.sqrt(np.sum((pyi - mu_y)**2 * py))

    if sigma_x == 0 or sigma_y == 0:
        print('Zero standard deviation.')
    else:
        correlation = np.sum(
            (pi - mu_x) * (pj - mu_y) * p) / (sigma_x * sigma_y)

    ###%%% Calculate Information Features %%%%
    # px_all = np.array([2/3, 0,0,0,0,0,0,0, 1/3,0,0,0,0,0,0,0,0,0])
    # py_all = np.array([1/3, 0,0,0,0,0,0,0, 2/3,0,0,0,0,0,0,0,0,0])
    [px_grid, py_grid] = np.meshgrid(px, py)
    [log_px_grid, log_py_grid] = np.meshgrid(np.log(px), np.log(py))
    h1 = -np.sum(p * np.log(px_all[pj] * py_all[pi]))
    h2 = -np.sum(px_grid * py_grid * (log_px_grid + log_py_grid))
    hx = -np.sum(px * np.log(px))
    hy = -np.sum(py * np.log(py))

    information_measure1 = (entropy - h1) / np.max([hx, hy])
    information_measure2 = np.sqrt(1 - np.exp(-2 * (h2 - entropy)))
    # if 1 - np.exp(-2*(h2 - entropy)) < 0:
    #     print()

    return [
        contrast_energy, contrast_inverse_moment, contrast_ave, contrast_var,
        contrast_entropy, intensity_ave, intensity_variance, intensity_entropy,
        entropy, energy, correlation, information_measure1,
        information_measure2
    ]


def extract_CGT_feats(bounds, a=0.5, r=0.2):
    info = {}
    info['alpha'] = a
    info['radius'] = r
    info['angle_bin_size'] = 10  ##% default: 10 degree bins
    info['angular_adjust'] = 0  ##% default: sets <1,0> at 0 degrees

    ##########got the different results SINCE MATLAB PCA module ????
    axis, minor_axis = fitEllipseToBoundary(bounds)  ###??
    # # % adjust vectors to all point up
    for j in range(len(axis)):
        if axis[j, 1] < 0:
            axis[j, :] = -axis[j, :]

    # ##test
    # axis = sio.loadmat('axis.mat')
    # axis = axis['axis']

    angle_degrees = 180 / np.pi * np.arctan(axis[:, 1] / axis[:, 0]) + 90
    angle_degrees = angle_degrees + info['angular_adjust']

    # %% entropy of co-occurence
    w = info['angle_bin_size']  ## % width of bin
    # %discretize angles to every d degrees
    discrete_angles = np.floor(angle_degrees / w) * w

    # % account for greater than or equal to 180
    discrete_angles[
        discrete_angles >= 180] = discrete_angles[discrete_angles >= 180] - 180

    # % build graph
    alpha = info['alpha']
    r = info['radius']
    VX, VY, x, y, edges, params = construct_ccgs_optimized(bounds, alpha, r)
    # mat_contents = sio.loadmat(mat_fname)

    # # % initialize co-occurence
    bin = list(range(0, 180, w))  ##% discretizations!!!
    # ## %% based on number of neighborhoods rather than number of bounds
    # for j in range(len(bounds)-1):
    #     for k in range(j+1, len(bounds)):
    #         edges[k,j] = edges[j,k]
    # edges = edges or edges.transpose()

    # % find gland networks
    # graphconncomp()
    graph = scipy.sparse.csr_matrix(edges)
    # print(graph)
    numcomp, group = scipy.sparse.csgraph.connected_components(
        csgraph=graph, directed=False, return_labels=True)

    ## % define a neighborhood for each gland network (number of neighborhoods = number of networks)
    c = []
    feats = []
    for ii in range(np.max(group)):
        p = np.zeros([int(180 / w), int(180 / w)])
        neighborhood_angles = discrete_angles[
            group ==
            ii]  ##% aggregate angles in gland network  ###bj different with MATLAB because the
        # print('ii',ii)
        for jj in range(len(bin)):
            for kk in range(jj, len(bin)):
                if np.sum(np.isin(neighborhood_angles, [bin[jj]])) and np.sum(
                        np.isin(neighborhood_angles, [bin[kk]])):
                    if jj != kk:
                        p[jj, kk] = np.sum(
                            np.isin(neighborhood_angles, [bin[jj]])) * np.sum(
                                np.isin(neighborhood_angles, [bin[kk]]))
                    else:
                        p[jj,
                          kk] = np.sum(np.isin(neighborhood_angles, [bin[jj]]))
        c_ = p / np.sum(p)
        c.append(c_)  ##% normalize co-occurence matrix
        feat = haralick_no_img(c_)
        feats.append(feat)
    temp_network, bin_edges = np.histogram(group, bins=numcomp)

    group_ind = np.where(temp_network > 1)
    group_ind = group_ind[0]

    ### % remove networks that don't give co-occurrence features
    for k in range(len(feats[0])):
        for i in range(len(feats)):
            if type(feats[i][0]) == str:
                # group_ind_ = group_ind==i
                group_ind_i = np.where(group_ind == i)
                # # group_ind_i = group_ind_i[0].tolist()
                # # group_ind.pop(group_ind_i[0])
                # group_ind[group_ind==i] = -1
                group_ind = np.delete(group_ind, group_ind_i)
    # np.sort(group_ind)
    newgroup_ind, origgroup_ind = zip(
        *sorted(enumerate(group_ind), key=itemgetter(1)))
    num_connected_comp = max(newgroup_ind)

    network = np.zeros(len(discrete_angles))
    for j in range(num_connected_comp):
        network[group == origgroup_ind[j]] = newgroup_ind[j] + 1

    ## % replicate feats based on the size of network
    ## %netfeats = feats;
    networksize = []
    for j in range(np.max(group)):
        networksize.append(np.sum(group == j))

    ## %% mean and standard deviation across bounds for each haralick feature
    CGT = []
    for k in range(len(feats[0])):
        feat = []
        n = 0
        for i in range(len(feats)):
            if type(feats[i][0]) != str:
                n = n + 1
                feat.append(np.tile(feats[i][k], (1, networksize[i])).tolist())
        # if np.
        #     print('iiiiiiiiii', i)

        feat = list_flatten(list_flatten(feat))
        if n > 0:
            CGT.append([
                np.mean(feat),
                np.std(feat, ddof=1),
                np.max(feat) - np.min(feat)
            ])
        else:
            CGT.append([0, 0, 0])

    info['alpha'] = alpha
    info['radius'] = r
    info['angle_bin_size'] = w

    CGT = list_flatten(CGT)

    return CGT, c, info, feats, network, edges
