from scipy import io as sio
import os
import numpy as np
import imageio

feat_dir = './sample'
area_dir = os.path.join(feat_dir, 'area.mat')
area = sio.loadmat(area_dir)['area'].transpose()

centroids_dir = os.path.join(feat_dir, 'centroids.mat')
centroids = sio.loadmat(centroids_dir)['centroids']

intensity_dir = os.path.join(feat_dir, 'mean_intensity.mat')
mean_intensity = sio.loadmat(intensity_dir)['mean_intensity'].transpose()
feat = np.hstack([centroids, area, mean_intensity])

img_dir = os.path.join(feat_dir, 'cropped.png')
img = imageio.imread(img_dir)


def load_dummy_data():
    return img, feat
