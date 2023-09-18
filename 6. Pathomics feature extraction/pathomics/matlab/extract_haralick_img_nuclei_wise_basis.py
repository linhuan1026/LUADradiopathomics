from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage import feature
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
import scipy


def MinMaxScaler_v2(X, GrayLimits=[100, 150], Num_level=8):

    slope = Num_level / (GrayLimits[1] - GrayLimits[0])
    intercept = -slope * (GrayLimits[0])
    X_scaled = slope * X + intercept

    return X_scaled


def Lharalick_img_nuclei_wise_basis(fname_mask='test_dense.png',
                              fname_intensity='test_intensity_image.png'):
    # fname_mask = 'test_dense.png'
    #image=Image.open(fname_mask)
    # image.show()
    # fname_intensity = 'test_intensity_image.png'

    #img=np.asarray(image)
    #pres = np.max(img*1.0)/2
    mask = fname_mask  #img>pres
    labels = measure.label(mask)

    #image_intensity = Image.open(fname_intensity).convert('L')
    image_intensity = Image.fromarray(fname_intensity).convert('L')
    image_intensity = np.array(image_intensity)
    props = measure.regionprops(label_image=labels,
                                intensity_image=image_intensity)

    feats_names = [
        'contrast_energy', 'contrast_inverse_moment', 'contrast_ave',
        'contrast_var', 'contrast_entropy', 'intensity_ave',
        'intensity_variance', 'intensity_entropy', 'entropy', 'energy',
        'correlation', 'information_measure1', 'information_measure2'
    ]

    feats = []

    for prop in props:
        # print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
        Gray_lim_max = prop['intensity_max']
        Gray_lim_min = prop['intensity_min']
        mask_label_nuclei = prop['image']
        org_nuclei_only = prop[
            'image_intensity']  ### 'image_intensity' did not give the value of the region outside of nuclei
        min_row, min_col, max_row, max_col = prop['bbox']
        org_nuclei = image_intensity[min_row:max_row, min_col:max_col]
        org_nuclei = np.clip(org_nuclei, Gray_lim_min, Gray_lim_max)
        org_nuclei_processed = MinMaxScaler_v2(org_nuclei,
                                               [Gray_lim_min, Gray_lim_max],
                                               8 - 1).astype(int)

        # org_nuclei_processed = MinMaxScaler((0,8-1)).fit_transform(org_nuclei).astype(int)

        SGLD = feature.graycomatrix(image=org_nuclei_processed,
                                    distances=[1],
                                    angles=[0],
                                    levels=8,
                                    symmetric=True)
        # # print(org_nuclei_processed.shape)

        SGLD = np.squeeze(SGLD)
        SGLD = np.double(SGLD)  #float16 may encounter error
        SGLD = SGLD / np.sum(SGLD)

        ### %%% Calculate Statistics %%%%
        pi, pj = np.where(SGLD[:, :] > 0)
        p = SGLD[pi, pj]
        p = p / np.sum(p)

        ####%marginal of x
        px_all = np.sum(SGLD, 1)
        pxi = np.where(px_all > 0)
        px = px_all[pxi]
        px = px / np.sum(px)

        ####%marginal of y
        py_all = np.sum(SGLD, 0)
        pyi = np.where(py_all > 0)
        py = py_all[pyi]
        py = py / np.sum(py)

        #### %%% Calculate Contrast Features %%%%
        all_contrast = np.abs(pi - pj)
        sorted_contrast = np.sort(all_contrast)
        sind = np.argsort(all_contrast)
        ind = np.where(sorted_contrast[1:] - sorted_contrast[:-1])
        ind = np.append(ind, len(all_contrast) - 1)
        contrast = sorted_contrast[ind]
        pcontrast = np.cumsum(p[sind])
        pcontrast = np.append(0, pcontrast[ind])
        pcontrast = pcontrast[1:] - pcontrast[:-1]

        contrast_energy = np.sum(contrast**2 * pcontrast)
        contrast_inverse_moment = np.sum((1 / (1 + contrast**2)) * pcontrast)
        contrast_ave = np.sum(contrast * pcontrast)
        contrast_var = np.sum((contrast - contrast_ave)**2 * pcontrast)
        # contrast_entropy = -np.sum( pcontrast*np.log(pcontrast) )   ####exit log(0)
        contrast_entropy = -np.sum(pcontrast * np.log(
            pcontrast, out=np.zeros_like(pcontrast),
            where=(pcontrast != 0)))  ####note:: exit log(0)

        #### %%% Calculate Intensity Features %%%%
        all_intensity = (pi + pj) / 2
        sorted_intensity = np.sort(all_intensity)
        sind = np.argsort(all_intensity)
        ind = np.where(sorted_intensity[1:] - sorted_intensity[:-1])
        ind = np.append(ind, len(all_intensity) - 1)
        intensity = sorted_intensity[ind]
        pintensity = np.cumsum(p[sind])
        pintensity = np.append(0, pintensity[ind])
        pintensity = pintensity[1:] - pintensity[:-1]

        intensity_ave = np.sum(intensity * pintensity)
        intensity_variance = np.sum(
            (intensity - intensity_ave)**2 * pintensity)
        intensity_entropy = -np.sum(pintensity * np.log(pintensity))

        #### %%% Calculate Probability Features %%%%
        entropy = -np.sum(p * np.log(p))
        energy = np.sum(p * p)

        #### %%% Calculate Correlation Features %%%%
        mu_x = np.sum(pxi * px)
        sigma_x = np.sqrt(np.sum((pxi - mu_x)**2 * px))
        mu_y = np.sum(pyi * py)
        sigma_y = np.sqrt(np.sum((pyi - mu_y)**2 * py))

        if sigma_x == 0 or sigma_y == 0:
            warnings.warn('Zero standard deviation.')
        else:
            correlation = np.sum(
                (pi - mu_x) * (pj - mu_y) * p) / (sigma_x * sigma_y)

        #### %%% Calculate Information Features %%%%
        [px_grid, py_grid] = np.meshgrid(px, py)
        [log_px_grid, log_py_grid] = np.meshgrid(np.log(px), np.log(py))
        h1 = -np.sum(p * np.log(px_all[pj] * py_all[pi]))
        h2 = -np.sum(px_grid * py_grid * (log_px_grid + log_py_grid))
        hx = -np.sum(px * np.log(px))
        hy = -np.sum(py * np.log(py))

        information_measure1 = (entropy - h1) / np.max([hx, hy])
        information_measure2 = np.sqrt(1 - np.exp(-2 * (h2 - entropy)))

        feats.append([
            contrast_energy, contrast_inverse_moment, contrast_ave,
            contrast_var, contrast_entropy, intensity_ave, intensity_variance,
            intensity_entropy, entropy, energy, correlation,
            information_measure1, information_measure2
        ])


    feats = np.array(feats)

    return feats, feats_names


# fname_mask = 'test_dense.png'
# fname_intensity = 'test_intensity_image.png'

# featvals, feats_names = Lharalick_img_nuclei_wise(fname_mask=fname_mask, fname_intensity=fname_intensity)
# print(featvals)
# print()
