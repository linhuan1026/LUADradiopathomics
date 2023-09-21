import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import argparse
import math
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--imgDir', type=str, default="", help="patients CT folder dir")

args = parser.parse_args()

def make_binary_dilation_mask(img, vertical_distance):
    zero_arr = np.zeros((512,512))
    mask = np.zeros_like(img)
    flag = False
    min = 1000
    max = 0
    for slice in range(img.shape[0]):
        if (img[slice, :, :] != zero_arr).any() and flag==False:
            if slice<min:
                min = slice
            if slice>max:
                max=slice
    if min-vertical_distance<0:
        min = 0
    else:
        min=min-vertical_distance
    if max+vertical_distance+1>=img.shape[0]:
        max = img.shape[0]
    else:
        max = max+vertical_distance+1
    for slice in range(min, max):
        mask[slice, :, :] = np.ones_like(mask[slice, :, :])
    return mask

def convert_roi_to_circle(img1, img2, img3, img4):
    mask1 = np.zeros_like(img1)
    mask2 = np.zeros_like(img1)
    mask3 = np.zeros_like(img1)
    mask4 = np.zeros_like(img1)
    for slice in tqdm(range(img1.shape[0])):
        # if slice != 318:
        #     continue
        if (img1[slice, :, :] != mask1).any():
            contours, hierarchy = cv.findContours(img1[slice, :, :], cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_NONE)
            rect1 = cv.minAreaRect(contours[0])
            center1 = rect1[0]
            radius1 = math.sqrt(rect1[1][0]**2 + rect1[1][1]**2)
            cv.circle(mask1, center1, int(radius1))
        if (img2[slice, :, :] != mask2).any():
            contours, hierarchy = cv.findContours(img2[slice, :, :], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            rect2 = cv.minAreaRect(contours[0])
            center2 = rect2[0]
            radius2 = math.sqrt(rect2[1][0]**2 + rect2[1][1]**2)
            cv.circle(mask2, center2, int(radius2))
        if (img3[slice, :, :] != mask3).any():
            contours, hierarchy = cv.findContours(img3[slice, :, :], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            rect3 = cv.minAreaRect(contours[0])
            center3 = rect3[0]
            radius3 = math.sqrt(rect3[1][0]**2 + rect3[1][1]**2)
            cv.circle(mask3, center3, int(radius3))
        if (img4[slice, :, :] != mask4).any():
            contours, hierarchy = cv.findContours(img4[slice, :, :], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            rect4 = cv.minAreaRect(contours[0])
            center4 = (int(rect4[0][0]),int(rect4[0][1]))
            radius4 = math.sqrt(rect4[1][0]**2 + rect4[1][1]**2)
            cv.circle(mask4, center4, int(radius4), (255,255,255), -1)
    return mask1, mask2, mask3, mask4

def loadSegArraywithID(fold,iden):
    segPath = os.path.join(fold, "nodule.nii.gz")
    seg = sitk.ReadImage(segPath)
    return seg

def setimageapcing(fold,iden,spacing,thickness):
    segPath = os.path.join(fold, iden + ".nrrd")
    image = sitk.ReadImage(segPath)
    image.SetSpacing([spacing, spacing, thickness])
    sitk.WriteImage(image,segPath)

def loadlungArraywithID(fold):
    lungPath = os.path.join(fold, "lung.nii.gz")
    lung = sitk.ReadImage(lungPath)
    return lung


def convert_distance_to_pixel(distence,spacing):
    pixel = int(distence/spacing)
    return pixel

# Load batch file
if __name__ == '__main__':
    imgDir = args.imgDir
    celibration = imgDir
    dirlist = os.listdir(imgDir)[:]

    for ind in tqdm(dirlist):
        path = os.path.join(imgDir, str(ind))
        celibration_path = os.path.join(celibration, str(ind))
        print(ind)

        img = sitk.ReadImage(path + "/{}.nii.gz".format(ind))
        [spacing, _, thickness] = img.GetSpacing()

        nodule = sitk.ReadImage(path + "/{}_seg.nii.gz".format(ind))
        nodule = sitk.GetArrayFromImage(nodule)

        rectify_roi = sitk.ReadImage(path + "/roi_rectify.nii.gz")
        rectify_roi = sitk.GetArrayFromImage(rectify_roi)

        mask_20 = sitk.ReadImage(path + "/peritumoral_20mm_nodule.nii.gz")
        mask_20 = sitk.GetArrayFromImage(mask_20)
        mask_15 = sitk.ReadImage(celibration_path + "/peritumoral_15mm_nodule.nii.gz")
        mask_15 = sitk.GetArrayFromImage(mask_15)
        mask_10 = sitk.ReadImage(celibration_path + "/peritumoral_10mm_nodule.nii.gz")
        mask_10 = sitk.GetArrayFromImage(mask_10)
        mask_5 = sitk.ReadImage(celibration_path + "/peritumoral_05mm_nodule.nii.gz")
        mask_5 = sitk.GetArrayFromImage(mask_5)

        new_mask_20 = mask_20 * rectify_roi
        new_mask_15 = mask_15 * rectify_roi
        new_mask_10 = mask_10 * rectify_roi
        new_mask_5 = mask_5 * rectify_roi

        new_mask_15 = sitk.GetImageFromArray(new_mask_15)
        new_mask_15.SetSpacing([spacing, spacing, thickness])
        new_mask_10 = sitk.GetImageFromArray(new_mask_10)
        new_mask_10.SetSpacing([spacing, spacing, thickness])
        new_mask_5 = sitk.GetImageFromArray(new_mask_5)
        new_mask_5.SetSpacing([spacing, spacing, thickness])
        new_mask_20 = sitk.GetImageFromArray(new_mask_20)
        new_mask_20.SetSpacing([spacing, spacing, thickness])

        sitk.WriteImage(new_mask_5, path + '/peritumoral_05mm_nodule.nii.gz')
        sitk.WriteImage(new_mask_10, path + '/peritumoral_10mm_nodule.nii.gz')
        sitk.WriteImage(new_mask_15, path + '/peritumoral_15mm_nodule.nii.gz')
        sitk.WriteImage(new_mask_20, path + '/peritumoral_20mm_nodule.nii.gz')

