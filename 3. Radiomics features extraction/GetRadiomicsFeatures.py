import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
import cv2
from radiomics import featureextractor
import argparse
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--imgDir', type=str, default="", help="patients CT folder dir")
parser.add_argument('--saveDir', type=str, default="", help="radiomics features file dir")
args = parser.parse_args()

def load_correct_roi_Array(path):
    file_list = os.listdir(path)
    for file in file_list:
        if len(file) >= 20:
            result_path = os.path.join(path, file)
            seg = sitk.ReadImage(result_path)
            return seg

def loadImgArraywithID(fold, iden):
    imgPath = os.path.join(fold, iden + ".nii.gz")
    img = sitk.ReadImage(imgPath)
    return img

def get_only_maximum_cross_sectional_area_roi(roi_array):
    depth, width, length = np.shape(roi_array)
    maximum_area = 0
    maximum_area_depth = 0
    zero_array = np.zeros((width, length))
    for d in range(depth):
        singal_roi = roi_array[d, :, :]
        if not (singal_roi == zero_array).all():
            contours, hierarchy = cv2.findContours(singal_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            area = cv2.contourArea(contours[0])
            if area > maximum_area:
                maximum_area = area
                maximum_area_depth = d
    new_roi = np.zeros((depth, width, length))
    new_roi[maximum_area_depth, :, :] = roi_array[maximum_area_depth, :, :]
    return new_roi

def extract_one_patient_radiomics(imgDir, ind, featureDict, img_thickness_list):
    path = os.path.join(imgDir, str(ind))

    img = loadImgArraywithID(path, ind)
    img_thickness_list.append(img.GetSpacing()[2])
    file_list = os.listdir(path)

    params = './Paramsescc.yaml'

    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    dictkey = []
    feature = []
    file_list.sort()
    for roi in file_list:
        if "seg" not in roi and "nodule" not in roi:
            continue
        mask = sitk.ReadImage(os.path.join(path, roi))
        mask.SetOrigin(img.GetOrigin())
        result = extractor.execute(img, mask)
        key = list(result.keys())
        if '_seg.nii.gz' not in roi:
            key = [roi[:-13] + item for item in key]
            key = key[47:]
            for jind in range(len(key)):
                feature.append(result[key[jind][key[jind].find('mm_') + 3:]])
            dictkey.extend(key)
        else:
            key = key[47:]
            for jind in range(len(key)):
                feature.append(result[key[jind]])
            dictkey.extend(key)

    featureDict[ind] = feature

    print(ind)
    return dictkey, feature



if __name__ == '__main__':
    imgDir = args.imgDir
    target_file = args.saveDir
    pool = mp.Pool(processes=10)
    dirlist = os.listdir(imgDir)[:]
    save_path = target_file
    # Feature Extraction
    featureDict = {}
    img_thickness_list = []
    for ind in tqdm(dirlist):
        path = os.path.join(imgDir, str(ind))
        img = loadImgArraywithID(path, ind)
        img_thickness_list.append(img.GetSpacing()[2])
        file_list = os.listdir(path)
        params = './Paramsescc.yaml'
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        dictkey = []
        feature = []
        file_list.sort()
        for roi in file_list:
            if "seg" not in roi and "nodule" not in roi:
                continue
            mask = sitk.ReadImage(os.path.join(path, roi))
            mask.SetOrigin(img.GetOrigin())
            result = pool.apply_async(extractor.execute, (img, mask, ))
            result = result.get()
            key = list(result.keys())
            if '_seg.nii.gz' not in roi:
                key = [roi[:-13] + item for item in key]
                key = key[47 + 17:]
                for jind in range(len(key)):
                    feature.append(result[key[jind][key[jind].find('mm_') + 3:]])
                dictkey.extend(key)
            else:
                key = key[47:]
                for jind in range(len(key)):
                    feature.append(result[key[jind]])
                dictkey.extend(key)
        featureDict[ind] = feature
        print(ind)

    dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
    dataframe.insert(0, 'Thickness', img_thickness_list)
    dataframe.to_csv(save_path)

