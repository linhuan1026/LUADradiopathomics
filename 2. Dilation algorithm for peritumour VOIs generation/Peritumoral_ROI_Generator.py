import numpy as np
import scipy
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os
from tqdm import tqdm
import argparse
import heapq
from skimage import measure
from lungmask import LMInferer

parser = argparse.ArgumentParser()
parser.add_argument('--imgDir', type=str, default="", help="patients CT folder dir")
args = parser.parse_args()

def keep_connected_area(img_mat_axial):
    connected_areas = measure.label(img_mat_axial, neighbors=8)
    print(np.max(connected_areas), np.min(connected_areas))
    connected_areas_properties = measure.regionprops(connected_areas)

    ## find the index of the first 1 largest number in the label
    list_areas = [prop.area for prop in connected_areas_properties]
    max_num_index_list = map(list_areas.index, heapq.nlargest(1, list_areas))
    max_num_index_list = list(max_num_index_list)
    print(connected_areas_properties[max_num_index_list[0]].label)

    ## mask all the elements of the index with 1
    first_largest = connected_areas_properties[max_num_index_list[0]].label
    connected_areas_1st = np.zeros_like(connected_areas)
    connected_areas_1st[connected_areas == first_largest] = 1
    return connected_areas_1st

def lungSeg(img_path, pid):
    inferer = LMInferer()
    img = sitk.ReadImage(os.path.join(img_path, pid + ".nii.gz"))
    lung_mask = inferer.apply(img)
    lung_mask = sitk.GetImageFromArray(lung_mask)
    sitk.WriteImage(lung_mask, os.path.join(img_path, "lung.nii.gz"))
    return lung_mask

def make_binary_dilation_mask(img, vertical_distance):
    zero_arr = np.zeros(img[0, :, :].shape)
    mask = np.zeros_like(img)
    flag = False
    min = 1000
    max = 0
    for slice in range(img.shape[0]):
        if (img[slice, :, :] != zero_arr).any() and flag == False:
            if slice < min:
                min = slice
            if slice > max:
                max = slice
    if min - vertical_distance < 0:
        min = 0
    else:
        min = min - vertical_distance
    if max + vertical_distance + 1 >= img.shape[0]:
        max = img.shape[0]
    else:
        max = max + vertical_distance + 1
    for slice in range(min, max):
        mask[slice, :, :] = np.ones_like(mask[slice, :, :])
    return mask

def loadSegArraywithID(fold, iden):
    segPath = os.path.join(fold, iden + "_seg.nii.gz")
    seg = sitk.ReadImage(segPath)
    return seg


def setimageapcing(fold, iden, spacing, thickness):
    segPath = os.path.join(fold, iden + ".nrrd")
    image = sitk.ReadImage(segPath)
    image.SetSpacing([spacing, spacing, thickness])
    sitk.WriteImage(image, segPath)

def loadlungArraywithID(fold):
    lungPath = os.path.join(fold, "lung.nii.gz")
    lung = sitk.ReadImage(lungPath)
    # lung_array = sitk.GetArrayFromImage(lung)
    return lung


def convert_distance_to_pixel(distence, spacing):
    pixel = int(distence / spacing)
    return pixel

if __name__ == '__main__':
    imgDir = args.imgDir

    dirlist = os.listdir(imgDir)[:]
    dirlist.sort()
    Peritumoral_area1 = 5  # mm
    Peritumoral_area2 = 10  # mm
    Peritumoral_area3 = 15  # mm
    Peritumoral_area4 = 20  # mm

    shape_error = []
    for ind in tqdm(dirlist):
        path = os.path.join(imgDir, str(ind))
        print(ind)
        nodule = sitk.ReadImage(path + '/' + ind + '.nii.gz')
        mask = loadSegArraywithID(path, ind)
        lung = lungSeg(path, ind)
        mask_o = sitk.GetArrayFromImage(mask)

        mask_o = keep_connected_area(mask_o)

        lung = sitk.GetArrayFromImage(lung)
        lung[lung > 1] = 1
        mask_o = np.array(mask_o, dtype='uint8', order=None)
        Spacing = nodule.GetSpacing()
        spacing = Spacing[0]
        thickness = Spacing[2]


        # 转换扩张距离 mm->pixcel
        Peritumoral_area_pixcel_5mm = convert_distance_to_pixel(Peritumoral_area1, spacing)
        Peritumoral_area_pixcel_10mm = convert_distance_to_pixel(Peritumoral_area2, spacing)
        Peritumoral_area_pixcel_15mm = convert_distance_to_pixel(Peritumoral_area3, spacing)
        Peritumoral_area_pixcel_20mm = convert_distance_to_pixel(Peritumoral_area4, spacing)

        vertical_peritumoral_distance_5mm = convert_distance_to_pixel(Peritumoral_area1, thickness)
        vertical_peritumoral_distance_10mm = convert_distance_to_pixel(Peritumoral_area2, thickness)
        vertical_peritumoral_distance_15mm = convert_distance_to_pixel(Peritumoral_area3, thickness)
        vertical_peritumoral_distance_20mm = convert_distance_to_pixel(Peritumoral_area4, thickness)

        # 保存扩张后的mask
        peritumoral_mask_5mm = np.array(mask_o, dtype='uint8', order=None)
        peritumoral_mask_10mm = np.array(mask_o, dtype='uint8', order=None)
        peritumoral_mask_15mm = np.array(mask_o, dtype='uint8', order=None)
        peritumoral_mask_20mm = np.array(mask_o, dtype='uint8', order=None)

        if peritumoral_mask_5mm.shape != lung.shape:
            shape_error.append(ind)
            continue

        dilation_mask_5mm = make_binary_dilation_mask(peritumoral_mask_5mm, vertical_peritumoral_distance_5mm) * lung
        dilation_mask_10mm = make_binary_dilation_mask(peritumoral_mask_10mm, vertical_peritumoral_distance_10mm) * lung
        dilation_mask_15mm = make_binary_dilation_mask(peritumoral_mask_15mm, vertical_peritumoral_distance_15mm) * lung
        dilation_mask_20mm = make_binary_dilation_mask(peritumoral_mask_20mm, vertical_peritumoral_distance_20mm) * lung

        # print("inite roi")
        struct = scipy.ndimage.generate_binary_structure(3, 1)
        scipy.ndimage.binary_dilation(input=mask_o, structure=struct, iterations=Peritumoral_area_pixcel_5mm,
                                      mask=dilation_mask_5mm, output=peritumoral_mask_5mm)
        scipy.ndimage.binary_dilation(input=mask_o, structure=struct, iterations=Peritumoral_area_pixcel_10mm,
                                      mask=dilation_mask_10mm, output=peritumoral_mask_10mm)
        scipy.ndimage.binary_dilation(input=mask_o, structure=struct, iterations=Peritumoral_area_pixcel_15mm,
                                      mask=dilation_mask_15mm, output=peritumoral_mask_15mm)
        scipy.ndimage.binary_dilation(input=mask_o, structure=struct, iterations=Peritumoral_area_pixcel_20mm,
                                      mask=dilation_mask_20mm, output=peritumoral_mask_20mm)

        mask_5mm_ = np.array(peritumoral_mask_5mm, dtype='uint8')
        mask_10mm_ = np.array(peritumoral_mask_10mm, dtype='uint8')
        mask_15mm_ = np.array(peritumoral_mask_15mm, dtype='uint8')
        mask_20mm_ = np.array(peritumoral_mask_20mm, dtype='uint8')

        mask_5mm = (mask_5mm_ - mask_o) * dilation_mask_5mm
        mask_10mm = (mask_10mm_ - mask_o - mask_5mm) * dilation_mask_10mm
        mask_15mm = (mask_15mm_ - mask_o - mask_5mm - mask_10mm) * dilation_mask_15mm
        mask_20mm = (mask_20mm_ - mask_o - mask_5mm - mask_10mm - mask_15mm) * dilation_mask_20mm

        opmask_20mm = (mask_20mm_) * dilation_mask_20mm + mask_o

        opmask_20mm[opmask_20mm > 1] = 1

        mask_5mm = sitk.GetImageFromArray(mask_5mm)
        mask_10mm = sitk.GetImageFromArray(mask_10mm)
        mask_15mm = sitk.GetImageFromArray(mask_15mm)
        mask_20mm = sitk.GetImageFromArray(mask_20mm)

        # Generate 20mm solid peritumoral area for manual calibration
        opmask_20mm = sitk.GetImageFromArray(opmask_20mm)

        mask_5mm.SetSpacing([spacing, spacing, thickness])
        mask_10mm.SetSpacing([spacing, spacing, thickness])
        mask_15mm.SetSpacing([spacing, spacing, thickness])
        mask_20mm.SetSpacing([spacing, spacing, thickness])

        opmask_20mm.SetSpacing([spacing, spacing, thickness])

        sitk.WriteImage(mask_5mm, path + '/peritumoral_05mm_nodule.nii.gz')
        sitk.WriteImage(mask_10mm, path + '/peritumoral_10mm_nodule.nii.gz')
        sitk.WriteImage(mask_15mm, path + '/peritumoral_15mm_nodule.nii.gz')
        sitk.WriteImage(mask_20mm, path + '/peritumoral_20mm_nodule.nii.gz')
        # roi_rectify.nii.gz is used to manual calibration
        sitk.WriteImage(opmask_20mm, path + '/roi_rectify.nii.gz')

