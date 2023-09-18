
from PIL import Image
import numpy as np
from skimage import measure
def extract_nuclei_props(fname_mask='test_dense.png', fname_intensity='test_intensity_image.png'): 
    '''
    extract the basic features for each nuclei
    
    Parameters
    ------------------------------ 
    fname_mask : path of the binary image (0-1 image). 
    fname_intensity : path of the original tissue image. Grayscale input image with intensities scaled in the [0 255] range.

    Returns
    ------------------------------ 
    feats : 
            features of ['area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
                                            'euler_number', 'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 
                                            'solidity', 'intensity_max', 'intensity_mean', 'intensity_min']
            [n, 18] 2D-array, n is the number of nuclei, 18 is the number of features
    feats_name : features name 

    '''
    # image=Image.open(fname_mask)
    # img=np.asarray(image) 
    # pres = np.max(img*1.0)/2
    # mask = img>pres
    # labels = measure.label(mask)

    mask = fname_mask
    labels = measure.label(mask)

    # image_intensity = Image.open(fname_intensity).convert('L')
    image_intensity = Image.fromarray(fname_intensity).convert('L')
    image_intensity = np.array(image_intensity)
    props = measure.regionprops(label_image=labels, intensity_image=image_intensity)
    
    atts = ['area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
            'euler_number', 'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 
            'solidity', 'intensity_max', 'intensity_mean', 'intensity_min']
    feats_name=['centroid_x', 'centroid_y'] + atts + ['intensity_std']

    feats = []
    for prop in props: 
        feats.append([prop.centroid[1], prop.centroid[0]] + [prop[att] for att in atts] + [np.std(prop['image_intensity'])]) 

    return np.array(feats), feats_name

# feats_props, feats_names = extract_nuclei_props('test_dense.png', 'test_intensity_image.png' )
# print()
