from PIL import Image
# Read Images
import numpy as np

import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage import morphology


def mask2bounds(fname,
                fname_intensity=None,
                atts=['area', 'intensity_mean', 'eccentricity'], 
                ifbinarization=True, **kwargs):
    '''
    binary image to bounds

    input:: path of the binary image
    return:: r_in, c_in, r_out, c_out, centroid_r, centroid_c
    '''
    #fname = './data/image1_mask.png'
    #fname_intensity = './data/image1.png'
    #image = Image.open(fname)

    #image = Image.fromarray(fname)
    # image.show()

    #img = np.asarray(image)
    #print(np.max(img))
    #pres = np.max(img * 1.0) / 2
    #mask = img > pres
    mask = fname.astype(bool) if ifbinarization else fname 
    #print('mask sum', np.sum(mask), 'mask.shape', mask.shape)
    #exit()
    #print(np.max(mask), mask.shape)
    labels = measure.label(mask)
    if fname_intensity.__class__ != 'NoneType':
        image_intensity = Image.fromarray(fname_intensity).convert('L')
        image_intensity = np.array(image_intensity)
        #print(np.max(image_intensity))
        props = measure.regionprops(label_image=labels,
                                    intensity_image=image_intensity)
    else:
        props = measure.regionprops(label_image=labels)
        print('did not embed the image_intensity.',
              'e.g., intensity_mean et al will be not available')

    # fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(9, 4))
    # plt.gray()
    # ax1.imshow(image_intensity)
    # ax2.imshow(mask, cmap='inferno'), plt.axis('off'), plt.title('labelled')
    # plt.show()

    bounds = []
    feats = []
    for prop in props:
        # print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
        rc_in = measure.find_contours(labels == prop.label, 0.5)[0]
        bimg = morphology.binary_dilation(labels == prop.label, np.ones([3,
                                                                         3]))
        rc = measure.find_contours(bimg, 0.5)[0]
        bounds.append([
            rc_in[:, 1].tolist(),
            rc_in[:, 0].tolist(),  #### 1 first, then 0 important
            rc[:, 1].tolist(),
            rc[:, 0].tolist(),
            prop.centroid[1],
            prop.centroid[0]
        ])
        if fname_intensity.__class__ != 'NoneType' and atts != None:
            feats.append([prop.centroid[1], prop.centroid[0]] +
                         [prop[att] for att in atts])
    return bounds, image_intensity, feats


# # filepath = 'test_sparse.png'
# # bounds = mask2bounds(fname=filepath)
# filepath = 'test_dense.png'
# fname_intensity = 'test_intensity_image.png'

# bounds, image_intensity, feats = mask2bounds(fname=filepath, fname_intensity=fname_intensity, atts=['area', ])
# print()

# fig, ax1 = plt.subplots(ncols=1, figsize=(9, 4))
# # plt.gray()
# ax1.imshow(image_intensity)
# for i in range(len(bounds)):
#     ax1.plot(bounds[i][0], bounds[i][1])
#     ax1.plot(bounds[i][2], bounds[i][3])
#     ax1.scatter(bounds[i][4], bounds[i][5], marker='*')
# plt.show()
# print()
