
import os
import glob
import numpy as np


MASK_PATH = './normal_tissue'  
TXT_PATH = './spot_normal.txt'         
patch_number = 100          
level = 6


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)

        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points


def run():
    mask_paths = glob.glob(os.path.join(MASK_PATH, '*.npy'))        
    mask_paths.sort()
    
    center_pointsss = np.empty([0,3])
    
    for mask_path in mask_paths:
        sampled_points = patch_point_in_mask_gen(mask_path, patch_number).get_patch_point()
        sampled_points = (sampled_points * 2 ** level).astype(np.int32)          # make sure the factor
        mask_name = os.path.split(mask_path)[-1].split(".")[0]
        name = np.full((sampled_points.shape[0], 1), mask_name)
        center_points = np.hstack((name, sampled_points))
        
        center_pointsss = np.append(center_pointsss, center_points, axis=0)

    with open(TXT_PATH, "a") as f:
        np.savetxt(f, center_pointsss, fmt="%s", delimiter=",")    


def main():
    run()


if __name__ == "__main__":
    main()
