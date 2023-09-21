# 2.Dilation algorithm for peritumour VOIs generation

---

## How to use

___

Prepare:

```
pip install -r requirements.txt
```

File Structure:

```
|--imgDir
    |--Patient001
        |--Patient001.nii.gz
        |--Patient001_seg.nii.gz
    |--Patient002
        |--Patient002.nii.gz
        |--Patient002_seg.nii.gz
    ......
```

Run:

Generate  Peritumour mask

```
python Peritumoral_ROI_Generator.py --imgDir FOLDER_DIR
```

When the peritumoral mask is generated, the lung mask will be automatically segmented and generated. After the script is run, the file directory structure is as follows:

```
|--imgDir
    |--Patient001
        |--Patient001.nii.gz
        |--Patient001_seg.nii.gz
        |--lung.nii.gz
        |--peritumoral_05mm_nodule.nii.gz
        |--peritumoral_10mm_nodule.nii.gz
        |--peritumoral_15mm_nodule.nii.gz
        |--peritumoral_20mm_nodule.nii.gz
        |--roi_rectify.nii.gz
    |--Patient002
        |--Patient002.nii.gz
        |--Patient002_seg.nii.gz
        |--lung.nii.gz
        |--peritumoral_05mm_nodule.nii.gz
        |--peritumoral_10mm_nodule.nii.gz
        |--peritumoral_15mm_nodule.nii.gz
        |--peritumoral_20mm_nodule.nii.gz
        |--roi_rectify.nii.gz
    ......
```

The 'roi_rectify .ni.gz 'file in the folder is used to manually correct all generated peritumoral masks. Steps can be ignored if not needed.

If you artificially corrected the tumor, run the following script:

```
python ROI_celibration.py --imgDir FOLDER_DIR
```

Running the above script will automatically correct all peritumor masks through the manually corrected 'roi_rectify.nii.gz 'file.
