# 4. Epithelial and stromal tissue segmentation on histopathological patches
## How to Use
Prepare:
```
pip install -r requirements.txt
```
Run:
```
python WSI_seg.py
```
Options:
```
--backbone                backbone name (default: resnet)
--out-stride              network output stride (default: 8)
--use-balanced-weight     whether to use balanced weights (default: False)
--sync-bn                 whether to use sync bn (default: auto)
--freeze-bn               whether to freeze bn parameters (default: False)
--no-cuda                 disables CUDA training
--gpu-ids                 use which GPU to train, must be a comma-separated list of integers only (default: 0)
--seed                    random seed (default: 1)
--resume                  put the path to resuming file if needed
--ft                      finetuning on a different dataset
--overlap                 overlap
--WSI_fold                put the path of input WSI
--save_dir                output save path
--resize                  resize int (default: 1)
```
  
