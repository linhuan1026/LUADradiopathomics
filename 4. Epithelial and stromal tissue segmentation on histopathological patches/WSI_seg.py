from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from skimage import morphology
import torch
import argparse
import os
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.calculate_weights import calculate_weigths_labels
from PIL import Image
from tkinter import filedialog
import time
from tqdm import tqdm

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample
      
        # print(mask.shape)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample


        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        return img

class WSI_seg(object):
    def __init__(self, args):
        self.args = args
        self.nclass = 5
        palette = [0]*768
        palette[0:3] = [205,51,51]
        palette[3:6] = [0,255,0]
        palette[6:9] = [65,105,225]
        palette[9:12] = [255,165,0]
        palette[12:15] = [255, 255, 255]
        self.palette = palette
        model = DeepLab(num_classes=4,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        
        self.model = model

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        resume = args.resume
        if args.resume is not None:
            if not os.path.isfile(resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(resume))

            checkpoint = torch.load(resume)
            W = checkpoint['state_dict']
            self.model.module.load_state_dict(W, strict=False)

            # if not self.args.ft:
            #     self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        self.model.eval()
        
    def gen_bg_mask(self, orig_img):
        orig_img = np.asarray(orig_img)
        img_array = np.array(orig_img).astype(np.uint8)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = np.uint8(binary)    
        dst = morphology.remove_small_objects(binary==255,min_size=10,connectivity=1)
        bg_mask = np.ones(orig_img.shape[:2])*(-100.00000)
        bg_mask[dst==True]=100.0000
        return bg_mask

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([Normalize(),ToTensor()])
        return composed_transforms(sample)

    def PILImageToCV(self, img):
        # PIL Image transform to OpenCV
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = np.asarray(img)
        # print(img.shape)
        # print(img)
        img = img[:,:,::-1]
        return img

    def CVImageToPIL(self, img):
        # OpenCV transform to PIL image
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img[:,:,::-1]
        # print(img.shape)
        img = Image.fromarray(np.uint8(img))
        return img

    def read_img(self, img_dir):
        img = cv2.imread(img_dir)
        
        return img

    def gain_network_output(self, WSI):
        H = WSI.shape[0]
        W = WSI.shape[1]
        num=0
        G = np.zeros((5, H, W))
        D = np.zeros((5, H, W))
        range_x = [0]
        range_y = [0]
        for x in tqdm(range(0, H, 224-self.args.overlap)):
            if x+224 > H:
                x = H-224
            for y in range(0, W, 224-self.args.overlap):
                if y+224 > W:
                    y = W-224
                patch_cv2 = WSI[x:x+224, y:y+224,:] # cv2
                patch_Image = self.CVImageToPIL(patch_cv2) # Image 
                patch = self.transform_val(patch_Image) # tensor
                patch = patch.unsqueeze(0) 
                # n,c,h,w = patch.size()
                if self.args.cuda:
                    patch = patch.cuda()
                with torch.no_grad():
                    output = self.model(patch)
                pred = output.data.cpu().type(torch.float64)
                pred_old =pred
                bg_mask = self.gen_bg_mask(patch_Image)
                bg_mask = torch.from_numpy(bg_mask).unsqueeze(dim=0).unsqueeze(dim=0).type(torch.float64)
                pred = torch.cat((pred,bg_mask),1).numpy()
                pred = pred[0]

                G[:, x:x+224, y:y+224] += pred
                D[:, x:x+224, y:y+224] += 1
                num+=1
        G /=D
        return G

    def gain_segmentation(self, G, WSI):

        H = G.shape[1]
        W = G.shape[2]
        stitch = np.zeros(shape=(H,W,3))
        big_mask = np.zeros(shape=(H,W,3))

        for x in range(0, H, 224):
            if x+224 > H:
                x = H - 224
            for y in range(0, W, 224):
                if y+224 > W:
                    y = W - 224
                cam = G[:, x:x+224, y:y+224]
                patch_cv2 = WSI[x:x+224, y:y+224,:] # cv2
                patch_Image = self.CVImageToPIL(patch_cv2) # Image 
                pred = np.argmax(cam, 0)
           
                visualimg = Image.fromarray(pred.astype(np.uint8), "P")
                visualimg.putpalette(self.palette)
                visualimg = visualimg.convert("RGB")

                mask_on_img = self.fuse_mask_and_img(visualimg, patch_Image)
                stitch[x:x+224, y:y+224,:] = mask_on_img   
                big_mask[x:x+224, y:y+224,:] = self.PILImageToCV(visualimg)

        return self.CVImageToPIL(stitch), self.CVImageToPIL(big_mask)

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = 0.5 * heatmap + 0.5 * np.float32(img)
        # cam = heatmap
        return np.uint8(255 * cam)

    def fuse_mask_and_img(self, mask, img):
        mask = self.PILImageToCV(mask)
        img = self.PILImageToCV(img)
        Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
        return Combine

    def seg(self, WSI_dir):
        
        WSI = self.read_img(WSI_dir)
        print('Shape of WSI:{}'.format(WSI.shape[0:2]))
        pred = self.gain_network_output(WSI)
        mask_on_WSI, big_mask = self.gain_segmentation(pred, WSI)

        return mask_on_WSI, big_mask, WSI.shape[0:2]

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='stage2_save_dir/all_20210924/checkpoint24.pth.tar',
                        help='put the path to resuming file if needed')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    parser.add_argument('--overlap', type=int, default=200, help='overlap')
    parser.add_argument('--WSI_fold', type=str, default='/media/hero/2T-wym/patch/for_tissue_10x_01/11/', help='put the path of WSI')
    parser.add_argument('--save_dir', type=str, default='/media/hero/2T-wym/patch/seg_tissue_for_tissue_10x/11/', help='save path')
    parser.add_argument('--resize', type=int, default=1, help='resize int')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.save_dir=='':
        args.save_dir = os.path.dirname(args.WSI_dir)

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)
    torch.manual_seed(args.seed)


    for j,k,files in os.walk(args.WSI_fold):
        for file in sorted(files):
            print(file)
            args.WSI_dir = os.path.join(j,file)

            WSI_seger = WSI_seg(args)

            begin_time = time.time()
            mask_on_WSI, mask, (H, W) = WSI_seger.seg(args.WSI_dir)
            end_time = time.time()
            run_time = end_time-begin_time
            print ('time consumption:',run_time)
            WSI_path_name = j.split('/')[-1]
            WSI_name = args.WSI_dir.split('/')[-1].split('.png')[0]

            save_dir_path = args.save_dir+WSI_path_name
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            if args.resize !=1:
                mask_on_WSI = mask_on_WSI.resize((int(W/args.resize), int(H/args.resize)),Image.ANTIALIAS)
                mask_on_WSI.save(os.path.join(save_dir_path, WSI_name + '-seg-downsample-{}.png'.format(args.resize)))

            else:
                mask_on_WSI.save(os.path.join(save_dir_path, WSI_name + '-seg.png'))
            mask.save(os.path.join(save_dir_path, WSI_name + '-mask.png'))

if __name__ == "__main__":
   main()     



