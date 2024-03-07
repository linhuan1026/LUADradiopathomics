import sys
import os
import cv2
import argparse
import logging
import json
import time
import glob
import pdb

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.wsi_producer import WSIPatchDataset  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
'''parser.add_argument('--wsi_path', default='./wsi/patient_001.svs', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')'''
parser.add_argument('--ckpt_path', default='./checkpoint/best.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--cnn_path', default='./checkpoint/cnn.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
'''parser.add_argument('--mask_path', default='./tissue_mask/patient_001.npy', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default='./seg_npy/patient_001.npy', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')'''
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=14, type=int, help='number of '
                    'workers to use to make batch, default 0')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')


def chose_model(mod):
    if mod == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        with torch.no_grad():
            data = Variable(data.cuda(non_blocking=True))
        output = model(data)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output.cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           :].cpu().data.numpy().flatten()
        probs_map[x_mask, y_mask] = probs
        count += 1

        # logging.info(
            # '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            # .format(
            #     time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
            #     dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, wsi_path, mask_path, cnn, flip='NONE', rotate='NONE'):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    num_GPU = len(args.GPU.split(','))
    batch_size = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(wsi_path, mask_path,
                        image_size=cnn['image_size'],
                        normalize=True,
                        flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args, wsi_path, mask_path, probs_map_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)

    mask = np.load(mask_path)                                    #tissue_mask
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)            #tissue_mask
    ckpt = torch.load(args.ckpt_path)
    model = chose_model(cnn['model'])
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    if not args.eight_avg:
        dataloader = make_dataloader(
            args, wsi_path, mask_path, cnn, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8

    np.save(probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    
    WSI_PATH = './wsi'
    MASK_PATH = './tissue_mask/'
    PROBS_MAP_PATH = './seg_npy/'

    wsi_paths = glob.glob((os.path.join(WSI_PATH,'*.svs')))
    wsi_paths.sort()
    pbar = tqdm(wsi_paths)
    for wsi_path in pbar:
        ID_name = wsi_path.split('/')[-1]
        ID_name = ID_name.split('.svs')[0]
        # print(ID_name)
        pbar.set_description("Processing %s" % ID_name)
    
        mask_path = MASK_PATH + ID_name + '.npy'
        probs_map_path = PROBS_MAP_PATH + ID_name + '.npy'

        time_now = time.time()
        if os.path.exists(probs_map_path):
            continue
        else:
            run(args, wsi_path, mask_path, probs_map_path)
        time_spent = time.time() - time_now
        print(time_spent)



if __name__ == '__main__':
    main()
