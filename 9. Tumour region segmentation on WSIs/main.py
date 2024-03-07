import sys
import os
import argparse
import logging
import json
import time


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel, BCELoss
from torch.optim import SGD
from torchvision import models
from torch import nn


from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from data.image_producer import ImageDataset   #noqa
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cnn_path', default='./configs/cnn.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default='./checkpoint/', metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=12, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')


def chose_model(cnn):
    if cnn['model'] == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def train_epoch(summary, summary_writer, cnn, model, loss_fn, optimizer,
                dataloader_train):
    model.train()

    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    dataiter_train = iter(dataloader_train)

    time_now = time.time()
    for step in range(steps):
        data_train, target_train = next(dataiter_train)
        data_train = Variable(data_train.float().cuda(device=None, non_blocking=False))
        target_train = Variable(target_train.float().cuda(device=None, non_blocking=False))
        output = model(data_train)
        output = torch.squeeze(output) # noqa
         
        loss = loss_fn(output, target_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)

        acc_data = (predicts == target_train).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
        loss_data = loss.data



        time_spent = time.time() - time_now
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] , loss_data, acc_data, time_spent))

        
        summary_train1 = {'epochs': 0, 'step': 0, 'train_acc': 0, 'train_loss': 0}
        summary_train1['step'] = summary['step']
        summary_train1['epochs'] = summary['epoch']
        summary_train1['train_acc'] = '{:.3f}'.format(acc_data)
        summary_train1['train_loss'] = '{:.5f}'.format(loss_data)


        # print({'epoch': summary_train1['epochs'],
        #        'step': summary_train1['step'],
        #        'train_loss': summary_train1['train_loss'],
        #        'train_acc': summary_train1['train_acc']
        #        })
        # torch.save({'epoch': summary_train1['epochs'],
        #             'step': summary_train1['step'],
        #             'train_loss': summary_train1['train_loss'],
        #             'train_acc': summary_train1['train_acc']
        #             },
        #            'train_log.txt'
        #            )

        summary['step'] += 1

        if summary['step'] % cnn['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

        ###
        loss_data = '{} {:.5f}'.format(summary_train1['step'], loss_data)  ###
        save_log(loss_data)
        ###

    summary['epoch'] += 1
    return summary


# save_log  loss
def save_log(loss_data):
    if os.path.exists('./resnet50/loss/loss.txt'):
        logFile = open('./resnet50/loss/loss.txt', 'a')
        logFile.write(loss_data + '\n')





def valid_epoch(summary, model, loss_fn,
                dataloader_valid):
    model.eval()

    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    dataiter_valid = iter(dataloader_valid)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        data_valid, target_valid = next(dataiter_valid)
        with torch.no_grad():
            data_valid = Variable(data_valid.float().cuda(device=None, non_blocking=False))
        target_valid = Variable(target_valid.float().cuda(device=None, non_blocking=True))

        output = model(data_valid)
        output = torch.squeeze(output) # important
        
        target_valid = target_valid.type_as(output)
        loss = loss_fn(output, target_valid)


        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        acc_data = (predicts == target_valid).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
        loss_data = loss.data

        loss_sum += loss_data
        acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def run(args):
    with open(args.cnn_path, 'r') as f:
        cnn = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cnn['batch_size'] * num_GPU
    batch_size_valid = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU

    model = chose_model(cnn)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1) 
    model = DataParallel(model, device_ids=None)
    model = model.cuda()
    loss_fn = BCEWithLogitsLoss().cuda()
    # loss_fn = BCELoss().cuda()
  
    optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum'])

    # dataset_train = ImageFolder(cnn['data_path_train'])
    # dataset_valid = ImageFolder(cnn['data_path_valid'])
    dataset_train = ImageDataset(cnn['data_path_train'],
                                 cnn['image_size'],
                                 cnn['normalize'])
    dataset_valid = ImageDataset(cnn['data_path_valid'],
                                 cnn['image_size'],
                                 cnn['normalize'])

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size_train,
                                  num_workers=num_workers)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    # summary_writer = SummaryWriter(args.save_path)              
    summary_writer = SummaryWriter()                            
    loss_valid_best = float('inf')
    for epoch in range(cnn['epoch']):
        summary_train = train_epoch(summary_train, summary_writer, cnn, model,
                                    loss_fn, optimizer, dataloader_train)

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
        # torch.save(model, os.path.join(args.save_path, 'train.pkl'))


        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, model, loss_fn,
                                    dataloader_valid)
        time_spent = time.time() - time_now

        logging.info('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], time_spent))

        summary_writer.add_scalar('valid/loss',
                                  summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar('valid/acc',
                                  summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'best.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
