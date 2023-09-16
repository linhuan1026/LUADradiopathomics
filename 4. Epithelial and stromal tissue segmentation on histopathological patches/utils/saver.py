import os
import shutil
import torch
from collections import OrderedDict
import glob
import pdb

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('stage2_save_dir')
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        self.id_list = [int(self.runs[i].split('_')[-1]) for i in range(len(self.runs))]
        # run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        run_id = max(self.id_list) + 1 if self.runs else 0
        
        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename):
        """Saves checkpoint to disk"""
        
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            # if self.runs:
            #     previous_miou = [0.0]
            #     for run in self.runs:
            #         run_id = run.split('_')[-1]
            #         path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
            #         if os.path.exists(path):
            #             with open(path, 'r') as f:
            #                 miou = float(f.readline())
            #                 previous_miou.append(miou)
            #         else:
            #             continue
            #     max_miou = max(previous_miou)
            #     if best_pred > max_miou:
            #         filename = str(run_id) + '_' + filename
            #         filename = os.path.join(self.experiment_dir, filename)
            #         torch.save(state, filename)
            # else:
            #     run_id = run.split('_')[-1]
            #     filename = str(run_id) + '_' + filename
            #     filename = os.path.join(self.experiment_dir, filename)
            #     torch.save(state, filename)
        else:
            filename = os.path.join(self.experiment_dir, filename)
            torch.save(state, filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()