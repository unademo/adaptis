import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
import torch
from adaptis.utils.exp import init_experiment
from projects.clothes3.segtrainer import SegTrainer

import argparse

def add_exp_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers',        default=4,          type=int,  metavar='N', help='Dataloader threads')
    parser.add_argument('--no-cuda',        default=False,      type=bool, help='disables CUDA training')
    parser.add_argument('--device',         default="cuda",         type=str, help='disables CUDA training')
    parser.add_argument('--ngpus',          default=1,          type=int,  help='number of GPUs')
    parser.add_argument('--gpus',           default='1,2,3',    type=str,  required=False)
    parser.add_argument('--batch-size',     default=6,          type=int, )
    parser.add_argument('--exp-name',       default='clothes3', type=str,  help='experiment name')
    parser.add_argument('--start-epoch',    default=0,          type=int,  help='Start epoch for learning schedule and for logging')
    parser.add_argument('--weights',        default="/home/unaguo/proj/adaptis-pytorch/projects/clothes3/experiments/clothes3/193_clothes3/checkpoints/ep-80-loss-0.7086685662737289_model.pth",       type=str,  help='Put the path to resuming file if needed')
    parser.add_argument('--val-batch-size', default=1,          type=int, )
    parser.add_argument('--no-exp',         default=False,      type=bool, help="Don't create exps dir")
    parser.add_argument('--dataset',        default="ClothesSegDataset", type=str,  help='dataset')
    parser.add_argument('--dataset-path',   default="/media/kk/databases/ClothesSegDataset", type=str,  help='Path to the dataset')

    parser.add_argument('--seg-num-epochs', default=200,        type=int,  help='num epochs without train_proposals')
    parser.add_argument('--seg-num-points', default=12,         type=int,  help='num points without train_proposals')
    parser.add_argument('--prop-num-epochs',default=10,         type=int,  help='num epochs with train_proposals')
    parser.add_argument('--prop-num-points',default=32,         type=int,  help='num points with train_proposals')

    return parser

if __name__ == '__main__':
    # Global Torch Settings
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Init args
    args = init_experiment(experiment_name='clothes3',
                           add_exp_args=add_exp_args,
                           script_path=__file__)
    
    # Training
    trainer = SegTrainer(args=args, )
    trainer.train(train_proposals=False, start_epoch=args.start_epoch)
    trainer.add_proposals_head()
    trainer.train(train_proposals=True)