from easydict import EasyDict as edict
from functools import partial
from copy import deepcopy
import torch
from torchvision import transforms
from adaptis.engine.trainer import AdaptISTrainer
from adaptis.model.toy.models import get_unet_model
from adaptis.model.losses import NormalizedFocalLossSigmoid, NormalizedFocalLossSoftmax, AdaptISProposalsLossIoU
from adaptis.model.metrics import AdaptiveIoU
from adaptis.data.toy import ToyDataset
from adaptis.utils import log
from adaptis.model import initializer

from albumentations import Compose, Blur, Flip, IAAAdditiveGaussianNoise

# import datasetzoo
from datasetzoo.nuts import NutsDataset

class SegTrainer():
    def __init__(self,args,):

    
        
        
        self.model_cfg = edict()
        
        self.model_cfg.input_normalization = {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }

        self.model_cfg.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.model_cfg.input_normalization['mean'],
                                 self.model_cfg.input_normalization['std']),
        ])

        # Settings
        self.loss_cfg_prop = edict()
        self.loss_cfg = edict()
        self.loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.50, gamma=2)
        self.loss_cfg_prop.instance_loss = NormalizedFocalLossSigmoid(alpha=0.50, gamma=2)
        self.loss_cfg.instance_loss_weight = 1.0
        self.loss_cfg_prop.instance_loss_weight = 0.0

        self.args = args
        self.num_epochs = self.args.seg_num_epochs  # = 160
        self.num_points = self.args.seg_num_points  # = 12

        self.loss_cfg.segmentation_loss = NormalizedFocalLossSoftmax(ignore_label=-1, gamma=1)
        self.loss_cfg.segmentation_loss_weight = 0.75

        self.num_epochs_prop = self.args.prop_num_epochs  # = 10
        self.num_points_prop = self.args.prop_num_points  # = 32

        self.loss_cfg_prop.proposals_loss = AdaptISProposalsLossIoU(self.args.batch_size)
        self.loss_cfg_prop.proposals_loss_weight = 1.0

        self.args.val_batch_size = self.args.batch_size
        self.args.input_normalization = self.model_cfg.input_normalization
    
        self.dataset = NutsDataset
        # self.dataset = getattr(datasetzoo,self.args.dataset)


    def train(self, train_proposals = False, start_epoch=0):
        num_points = self.num_points if not train_proposals else self.num_points_prop
        num_epochs = self.num_epochs if not train_proposals else self.num_epochs_prop
        loss_cfg = self.loss_cfg if not train_proposals else self.loss_cfg_prop

        val_loss_cfg = deepcopy(loss_cfg)
        # training using DataParallel is not implemented
        norm_layer = torch.nn.BatchNorm2d
        self.model = get_unet_model(loss_cfg, val_loss_cfg, norm_layer=norm_layer)
        self.model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))
        
        # Augmentation
        train_augmentator = Compose([
            Blur(blur_limit=(2, 4)),
            IAAAdditiveGaussianNoise(scale=(10, 40), p=0.5),
            Flip()
        ], p=1.0)
        
        # Datasets
        trainset = self.dataset(
            self.args.dataset_path,
            split='train',
            num_points=num_points,
            augmentator=train_augmentator,
            with_segmentation=True,
            points_from_one_object=train_proposals,
            input_transform=self.model_cfg.input_transform
        )
    
        valset = self.dataset(
            self.args.dataset_path,
            split='test',
            augmentator=train_augmentator,
            num_points=num_points,
            with_segmentation=True,
            points_from_one_object=train_proposals,
            input_transform=self.model_cfg.input_transform
        )
    
        # Other Settings
        optimizer_params = {
            'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
        }
    
        if not train_proposals:
            lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                                   last_epoch=-1)
        else:
            lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                                   last_epoch=-1)
        
        # Startup AdaptISTrainer
        trainer = AdaptISTrainer(self.args, self.model, self.model_cfg, loss_cfg,
                                 trainset, valset,
                                 num_epochs=num_epochs,
                                 optimizer_params=optimizer_params,
                                 lr_scheduler=lr_scheduler,
                                 checkpoint_interval=40 if not train_proposals else 5,
                                 image_dump_interval=600 if not train_proposals else -1,
                                 train_proposals=train_proposals,
                                 metrics=[AdaptiveIoU()])
    
        log.logger.info(f'Starting Epoch: {start_epoch}')
        log.logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            trainer.training(epoch)
            trainer.validation(epoch)


    def add_proposals_head(self):
        self.model.add_proposals_head()
