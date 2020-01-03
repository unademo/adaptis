import math

from collections import Mapping
from collections import namedtuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseData(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """
    
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)
    
    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        
        ret = BaseData(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        
        return ret
    
    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, BaseData):
            new_config_dict = vars(new_config_dict)
        
        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

class AdaptIS(nn.Module):
    def __init__(self,
                 backbone,
                 adaptis_head,
                 segmentation_head=None,
                 proposal_head=None,
                 with_proposals=False,
                 spatial_scale=1.0):
        super(AdaptIS, self).__init__()

        self.with_proposals = with_proposals
        self.spatial_scale = spatial_scale

        self.backbone = backbone
        self.adaptis_head = adaptis_head
        self.segmentation_head = segmentation_head
        self.proposals_head = [proposal_head]
        if with_proposals:
            self.add_proposals_head()

    def add_proposals_head(self):
        self.with_proposals = True

        for param in self.parameters():
            param.requires_grad = False
        self.proposals_head = self.proposals_head[0]

    @staticmethod
    def namedtuple_with_defaults(typename, field_names, default_values=()):
        T = namedtuple(typename, field_names)
        T.__new__.__defaults__ = (None,) * len(T._fields)
        if isinstance(default_values, Mapping):
            prototype = T(**default_values)
        else:
            prototype = T(*default_values)
        T.__new__.__defaults__ = tuple(prototype)
        return T

    # @staticmethod
    # def make_named_outputs(outputs):
        # keys, values = list(zip(*self.outputs))
        # keys, values = list(outputs.keys()),list(outputs.values())

        # named_outputs = namedtuple('outputs', keys,defaults=(None,)*len(keys))(*values)
        
        # named_outputs = namedtuple('outputs', keys)
        # named_outputs.__new__.__defaults__=(values[0],) * len(named_outputs._fields)

        # named_outputs = namedtuple_with_defaults('outputs', keys, outputs)

        
        # print(type(named_outputs))
        # return named_outputs
    
    @staticmethod
    def make_named_outputs(outputs):
        keys, values = list(zip(*outputs))
        named_outputs = namedtuple('output',keys)(*values)
        return named_outputs

    def forward(self, x, points):
        orig_size = x.size()[2:]
        # self.outputs = []
        self.outputs = {'instances':None, 'semantic':None, }
        backbone_features = self.backbone(x)

        # instances
        instance_out = self.adaptis_head(backbone_features, points)
        if not math.isclose(self.spatial_scale, 1.0):
            instance_out = F.interpolate(instance_out, orig_size, mode='bilinear', align_corners=True)
        # self.outputs.append(('instances', instance_out))
        self.outputs['instances'] = instance_out

        # semantic
        if self.segmentation_head is not None:
            semantic_out = self.segmentation_head(backbone_features)
            if not math.isclose(self.spatial_scale, 1.0):
                semantic_out = F.interpolate(semantic_out, orig_size, mode='bilinear', align_corners=True)
            # self.outputs.append(('semantic', semantic_out))
            self.outputs['semantic'] = semantic_out

        # proposals
        if self.with_proposals:
            backbone_features = backbone_features.detach()
            proposals_out = self.proposals_head(backbone_features)
            proposals_out = self.adaptis_head.EQF(proposals_out, points.detach())
            # self.outputs.append(('proposals', proposals_out))
            self.outputs['proposals']=proposals_out

        # named_outputs = (BaseData(self.outputs),)

        # named_outputs = self.make_named_outputs(self.outputs)
        # keys, values = list(zip(*self.outputs))
        # named_outputs = self.namedtuple_with_defaults('output', keys, values)
        
        # keys = list(self.outputs.keys())
        # named_outputs = self.namedtuple_with_defaults('outputs', keys, self.outputs)

        return self.outputs

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights)
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)
        
    def get_trainable_params(self):
        trainable_params = []

        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params


class AdaptISWithLoss(nn.Module):
    def __init__(self,
                 backbone,
                 adaptis_head,
                 loss_cfg,
                 val_loss_cfg,
                 segmentation_head=None,
                 proposal_head=None,
                 with_proposals=False,
                 spatial_scale=1.0):
        super(AdaptISWithLoss, self).__init__()
        
        self.with_proposals = with_proposals
        self.spatial_scale = spatial_scale
        
        self.backbone = backbone
        self.adaptis_head = adaptis_head
        self.segmentation_head = segmentation_head
        self.proposals_head = [proposal_head]
        if with_proposals:
            self.add_proposals_head()
        self.loss_cfg =loss_cfg
        self.val_loss_cfg =val_loss_cfg
    
    def add_proposals_head(self):
        self.with_proposals = True
        
        for param in self.parameters():
            param.requires_grad = False
        self.proposals_head = self.proposals_head[0]
    
    
    def _add_loss(self, loss_name, total_loss, validation, lambda_loss_inputs, ): #losses_logging,
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            # losses_logging[loss_name].append(loss.detach().cpu().numpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss
    
    
    def forward(self, x, points, validation, instance, semantic):  #
        orig_size = x.size()[2:]
        # self.outputs = []
        self.outputs = {'instances': None, 'semantic': None, }
        backbone_features = self.backbone(x)
        
        # instances
        instance_out = self.adaptis_head(backbone_features, points)
        if not math.isclose(self.spatial_scale, 1.0):
            instance_out = F.interpolate(instance_out, orig_size, mode='bilinear', align_corners=True)
        # self.outputs.append(('instances', instance_out))
        self.outputs['instances'] = instance_out
        
        # semantic
        if self.segmentation_head is not None:
            semantic_out = self.segmentation_head(backbone_features)
            if not math.isclose(self.spatial_scale, 1.0):
                semantic_out = F.interpolate(semantic_out, orig_size, mode='bilinear', align_corners=True)
            # self.outputs.append(('semantic', semantic_out))
            self.outputs['semantic'] = semantic_out
        
        # proposals
        if self.with_proposals:
            backbone_features = backbone_features.detach()
            proposals_out = self.proposals_head(backbone_features)
            proposals_out = self.adaptis_head.EQF(proposals_out, points.detach())
            # self.outputs.append(('proposals', proposals_out))
            self.outputs['proposals'] = proposals_out
        
        # named_outputs = (BaseData(self.outputs),)
        
        # named_outputs = self.make_named_outputs(self.outputs)
        # keys, values = list(zip(*self.outputs))
        # named_outputs = self.namedtuple_with_defaults('output', keys, values)
        
        # keys = list(self.outputs.keys())
        # named_outputs = self.namedtuple_with_defaults('outputs', keys, self.outputs)

        loss = 0.0
        # losses_logging = defaultdict(list)
        loss = self._add_loss('instance_loss', loss, validation,
                              lambda: (self.outputs["instances"], instance))
        loss = self._add_loss('segmentation_loss', loss, validation,
                              lambda: (self.outputs["semantic"], semantic))
        loss = self._add_loss('proposals_loss', loss, validation,
                              lambda: (self.outputs["instances"], self.outputs["proposals"], instance))


        return loss, self.outputs #
    
    
    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights)
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)
    
    
    def get_trainable_params(self):
        trainable_params = []
        
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        return trainable_params