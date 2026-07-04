# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.models.losses.cross_entropy_loss import soft_cross_entropy

@MODELS.register_module()
class MESADistilLoss(nn.Module):
    """MESA Distillation Loss.

    Args:
        reduction (str): The method used to reduce the loss. Options are "mean", "sum" and "none". Default: "mean".
        loss_weight (float): Weight of the loss. Default: 1.0.
        T (float): Temperature for distillation. Default: 3.
        kl_scale (float): Scale factor for KL divergence loss. Default: 50.
        with_kl (bool): Whether to use KL divergence loss. Default: True.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 T=3,
                 kl_scale = 50,
                 with_kl=True,
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.with_kl = with_kl
        if self.with_kl:
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
            self.T = T
            self.kl_scale = kl_scale
        
    def forward(self,
                cls_score,
                pseudo_score=None,
                **kwargs):
        if pseudo_score is None:
            return 0
        
        if self.with_kl:
            p = nn.functional.log_softmax(cls_score/self.T, dim=1)
            q = nn.functional.softmax(pseudo_score/self.T, dim=1)
            loss = self.kl_scale * self.kl_loss(p, q)
            if self.reduction == "mean":
                loss = loss.mean()
        else:
            pseudo_score = pseudo_score.softmax(dim=1).detach()
            loss = soft_cross_entropy(cls_score, pseudo_score, reduction=self.reduction)
        
        
        return self.loss_weight * loss
