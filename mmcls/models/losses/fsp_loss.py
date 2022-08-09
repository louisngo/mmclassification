# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def fsp_matrix(fm1, fm2):
    if fm1.size(2) > fm2.size(2):  # fm1: (N, C1, H, W), fm2: (N, C2, H, W)
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))  # equal H, W
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)  # fm1: (N, C1, HW)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1)  # fm2: (N, C2, HW)
    fm2 = fm2.transpose(1, 2)  # fm2: (N, HW, C2)
    fsp = fm1.bmm(fm2) / fm1.size(2)
    return fsp


@LOSSES.register_module()
class FspLoss(nn.Module):

    def __init__(self, reduction='batchmean'):
        super(FspLoss, self).__init__()
        self.reduction = reduction

    def forward(self, student_layers, teacher_layers):
        assert len(student_layers) == len(teacher_layers)
        loss_kd = 0
        no_matrix = len(student_layers) - 1
        for i in range(no_matrix):
            fsp_s = fsp_matrix(student_layers[i], student_layers[i + 1])
            fsp_t = fsp_matrix(teacher_layers[i], teacher_layers[i + 1])
            loss_kd += (fsp_s - fsp_t).pow(2).mean()
        loss_kd /= (no_matrix + 1)

        return loss_kd
