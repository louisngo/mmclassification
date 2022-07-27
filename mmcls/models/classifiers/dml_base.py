from ..builder import (CLASSIFIERS, build_backbone, build_head,
                       build_neck, build_loss)
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import torch.nn.functional as F
from torch import nn
from shutil import ExecError

@CLASSIFIERS.register_module()
class BaseClassifierDML(BaseClassifier):
    def __init__(self,
                 backbone,
                 kd_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(BaseClassifierDML, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.student1 = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['student1']),
                'neck': build_neck(neck['student1']),
                'head': build_head(head['student1'])
            }
        )

        self.student2 = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['student2']),
                'neck': build_neck(neck['student2']),
                'head': build_head(head['student2'])
            }
        )

        self.criterionCls = F.cross_entropy
        self.criterionKD = build_loss(kd_loss)
        self.lambda_kd = train_cfg['lambda_kd']
        # self.teacher_ckpt = train_cfg['teacher_checkpoint']
        # self.load_teacher()

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def load_teacher(self):
        try:
            self.teacher.load_state_dict(
                torch.load(self.teacher_ckpt)['state_dict'])
            print(f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
        except:
            print(f'Teacher pretrained model has not been loaded {self.teacher_ckpt}')
        for param in self.teacher.parameters():
            param.required_grad = False

    def extract_feat(self, model, img):
        """Directly extract features from the specified stage."""

        x_feat = model['backbone'](img)
        if isinstance(x_feat, tuple):
            x = x_feat[-1]
        x = model['neck'](x)
        return x, x_feat

    def get_logits(self, model, img):
        x, x_feat = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit, x_feat

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        student1_logit, _ = self.get_logits(self.student1, img)
        student2_logit, _ = self.get_logits(self.student2, img)
        loss_cls1 = self.criterionCls(student1_logit, gt_label)
        loss_cls2 = self.criterionCls(student2_logit, gt_label)

        loss_kd1 = self.criterionKD(
            student1_logit, student2_logit) * self.lambda_kd
        loss_kd2 = self.criterionKD(
            student2_logit, student1_logit) * self.lambda_kd

        losses = dict(loss_cls1=loss_cls1,
                      loss_cls2=loss_cls2,
                      loss_kd1=loss_kd1,
                      loss_kd2=loss_kd2)
        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x, _ = self.extract_feat(self.student1, img)
        if isinstance(x, tuple):
            x = x[-1]
        res = self.student1['head'].simple_test(x)

        return res