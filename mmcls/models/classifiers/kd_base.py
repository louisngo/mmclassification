from ..builder import (CLASSIFIERS, build_backbone, build_head,
                       build_neck, build_loss)
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import torch.nn.functional as F
from torch import nn
from shutil import ExecError

@CLASSIFIERS.register_module()
class BaseClassifierKD(BaseClassifier):
    def __init__(self,
                 backbone,
                 kd_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(BaseClassifierKD, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['student']),
                'neck': build_neck(neck['student']),
                'head': build_head(head['student'])
            }
        )

        self.teacher = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['teacher']),
                'neck': build_neck(neck['teacher']),
                'head': build_head(head['teacher'])
            }
        )

        self.criterionCls = F.cross_entropy
        self.criterionKD = build_loss(kd_loss)
        self.lambda_kd = train_cfg['lambda_kd']
        self.teacher_ckpt = train_cfg['teacher_checkpoint']
        print(self.teacher_ckpt)
        self.load_teacher()

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

            end_of_first_stage = train_cfg.get('end_of_first_stage', None)
            if end_of_first_stage is not None:
                self.end_of_first_stage = end_of_first_stage
                self.criterionKD2 = build_loss(train_cfg["kd_loss_2"])
                self.stage_cnt = 1
                self.epoch_cnt = 0
            else:
                self.end_of_first_stage = None

    def load_teacher(self):
        # try:
        #     self.teacher.load_state_dict(
        #         torch.load(self.teacher_ckpt)['state_dict'])
        #     print(f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
        # except:
        #     print(f'Teacher pretrained model has not been loaded {self.teacher_ckpt}')
        self.teacher.load_state_dict(
                 torch.load(self.teacher_ckpt)['state_dict'])
        print(f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
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
        x = model['head'].fc(x)  # head
        return x, x_feat

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

        with torch.no_grad():
            teacher_logit, t_feat = self.get_logits(self.teacher, img)
        student_logit, s_feat = self.get_logits(self.student, img)
        loss_cls = self.criterionCls(student_logit, gt_label)  # * (1 - self.lambda_kd) # todo
        if self.end_of_first_stage is not None:
            if self.epoch_cnt < self.end_of_first_stage:
                if self.stage_cnt == 391:
                    self.epoch_cnt += 1
                    self.stage_cnt = 0
                self.stage_cnt += 1
                loss_kd = self.criterionKD(s_feat, t_feat)
            else:
                self.end_of_first_stage = None
                self.criterionKD = self.criterionKD2
                self.lambda_kd = 0.9
                loss_kd = self.criterionKD(
                     student_logit, teacher_logit.detach()) * self.lambda_kd
        else:
            loss_kd = self.criterionKD(
                student_logit, teacher_logit.detach()) * self.lambda_kd
        # loss_kd = self.criterionKD(
        #     student_logit, teacher_logit.detach()) * self.lambda_kd
        # loss_kd = self.criterionKD(s_feat, t_feat)

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)
        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x, x_feat = self.extract_feat(self.student, img)
        if isinstance(x, tuple):
            x = x[-1]
        res = self.student['head'].simple_test(x)

        return res
