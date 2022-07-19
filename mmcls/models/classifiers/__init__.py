# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .kd_base import BaseClassifierKD

__all__ = ['BaseClassifier', 'ImageClassifier', 'BaseClassifierKD']
