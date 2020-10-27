############################################################################# 
#
# VFRAME Training
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from dataclasses import dataclass

from vframe.settings import app_cfg, modelzoo_cfg


@dataclass
class TFProjectConfig:
  # TensorFlow training project
  # input/output
  annotations: str
  images: str
  output: str
  split_train: float=0.6
  split_test: float=0.2
  split_val: float=0.2
  random_seed: int=0


@dataclass
class YoloProjectConfig:
  # YOLO file i/o
  annotations: str
  output: str
  images: str
  cfg: str
  weights: str
  darknet: str=app_cfg.FP_DARKNET_BIN
  logfile: str=app_cfg.FN_LOGFILE
  show_output: bool=False
  gpu_idx_init: int=0
  gpu_idxs_resume: List = field(default_factory=lambda: [0])
  classes: str=app_cfg.FN_CLASSES
  images_labels: str=app_cfg.DN_IMAGES_LABELS
  use_symlinks: bool=True
    
  # YOLO network config
  subdivisions: int=16
  batch_size: int=64
  image_size: int=416

  # Hyperparameters
  focal_loss: bool=False
  learning_rate: float=0.001  # yolov4
  batch_ceiling: int=50000  # total max batches, overrides suggested values

  # Data augmentation
  cutmix: bool=False
  mosaic: bool=True
  mixup: bool=False
  blur: bool=False



