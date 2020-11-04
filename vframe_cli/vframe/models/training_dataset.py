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
  show_images: bool=True
  gpu_idx_init: List = field(default_factory=lambda: [0])
  gpu_idxs_resume: List = field(default_factory=lambda: [0])
  classes: str=app_cfg.FN_CLASSES
  images_labels: str=app_cfg.DN_IMAGES_LABELS
  use_symlinks: bool=True
    
  # YOLO network config
  subdivisions: int=16
  batch_size: int=64
  batch_normalize: bool=True
  width: int=608
  height: int=608
  focal_loss: bool=False
  learning_rate: float=0.001  # yolov4
  batch_ceiling: int=50000  # total max batches, overrides suggested values

  # Data augmentation
  flip: bool=True
  resize: float=1.0
  jitter: float=0.3
  exposure: float=1.5
  saturation: float=1.5
  hue: float=0.1
  cutmix: bool=False
  mosaic: bool=False
  mosaic_bound: bool=False
  mixup: bool=False
  blur: bool=False
  gaussian_noise: int=0

  def __post_init__(self):
    #learning_rate = 0.00261 / GPUs
    #self.learning_rate = self.learning_rate / len(self.gpu_idxs_resume)
    # force mosaic bound false if not using mosaic augmentation
    self.mosaic_bound = False if not self.mosaic else self.mosaic_bound


