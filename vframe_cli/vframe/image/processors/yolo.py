############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import numpy as np
import cv2 as cv

from vframe.settings import app_cfg
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils import im_utils

class YOLOProc(DetectionProc):


  def _pre_process(self, im):
    """Pre-process image
    """
    
    cfg = self.dnn_cfg

    if cfg.width % 32:
      wh = int(round(cfg.width / 32)) * 32
      cfg.width = wh
      cfg.height = wh
      self.log.warn(f'YOLO width and height must be multiple of 32. Setting to: {wh}')

    im = im_utils.resize(im, width=cfg.width, height=cfg.height, force_fit=cfg.fit)
    self.frame_dim = im.shape[:2][::-1]
    dim = self.frame_dim if cfg.fit else cfg.size
    blob = cv.dnn.blobFromImage(im, cfg.scale, dim, cfg.mean, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)


  def _post_process(self, outs):
    """Post process net output for YOLO object detection
    Network produces output blob with a shape NxC where N is a number of
    detected objects and C is a number of classes + 4 where the first 4
    numbers are [center_x, center_y, width, height]
    """
    
    detect_results = []

    for out in outs:
      out_filtered_idxs = np.where(out[:,5:] > self.dnn_cfg.threshold)
      out = [out[x] for x in out_filtered_idxs[0]]
      for detection in out:
        scores = detection[5:]
        class_idx = np.argmax(scores)
        confidence = scores[class_idx]
        if confidence > self.dnn_cfg.threshold:
          cx, cy, w, h = detection[0:4]
          bbox = BBoxNorm.from_cxcywh_dim((cx, cy, w, h), *self.frame_dim)
          label = self.labels[class_idx] if self.labels else ''
          detect_result = DetectResult(class_idx, confidence, bbox, label)
          detect_results.append(detect_result)

    if self.dnn_cfg.nms:
      detect_results = self._nms(detect_results)

    return DetectResults(detect_results, self._perf_ms())