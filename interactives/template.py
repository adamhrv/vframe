# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import os
from os.path import join
from pathlib import Path

import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
# %%
sys.path.append(join(Path(os.getcwd()).parent, 'vframe_cli'))
from vframe.settings import app_cfg
from vframe.models.dnn import DNN
from vframe.models.color import Color
from vframe.image.dnn_factory import DNNFactory
from vframe.settings import app_cfg as vframe_cfg
from vframe.settings import modelzoo_cfg
from vframe.utils import file_utils, draw_utils, im_utils

# %%
def pltim(im):
  im = im_utils.bgr2rgb(im)
  plt.figure(figsize=(10,6))
  plt.xticks([])
  plt.yticks([])
  plt.imshow(im)

# %%
dnn_cfg = modelzoo_cfg.modelzoo.get('yoloface')
dnn = DNNFactory.from_dnn_cfg(dnn_cfg)

# %%
fp_dir_dataset = '/media/adam/megapixels_2tb/data_store/datasets/people/megaface/'
fp_dir_out = '/media/adam/megapixels_2tb/data_store/datasets/people/megaface/images_by_license/cc_by_montage/'
fp_dir_ims = join(fp_dir_dataset, 'images_by_license/cc_by')
fp_ims = file_utils.glob(join(fp_dir_ims, '*'))
print(len(fp_ims))

# %%
n_imgs = 10
dim = (500,500)

for i in range(n_imgs):
  # get N images with faces
  fp_im = fp_ims[i]
  print(fp_im)
  im = cv.imread(fp_im)
  results = dnn.infer(im)
  bboxes = [d.bbox for d in results.detections]
  if not len(bboxes):
    print(f'no faces in {fp_im}')
    continue
  print(f'Found {len(bboxes)} faces')

  # blur faces
  im = im_utils.blur_bbox_soft(im, bboxes, iters=2, expand_per=-0.15, 
    mask_k_fac=0.25, im_k_fac=0.995, multiscale=True)

  # find largest face
  areas = [b.area for b in bboxes]
  max_idx = np.argmax(areas)
  bbox = bboxes[max_idx]

  # Crop expanded bbox
  bbox_exp = bbox.expand_per(2).square()

  # rescale image and bbox
  scale = dim[0] / bbox_exp.width
  im_crop = im_utils.crop_roi(im, bbox_exp)
  width_new = bbox_exp.width * scale
  im_crop = im_utils.resize(im_crop, width=width_new)
  bbox_translated = bbox.translate(0-bbox_exp.x1, 0-bbox_exp.y1)
  bbox_translated = bbox_translated.scale(scale)
  im_crop = draw_utils.draw_bbox(im_crop, bbox_translated, color=app_cfg.WHITE, stroke=6)
  
  # write
  fp_out = join(fp_dir_out, Path(fp_im).name)
  cv.imwrite(fp_out, im_crop)
