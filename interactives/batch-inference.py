# %% markdown
# # Batch inference
# %% codecell
%load_ext autoreload
%autoreload 2

import os
from os.path import join
from glob import glob
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Tuple, List
from dataclasses import dataclass
import time

from urllib.parse import unquote
import urllib.parse
from scipy.io import loadmat
from dacite import from_dict
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import yaml
from tqdm.notebook import tqdm, trange
import dacite
# %% codecell
DIR_ROOT = Path(_dh[0]).parent.parent.parent.parent.parent  # global  nb path
DIR_CLI = join(DIR_ROOT, 'vframe_cli')

import sys
sys.path.append(DIR_CLI)
from vframe.settings import app_cfg
from vframe.models.color import Color
from vframe.models.geometry import BBox, Point
from vframe.utils import file_utils, im_utils, draw_utils, video_utils
from pymediainfo import MediaInfo

from vframe.image.dnn_factory import DNNFactory
from vframe.models.dnn import DNN
from vframe.settings import app_cfg, modelzoo_cfg
# %% codecell
fp_vid = '/data_store_nas/datasets/syrianarchive/chemical_attack_dataset/videos/2e062dc9e958cf59a5435f935af698895c51e8cd184f9954bd8fac0320ab7a0c.mp4'
# %% codecell
video_stream = video_utils.FileVideoStream(fp_vid)
# %% codecell
video_stream.start()
# %% codecell
frames = []
for i in range(256):
  frames.append(video_stream.read())
# %% codecell
dnn_cfg = modelzoo_cfg.modelzoo.get('yoloface2')
# %% codecell
dnn_cfg.use_gpu()
#dnn_cfg.use_cpu()
# %% codecell
dnn = DNNFactory.from_dnn_cfg(dnn_cfg)
# %% codecell
ln = dnn.net.getLayerNames()
output_layers = [ln[i[0] - 1] for i in dnn.net.getUnconnectedOutLayers()]
output_layers = output_layers[-1]
# %% codecell
def batch(input_images):
  dnn.net.setInput(input_images)
  output = dnn.net.forward(output_layers)
# %% codecell
wh = (608,608)
wh = (768,768)
input_image = np.random.random((1, 3, *wh))
input_images_single = np.random.random((1, 3, *wh))
input_images_batch = np.random.random((8, 3, *wh))
# %% codecell
# This breaks
iters = 10

# warmup
dnn.net.setInput(input_image)
output = dnn.net.forward(output_layers)

st = time.time()
for i in range(iters):
  batch(input_images_single)
fps = (len(input_images_single) * iters) / (time.time() - st)
print(f'{iters} iterations at {len(input_images_single)} batch size: {fps:.2f} FPS')

dnn.net.setInput(input_image)
output = dnn.net.forward(output_layers)

st = time.time()
for i in range(iters):
  batch(input_images_batch)
fps = (len(input_images_batch) * iters) / (time.time() - st)
print(f'{iters} iterations at {len(input_images_batch)} batch size: {fps:.2f} FPS')
# %% codecell
len(input_images_single)
# %% codecell

# %% codecell
# This breaks
st = time.time()
iters = 10
batch_size = 1
for i in range(iters):
  batch()
t = (st - time.time()) / 1
print(f'{t:.2f} seconds')
# %% codecell
# This breaks
%timeit batch()
# %% codecell
im_utils.np2pil(frames[0])
# %% codecell
a = list(range(10))
print(sum(a)/len(a))
a.remove(0)
print(a)
a.append(11)
print(sum(a)/len(a))
print(a)
# %% codecell

# %% codecell

# %% codecell

# %% codecell
for i in range(1,7):
  fp = f'/home/adam/Downloads/vframe/detections_p{i}.json'
  data = file_utils.load_json(fp)

  n_detections = 0
  for item in data:
    for frame_idx, frame_data in item.get('frames_data').items():
      for model_name, model_results in frame_data.items():
        d = len(model_results.get('detections', []))
        n_detections += d
  print(i, n_detections)
# %% codecell
