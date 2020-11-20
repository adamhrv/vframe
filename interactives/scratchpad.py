# %% markdown
# # Scratchpad temporary code

# %% codecell
import os
from os.path import join
from glob import glob
from pathlib import Path
from dataclasses import asdict
#
from urllib.parse import unquote
import urllib.parse
from scipy.io import loadmat
from dacite import from_dict
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import yaml
from tqdm.notebook import tqdm, trange
import dacite
from pymediainfo import MediaInfo
# %% codecell
import sys
BASE_DIR = os.getcwd()  # if project loaded in root view
DIR_CLI = join(BASE_DIR, 'vframe_cli')
sys.path.append(DIR_CLI)
from vframe.settings import app_cfg
from vframe.models.color import Color
from vframe.models.geometry import BBox
from vframe.utils import file_utils, im_utils, draw_utils, video_utils
from vframe.image.dnn_factory import DNNFactory
from vframe.models.dnn import DNN
from vframe.settings import app_cfg

# %% codecell
