############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import sys
from math import sqrt
from os.path import join

import logging
import numpy as np
import cv2 as cv
import PIL
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cycler as mpl_cycler

from vframe.models import types
from vframe.models.bbox import BBoxNorm, BBoxDim
from vframe.models.color import Color
from vframe.utils import im_utils
from vframe.settings import app_cfg

# -----------------------------------------------------------------------------
#
# Font Manager
#
# -----------------------------------------------------------------------------

class FontManager:
  
  fonts = {}
  log = logging.getLogger('vframe')
  
  def __init__(self):
    # build/cache a dict of common font sizes
    for i in range(10, 60, 2):
      self.fonts[i] = ImageFont.truetype(join(app_cfg.FP_ROBOTO_400), i)

  def get_font(self, pt):
    """Returns font and creates/caches if doesn't exist
    """
    if not pt in self.fonts.keys():
      self.fonts[pt] = ImageFont.truetype(join(app_cfg.FP_ROBOTO_400), pt)
    return self.fonts[pt]



# -----------------------------------------------------------------------------
#
# Matplotlib utils
#
# -----------------------------------------------------------------------------

def pixels_to_figsize(opt_dim, opt_dpi):
  """Converts pixel dimension to inches figsize
  """
  w, h = opt_dim
  return (w / opt_dpi, h / opt_dpi)


# Plot style
def set_matplotlib_style(plt, style_name='ggplot', figsize=(12,6)):
  """Sets matplotlib stylesheet with custom colors
  """

  plt.style.use(style_name)

  plt.rcParams['font.family'] = 'sans-serif'
  plt.rcParams['font.serif'] = 'Helvetica'
  plt.rcParams['font.monospace'] = 'Andale Mono'
  plt.rcParams['font.size'] = 14
  plt.rcParams['axes.labelsize'] = 14
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams['axes.titlepad'] = 20
  plt.rcParams['axes.labelpad'] = 14
  plt.rcParams['axes.titlesize'] = 18
  plt.rcParams['xtick.labelsize'] = 12
  plt.rcParams['ytick.labelsize'] = 12
  plt.rcParams['legend.fontsize'] = 12
  plt.rcParams['figure.titlesize'] = 16

  cycler = mpl_cycler('color', ['#0A1EFF', '#1EBAA8', '#CABD84', '#BC8D49', '#C04D3C', '#8EBA42', '#FFB5B8'])
  plt.rcParams['axes.prop_cycle'] = cycler
  


# -----------------------------------------------------------------------------
#
# Drawing utils
#
# -----------------------------------------------------------------------------

fonts = {}
log = logging.getLogger('vframe')
color_red = Color.from_rgb_int((255, 0, 0))
color_green = Color.from_rgb_int((0, 255, 0))
color_blue = Color.from_rgb_int((0, 0, 255))
color_white = Color.from_rgb_int((255, 255, 255))
color_black = Color.from_rgb_int((0, 0, 0))



# -----------------------------------------------------------------------------
#
# Rotated BBox
#
# -----------------------------------------------------------------------------

def draw_rotated_bbox_cv(im, rbbox_norm, stroke_weight=2, color=color_green):
  """Draw rotated bbox using opencv
  """
  if im_utils.is_pil(im):
    im = im_utils.pil2np()
    was_pil = True
  else:
    color.swap_rb()
    was_pil = False

  dim = im.shape[:2][::-1]
  rbbox_dim = rbbox_norm.to_rbbox_dim(dim)
  vertices = rbbox_dim.vertices
  color_rgb = color.rgb_int
  for i, p in enumerate(vertices):
    p1 = vertices[i]
    p2 = vertices[(i + 1) % 4]
    if stroke_weight == -1:
      im = cv.line(im, p1.xy, p2.xy, color_rgb, 0, cv.LINE_AA, -1)
    else:
      im = cv.line(im, p1.xy, p2.xy, color_rgb, stroke_weight, cv.LINE_AA)

  if was_pil:
    im = im_utils.pil2np(im)

  return im



def draw_rotated_bbox_pil(im, rbbox_norm, stroke_weight=2, color=color_green, expand=0.0):
  """Draw rotated bbox using PIL
  """
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    was_np = True
  else:
    was_np = False

  # TODO implement expand on rbbox
  rbbox_dim = rbbox_norm.to_rbbox_dim(im.size)
  points = rbbox_dim.vertices
  vertices = [p.xy for p in points]
  color_rgb = color.rgb_int
  canvas = ImageDraw.Draw(im)

  if stroke_weight == -1:
    canvas.polygon(vertices, fill=color_rgb)
  else:
    canvas.polygon(vertices, outline=color_rgb)

  del canvas

  if was_np:
    im = im_utils.pil2np(im)

  return im


# -----------------------------------------------------------------------------
#
# Segmentation masks
#
# -----------------------------------------------------------------------------

def draw_mask(im, bbox_norm, mask, threshold=0.3,  mask_blur_amt=21, color=color_green, blur_amt=None, color_alpha=0.6):
  """Draw image mask overlay
  """
  dim = im.shape[:2][::-1]
  bbox_dim = bbox_norm.to_bbox_dim(dim)
  x1, y1, x2, y2 = bbox_dim.xyxy
  mask = cv.resize(mask, bbox_dim.wh, interpolation=cv.INTER_NEAREST)
  mask = cv.blur(mask, (mask_blur_amt, mask_blur_amt))
  mask = (mask > threshold)
  # extract the ROI of the image
  roi = im[y1:y2,x1:x2]
  if blur_amt is not None:
    roi = cv.blur(roi, (blur_amt, blur_amt))
  roi = roi[mask]
  if color is not None:
    color_rgb = color.rgb_int[::-1]  # rgb to bgr
    roi = ((color_alpha * np.array(color_rgb)) + ((1 - color_alpha) * roi))
  # store the blended ROI in the original image
  im[y1:y2,x1:x2][mask] = roi.astype("uint8")
  return im



# -----------------------------------------------------------------------------
#
# BBoxes
#
# -----------------------------------------------------------------------------

# def draw_bbox_cv(im, bboxes, color, stroke_weight):
#   """Draws BBox onto Numpy image using np broadcasting
#   :param bbox: BBoxDiim
#   :param color: Color
#   :param stroke_weight: int
#   """
#   for bbox in bboxes:
#     im = cv.rectangle(im, bbox_dim.p1.xy, bbox_dim.p2.xy, color, stroke_weight)
#   return im


# def _draw_bbox_np(im, bboxes, color, stroke_weight):
#   """Draws BBox onto cv image using np broadcasting
#   :param bbox: BBoxDiim
#   :param color: Color
#   :param stroke_weight: int
#   """
#   for bbox in bboxes:
#     im[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = color.bgr_int
#   return im


def _draw_bbox_pil(canvas, bboxes, color, stroke_weight):
  """Draws BBox onto PIL.ImageDraw
  :param bbox: BBoxDiim
  :param color: Color
  :param stroke_weight: int
  :returns PIL.ImageDraw
  """
  for bbox in bboxes:
    xyxy = (bbox.p1.xy, bbox.p2.xy)  
    if stroke_weight == -1:
      canvas.rectangle(xyxy, fill=color.rgb_int)
    else:
      canvas.rectangle(xyxy, outline=color.rgb_int, width=stroke_weight)
  return canvas
  

def draw_bbox(im, bboxes, color=None, stroke_weight=None, expand=None,
  label=None, text_color=None, text_size=None, text_bg_color=None, text_bg_padding=None):
  """Draws BBox on image
  :param im: PIL.Image or numpy.ndarray
  :param bboxes: list(BBoxNorm)
  :param color: Color
  :param label: String
  :param stroke_weight: int
  :param text_size: int
  :param expand: float
  """
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    dim = im.size
    was_np = True
  else:
    was_np = False

  if not type(bboxes) == list:
    bboxes = [bboxes]
  for i in range(len(bboxes)):
    bbox = bboxes[i]
    if expand is not None:
      bbox = bbox.expand(expand)
    bboxes[i] = bbox.to_bbox_dim(dim)

  color = app_cfg.GREEN if color is None else color
  stroke_weight = app_cfg.DEFAULT_STROKE_WEIGHT if stroke_weight is None else stroke_weight
  canvas = ImageDraw.ImageDraw(im)

  # draw label
  if label:
    text_bg_color = color if text_bg_color is None else text_bg_color
    text_bg_padding = int(0.2 * text_size) if text_bg_padding is None else text_bg_padding
    # bbox of background
    pt_bbox = PointDim(*bbox.xy, dim)
    bbox_dim_text = bbox_from_label(pt, dim, label, font, text_size, text_bg_padding)
    if bbox_dim_text.y1 > bbox_dim_text.h:
      # move above main bbox
      bbox_dim_text.move(0, bbox_dim_text.h)
    text_pt = PointDim(*bbox_dim_text.xy).move(text_bg_padding, text_bg_padding)
    _draw_bbox_pil(canvas, bbox_bg_norm, text_bg_color, -1)
    # point of text origin
    _draw_text_pil(canvas, label, text_color, text_bg_color, text_bg_padding)

  # draw bbox
  _draw_bbox_pil(canvas, bboxes, color, stroke_weight)

  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)
  return im





# -----------------------------------------------------------------------------
#
# Text
#
# -----------------------------------------------------------------------------

def bbox_from_label(pt, dim, label, font, text_size, padding):
  """Creates BBoxDim based on label, font size, and padding
  """
  tw, th = font.getsize(label)
  return BBoxDim.from_xywh_dim((pt.x, pt.y, tw + padding, th + padding), dim)


def _draw_text_pil(canvas, bboxes, color, label, text_size=None, text_color=None, bg_color=None, bg_padding=None, 
  knockout_size=None, knockout_color=None):
  """Draws bbox and annotation
  :param im: PIL.Image or numpy.ndarray
  :param bboxes: list(BBoxDim)
  :param color: Color
  :param label: String
  :param stroke_weight: int
  :param text_size: int
  :param expand: float
  """

  bg_color = app_cfg.BLACK if not bg_color else bg_color
  text_color = bg_color.get_fg_color() if not text_color else text_color

  for bbox in bboxes:
    

    xyxy = bbox.xyxy
    if stroke_weight == -1:
      canvas.rectangle(xyxy, fill=fill_color)
    else:
      canvas.rectangle(xyxy, outline=fill_color, width=stroke_weight)

    # draw label
    label = label.upper()
    font = text_mngr.get_font(text_size)
    tw, th = font.getsize(label)

    pad_left, pad_top = (int(0.075 * th), int(0.1 * th))
    tbw, tbh = (tw + pad_left, th + pad_top)

    if tbh > bbox.y1:
      # draw inside
      xyxy = (bbox.x1, bbox.y1, bbox.x1 + tbw, bbox.y1 + tbh)
      canvas.rectangle(xyxy, fill=fill_color)
      x1, y1, x2, y2 = xyxy
      canvas.text((x1 + pad_left, y1 + pad_top), label, text_color, font)
    else:
      # draw on top
      xyxy = (bbox.x1, bbox.y1 - pad_top, bbox.x1 + tbw, bbox.y1)
      x1,y1,x2,y2 = xyxy
      canvas.rectangle(xyxy, fill=fill_color)
      canvas.text((x1 + (pad_left * tw), y1 + (pad_top * th)), label, text_color, font)


def _dalt_raw_text_pil(canvas, text, point, text_size, color, font, padding=None, bg_color=None, knockout=None):
  if padding is not None:
    rgb_bg = app_cfg.BLACK if not bg_color else bg_color
    point_dim = point.to_point_dim(dim)
    font = text_mngr.get_font(text_size)
    tw, th = font.getsize(text)
    rgb = color.rgb_int
    #th_offset = int(text_height_adj_factor * th) 
    #tw_offset = 0 - int(0.0 * tw)
    tw_offset = 0
    th_offset = 0

    xyxy = (pt.x, pt.y + th_offset, pt.x + tw + tw_offset, pt.y + th)
    xyxy = BBoxDim(*xyxy, dim).expand_px(padding).xyxy
    canvas.rectangle(xyxy, fill=rgb)


  rgb = color.rgb_int
  canvas.text((pt.x, pt.y), text, rgb, font)


def draw_text(im, bboxes, color, label, text_size=None, text_color=None, bg_color=None, bg_padding=None):
  """Draws label with background
  :param im: PIL.Image or numpy.ndarray
  :param bboxes: list(BBoxDim)
  :param color: Color
  :param label: String
  :param stroke_weight: int
  :param text_size: int
  :param expand: float
  """
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    dim = im.size
    was_np = True
  else:
    was_np = False

  if not type(bboxes) == list:
    bboxes = [bboxes]
  for i in range(len(bboxes)):
    bbox = bboxes[i]
    if expand is not None:
      bbox = bbox.expand(expand)
    bboxes[i] = bbox.to_bbox_dim(dim)

  canvas = ImageDraw.ImageDraw(im)

  for bbox in bboxes:
    self._draw_text_pil(canvas, color, label, text_size=None, text_color=None, bg_color=None, bg_padding=None)

  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)
  return im








# -----------------------------------------------------------------------------
#
# init instances
#
# -----------------------------------------------------------------------------

text_mngr = FontManager()


# def draw_text_cv(im, text, pt, size=1.0, color=None):
#   """Draws degrees as text over image
#   """
#   if im_utils.is_pil(im):
#     im = im_utils.pil2np(im)
#     was_pil = True
#   else:
#     was_pil = False

#   dim = im.shape[:2][::-1]
#   pt_dim = pt.to_point_dim(dim)
#   color = app_cfg.GREEN if not color else color
#   rgb = color.rgb_int
#   cv.putText(im, text, pt_dim.xy, cv.text_HERSHEY_SIMPLEX, size, rgb, thickness=1, lineType=cv.LINE_AA)

#   if was_pil:
#     im = im_utils.pil2np(im)

#   return im