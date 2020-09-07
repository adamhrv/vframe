############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.utils.click_utils import processor
from vframe.utils.click_utils import show_help
from vframe.models.types import ModelZooClickVar, ModelZoo, FrameImage
from vframe.settings import app_cfg

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=app_cfg.DEFAULT_DETECT_MODEL,
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=True,
  help='Use GPU or CPU for inference')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('-r', '--rotate', 'opt_rotate', 
  type=click.Choice(app_cfg.ROTATE_VALS.keys()), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, pipe, opt_model_enum, opt_data_key, opt_gpu, 
  opt_dnn_threshold, opt_dnn_size, opt_rotate, opt_verbose):
  """Detect objects"""
  
  from os.path import join
  from pathlib import Path
  import traceback

  import cv2 as cv

  from vframe.models.dnn import DNN
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  # ---------------------------------------------------------------------------
  # initialize
  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  # override dnn_cfg vars with cli vars
  if opt_gpu:
    dnn_cfg.use_gpu()
  else:
    dnn_cfg.use_cpu()
  if all(opt_dnn_size):
    dnn_cfg.width = opt_dnn_size[0]
    dnn_cfg.height = opt_dnn_size[1]
  if opt_dnn_threshold is not None:
    dnn_cfg.threshold = opt_dnn_threshold
    
  # rotate cv, np vals
  cv_rot_val = app_cfg.ROTATE_VALS[opt_rotate]
  np_rot_val =  int(opt_rotate) // 90  # counter-clockwise 90 deg rotations

  if not opt_data_key:
    opt_data_key = model_name

  # create dnn cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # ---------------------------------------------------------------------------
  # process

  while True:

    # get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(FrameImage.ORIGINAL)
    
    # rotate if optioned  
    if cv_rot_val is not None:
      im = cv.rotate(im, cv_rot_val)
    
    # detect
    try:
      results = cvmodel.infer(im)

      # rotate if optioned
      if results and np_rot_val != 0:
        for detect_results in results.detections:
          detect_results.bbox = detect_results.bbox.rot90(np_rot_val)

    except Exception as e:
      results = {}
      app_cfg.LOG.error(f'Could not detect: {header.filepath}')
      tb = traceback.format_exc()
      app_cfg.LOG.error(tb)

    # debug
    if opt_verbose:
      app_cfg.LOG.debug(f'{cvmodel.dnn_cfg.name} detected: {len(results.detections)} objects')

    # update data
    if results:
      pipe_data = {opt_data_key: results}
      header.add_data(pipe_data)
    
    # continue processing
    pipe.send(pipe_item)