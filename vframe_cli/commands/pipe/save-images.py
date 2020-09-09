############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models import types
from vframe.utils import click_utils
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-e', '--ext', 'opt_ext', default=None,
  type=types.ImageFileExtVar,
  help=click_utils.show_help(types.ImageFileExt))
@click.option('-f', '--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--prefix', 'opt_prefix', default='',
  help='Filename prefix')
@click.option('--suffix', 'opt_suffix', default='',
  help='Filename suffix')
@click.option('--numbered', 'opt_numbered', is_flag=True,
  help='Number files sequentially')
@click.option('-q', '--quality', 'opt_quality', default=100, type=click.IntRange(0,100, clamp=True),
  show_default=True,
  help='JPEG write quality')
@click.option('--subdirs', 'opt_keep_subdirs', is_flag=True,
  help='Keep subdirectory structure in output directory')
@processor
@click.pass_context
def cli(ctx, pipe, opt_dir_out, opt_ext, opt_frame_type, opt_prefix, opt_suffix,
  opt_numbered, opt_quality, opt_keep_subdirs):
  """Save to images"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.utils import file_utils


  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  file_utils.ensure_dir(opt_dir_out)
  frame_count = 0


  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(opt_frame_type)

    # filename options
    if opt_numbered:
      stem = file_utils.zpad(frame_count)
      frame_count += 1
    else:
      stem = Path(header.filename).stem

    # default to same extension unless extension is optioned
    if not opt_ext:
      ext = file_utils.get_ext(header.filename)
    else:
      ext = opt_ext.name.lower()
    fn = f'{opt_prefix}{stem}{opt_suffix}.{ext}'

    # if subdirs is optioned, copy subdirs from optioned value
    if opt_keep_subdirs:
      path_in = Path(ctx.obj['fp_input'])
      path_out = Path(opt_dir_out)
      path_file = Path(header.filepath)
      fp_subdir = str(path_file.relative_to(path_in).parent)
      log.debug(fp_subdir)
      fp_out = join(opt_dir_out, fp_subdir, fn)
    else:
      fp_out = join(opt_dir_out, fn)

    file_utils.ensure_dir(fp_out)
    # write image
    if ext == 'jpg':
      cv.imwrite(fp_out, pipe_item.get_image(opt_frame_type), [int(cv.IMWRITE_JPEG_QUALITY), opt_quality])
    else:
      cv.imwrite(fp_out, pipe_item.get_image(opt_frame_type))

    # continue pipestream
    pipe.send(pipe_item)