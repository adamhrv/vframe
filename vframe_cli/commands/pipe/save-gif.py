############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils import click_utils
from vframe.models.types import FrameImage, FrameImageVar, VideoFileExt, VideoFileExtVar
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-f', '--frame', 'opt_frame_type', default='draw',
  type=FrameImageVar,
  help=click_utils.show_help(FrameImage))
@processor
@click.pass_context
def cli(ctx, pipe, opt_dir_out, opt_frame_type):
  """Save video frames to animated GIF"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.utils import file_utils
  

  # ---------------------------------------------------------------------------
  # initialize

  file_utils.ensure_dir(opt_dir_out)
  ext = opt_ext.name.lower()

  frame_count = 0
  filepath = None
  is_writing = False
  video_out = None

  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    pipe_item = yield
    header = ctx.obj['header']

    if header.frame_index == header.frame_end and video_out is not None:
      # end writing video to container file
      #video_out.release()
      is_writing = False
      filepath = None

    # start new video if new headers
    if header.filepath != filepath and header.frame_index == 0:
      filepath = header.filepath
      fn = Path(header.filename).stem
      fp_out = join(opt_dir_out, f'{fn}.{ext}')
      if opt_frame_type == FrameImage.ORIGINAL:
        dim =  header.dim
      elif opt_frame_type == FrameImage.DRAW:
        dim = header.dim_draw

      # start writing video to container file
      #video_out = cv.VideoWriter(fp_out, four_cc, header.fps, tuple(dim))
      is_writing = True

    if is_writing:
      im = pipe_item.get_image(FrameImage.DRAW)
      # write frame
      #video_out.write(im)


    pipe.send(pipe_item)