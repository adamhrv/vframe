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

@click.command('')
@click.option('-c', '--color', 'opt_color', 
  default=(0,255,0),
  help='font color in RGB int (eg 0 255 0)')
@click.option('-t', '--text', 'opt_caption', required=True,
  help='Caption text')
@click.option('-x', '--x', 'opt_x', required=True, default=20,
  help='X position in pixels. Use negative for distance from bottom.')
@click.option('-y', '--y', 'opt_y', required=True, default=-20,
  help='Y position in pixels. Use negative for distance from bottom.')
@click.option('-a', '--alpha', 'opt_alpha', default=1.0,
  help='Opacity of font')
@click.option('--font-size', 'opt_font_size', default=14,
  help='Font size for labels')
@click.option('--knockout', 'opt_knockout', default=None, type=int,
  help='Knockout pixel distance')
@click.option('--bg', 'opt_bg', is_flag=True,
  help='Add text background')
@click.option('--bg-color', 'opt_bg_color', default=(0,0,0))
@click.option('--bg-padding', 'opt_bg_padding', default=None, type=int)
@processor
@click.pass_context
def cli(ctx, pipe, opt_caption, opt_x, opt_y, opt_color, opt_font_size, 
  opt_alpha, opt_knockout, opt_bg, opt_bg_color, opt_bg_padding):
  """Add text caption"""
  
  """
  TODO: 
  - add background
  """

  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.models.color import Color
  from vframe.models.bbox import PointNorm
  from vframe.utils import im_utils, draw_utils
  
  # ---------------------------------------------------------------------------
  # initialize

  color_txt = Color.from_rgb_int(opt_color)
  color_bg = Color.from_rgb_int(opt_bg_color)


  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(types.FrameImage.DRAW)

    # draw text
    h,w,c = im.shape
    xy = [opt_x, opt_y]

    if xy[0] < 0:
      xy[0] = w + xy[0]
    
    if xy[1] < 0:
      xy[1] = h + xy[1]

    pt_norm = PointNorm(xy[0] / w, xy[1] / h)

    # draw background if optioned
    if opt_bg:
      im = draw_utils.draw_text_bg(im, opt_caption, pt_norm, opt_font_size,
       padding=opt_bg_padding, color=color_bg)
    
    # draw text
    im = draw_utils.draw_label(im, opt_caption, pt_norm, opt_font_size, color_txt, 
      knockout=opt_knockout)
    
    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)

