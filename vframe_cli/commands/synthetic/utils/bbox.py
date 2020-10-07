############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.settings import app_cfg


opts_sources = [app_cfg.DN_REAL, app_cfg.DN_MASK, app_cfg.DN_COMP, app_cfg.DN_BBOX]

@click.command()
@click.option('-i', '--input', 'opt_dir_render', required=True)
@click.option('--type', 'opt_type', type=click.Choice(opts_sources),
  default=app_cfg.DN_COMP,
  help='Output dir')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', default=12,
  help='Number threads')
@click.option('--font-size', 'opt_font_size', default=14)
@click.option('--bbox-norm', 'opt_use_bbox_norm', is_flag=True,
  help="Use old annotation bbox norm format")
@click.pass_context
def cli(ctx, opt_dir_render, opt_type, opt_slice, opt_threads, opt_font_size, opt_use_bbox_norm):
  """Generates bounding box images"""
  
  from os.path import join

  from PIL import Image
  import pandas as pd
  from glob import glob
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool

  from vframe.utils import file_utils, draw_utils
  from vframe.models.color import Color
  from vframe.models.geometry import BBox

  log = app_cfg.LOG
  log.info('Draw annotations')

  file_utils.ensure_dir(join(opt_dir_render, app_cfg.DN_BBOX))

  # glob images
  dir_glob = str(Path(opt_dir_render) / opt_type / '*.png')
  fps_ims = sorted(glob(dir_glob))
  if any(opt_slice):
    fps_ims = fps_ims[opt_slice[0]:opt_slice[1]]
  log.info(f'found {len(fps_ims)} images in {dir_glob}')

  # load annotation meta
  fp_annos = join(opt_dir_render, app_cfg.FN_ANNOTATIONS)
  log.debug(f'Load: {fp_annos}')
  try:
    df_annos = pd.read_csv(fp_annos)
  except Exception as e:
    log.warn('No annotations. Exiting')
    return

  def pool_worker(fp_im):
    # load image
    im = Image.open(fp_im)
    dim = im.size
    fn = Path(fp_im).name

    # group by filename
    df_fn = df_annos[df_annos.filename == fn]
    #df_fn = df_fn[df_fn.label_index != 0]

    if not len(df_fn) > 0:
      log.warning(f'No annotations in: {fn}')
    # draw bboxes
    for rf in df_fn.itertuples():
      if rf.label == 'Background':
        continue
      if opt_use_bbox_norm:
        bbox = BBox.from_xyxy_norm(rf.x1, rf.y1, rf.x2, rf.y2, *dim)
      else:
        bbox = BBox(rf.x1, rf.y1, rf.x2, rf.y2, *dim)
      #color = Color.from_rgb_hex(rf.color)
      color = Color.from_rgb_int((255,255,255))
      #bbox_nlc = bbox_norm.to_labeled(rf.label, rf.label_index, rf.filename).to_colored(color)
      im = draw_utils.draw_bbox(im, bbox, color=color, label=rf.label, size_label=opt_font_size)

    # write file
    fp_out = join(opt_dir_render, app_cfg.DN_BBOX, Path(fp_im).name)
    im.save(fp_out)

  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, fps_ims), total=len(fps_ims)))


  