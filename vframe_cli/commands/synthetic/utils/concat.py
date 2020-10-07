############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

ext_choices = ['jpg', 'png']

@click.command()
@click.option('-i', '--input', 'opt_dirs_in', required=True, multiple=True,
  help='Path to directory of images')
@click.option('--subdirs', 'opt_subdirs', is_flag=True,
  help='Glob all sub directories for each input directory, but not recursively')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output dir')
@click.option('-e', '--ext', 'opt_ext', default='png',
  type=click.Choice(ext_choices),
  help='Path to output dir')
@click.option('--symlink/--copy', 'opt_symlink', is_flag=True,
  default=True,
  help='Symlink or copy images to new directory'
  )
@click.pass_context
def cli(ctx, opt_dirs_in, opt_subdirs, opt_dir_out, opt_ext, opt_symlink):
  """Concatenate multiple render directories"""

  from os.path import join
  from pathlib import Path
  from glob import glob

  import pandas as pd
  from tqdm import tqdm

  from vframe.settings import app_cfg
  from vframe.utils import file_utils
  from vframe_synthetic.settings import plugin_cfg

  log = app_cfg.LOG
  dir_images = join(opt_dir_out, plugin_cfg.DN_IMAGES)
  file_utils.ensure_dir(dir_images)

  dirs_input = []
  if opt_subdirs:
    for d in opt_dirs_in:
      dirs_input += glob(join(d, '*'))

  log.info(f'Concatenating {len(dirs_input)} directories')
  dfs = []

  for dir_in in dirs_input:
    log.debug(dir_in)
    # concat dataframe
    fp_annos = join(dir_in, plugin_cfg.FN_ANNOTATIONS)
    if not Path(fp_annos).is_file():
      log.warn(f'{fp_annos} does not exist. Skipping')
      continue
    _df = pd.read_csv(fp_annos)
    dfs.append(_df)

    # symlink real images
    for sf in _df.itertuples():
      fp_src = join(dir_in, plugin_cfg.DN_REAL, sf.filename)
      fpp_dst = Path(join(dir_images, sf.filename))
      if fpp_dst.is_symlink():
        fpp_dst.unlink()
      fpp_dst.symlink_to(fp_src)

  df = pd.concat(dfs)

  fp_out = join(opt_dir_out, plugin_cfg.FN_ANNOTATIONS)
  df.to_csv(fp_out, index=False)
  log.info(f'Wrote new annotations file with {len(df):,} items')