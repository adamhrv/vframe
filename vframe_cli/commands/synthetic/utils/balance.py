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

ext_choices = ['jpg', 'png']

@click.command()
@click.option('-i', '--input', 'opt_dir_ims', required=True,
  help='Path to balanced diretory images')
@click.option('--dry-run/--delete', 'opt_dry_run', is_flag=True, default=True,
  help='Dry run, do not delete any files')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(ctx, opt_dir_ims, opt_dry_run, opt_verbose):
  """Balance image directories after review"""

  import os
  from os.path import join
  from pathlib import Path
  from glob import glob

  from tqdm import tqdm

  from vframe_synthetic.settings import plugin_cfg

  log = app_cfg.LOG

  fp_ims_real = sorted([im for im in glob(str(Path(opt_dir_ims) / plugin_cfg.DN_REAL / '*.png'))])
  fp_ims_mask = sorted([im for im in glob(str(Path(opt_dir_ims) / plugin_cfg.DN_MASK / '*.png'))])

  log.info(f'Real: {len(fp_ims_real):,}. Masks: {len(fp_ims_mask):,}')

  if len(fp_ims_real) == len(fp_ims_mask):
    log.info('Same number of images.')
    return
  
  n_delete = len(fp_ims_mask) - len(fp_ims_real)

  if opt_dry_run:
    log.info(f'Not deleting {n_delete:,} images. Use "--delete" flag to remove images.')
  else:
    log.info(f'Deleting {n_delete:,} images')

  # list of real image names to check
  fns_real = [Path(fp).name for fp in fp_ims_real]

  # Check if mask image names exist in real dir
  for fp_im_mask in tqdm(fp_ims_mask):
    # delete if not paired
    if not Path(fp_im_mask).name in fns_real:
      # remove
      if opt_dry_run:
        msg = f'Not removing: {fp_im_mask}'
      else:
      # verbote
        os.remove(fp_im_mask)
        msg = f'Removed: {fp_im_mask}'
      if opt_verbose:
        log.info(msg)




