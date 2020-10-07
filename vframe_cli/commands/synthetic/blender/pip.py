############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from os.path import join
from pathlib import Path
import click

from vframe.models import types
from vframe.settings import app_cfg
from vframe.utils import click_utils


@click.command()
@click.option('-r', '--requirement', 'opt_fp_in', help='Path to a requirements.txt')
@click.option('-i', '--install', 'opt_packages', multiple=True, help='Python packages')
@click.option('--blender', 'opt_fp_blender', default=app_cfg.FP_BLENDER_BIN,
  help='Path to Blender binary', show_default=True)
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_blender, opt_packages):
  """Blender pip installer """
  
  import os
  import subprocess

  log = app_cfg.LOG
  log.info('Blender PIP')

  cmds = []
  version = app_cfg.BLENDER_VERSION
  dir_parent = Path(opt_fp_blender).parent

  # check for and ensure Blender pip was installed
  fp_pip_bin = join(dir_parent, f'{version}/python/bin/pip3.7')
  if not Path(fp_pip_bin).is_file():
    log.info('PIP was not yet installed. Ensuring pip...')
    fp_python_bin = join(dir_parent, f'{version}/python/bin/python3.7m')
    fp_ensure_bin = join(dir_parent, f'{version}/python/lib/python3.7/ensurepip')
    cmd = [fp_python_bin, fp_ensure_bin]
    cmds.append(cmd)

  if opt_fp_in:
    fp_cur = os.path.realpath(opt_fp_in)
    args = [fp_pip_bin, 'install', '-r', fp_cur]
    cmds.append(args)
  elif opt_packages:
    for package in opt_packages:
      args = [fp_pip_bin, 'install', package]
      cmds.append(args)

  if not len(cmds):
    log.error('"-i/--install" name of packge. Or type --help')
  else:
    for cmd in cmds:
      log.debug(' '.join(cmd))
      subprocess.call(cmd, stdin=None, stdout=None, stderr=None, shell=False)