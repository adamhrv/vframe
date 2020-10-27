############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

@click.command('')
@click.option('-i', '--input', 'opt_fp_cfg', required=True,
  help='Path YAML job config')
@click.pass_context
def cli(ctx, opt_fp_cfg):
  """Creates new YOLO project from config file"""

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import load_yaml
  from vframe.models.training_dataset import YoloProjectConfig
  from vframe.utils.yolo_utils import YoloProject


  app_cfg.LOG.info(f'YOLO project from: {opt_fp_cfg}')
  
  # load config file
  cfg = load_yaml(opt_fp_cfg, data_class=YoloProjectConfig)

  # generate YOLO project files
  YoloProject.from_cfg(cfg)