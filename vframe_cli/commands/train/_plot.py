############################################################################# 
#
# VFRAME Training
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

"""Create a YOLOV3 Object Detection project
"""


import click

@click.command()
@click.option('-i', '--input', 'opt_fp_log', required=True,
  help='Path to logfile')
@click.option('-x', '--x', 'opt_x', type=(int, int), default=(None, None))
@click.option('-y', '--y', 'opt_y', type=(float, float), default=(None, None))
@click.pass_context
def cli(ctx, opt_fp_log, opt_x, opt_y):
  """Plots YOLO training logfile"""

  from vframe.utils import log_utils, file_utils
  from vframe.utils.yolo_utils import YoloProject


  log = log_utils.Logger.getLogger()
  log.info(f'Generate YOLO logfile plot: {opt_fp_log}')
  
  YoloProject.create_plot(opt_fp_log, xmin_max=opt_x, ymin_max=opt_y)