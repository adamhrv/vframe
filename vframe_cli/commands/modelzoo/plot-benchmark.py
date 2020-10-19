############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

plot_types = ['bar', 'line']

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to benchmark CSV')
@click.option('-o', '--output', 'opt_output',
  help="Path to output PNG")
@click.option('-t', '--type', 'opt_plot_type', 
  type=click.Choice(plot_types),
  default='bar', show_default=True,
  help='Type of plot to generate')
@click.option('--dpi', 'opt_dpi', 
  default=72, show_default=True,
  help="Pixels per inch resolution for output")
@click.option('--figsize', 'opt_figsize',
  default=(1280, 720), show_default=True,
  help="matplotlib figure size (pixels")
@click.option('--prefix', 'opt_prefix', 
  default='plot', show_default=True,
  help='Filename prefix')
@click.option('--title/--no-title', 'opt_title', is_flag=True, default=True,
  help='Show title')
@click.option('--sort/--no-sort', 'opt_sort', is_flag=True,
  default=True,
  help="Sort")
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help="Overwrite file")
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_plot_type, opt_dpi, opt_figsize, 
  opt_prefix, opt_title, opt_sort, opt_force):
  """Plot benchmark FPS results"""

  # ------------------------------------------------
  # imports

  import os
  from os.path import join
  from glob import glob
  from dataclasses import asdict
  from operator import itemgetter
  from pathlib import Path

  import matplotlib.pyplot as plt
  import matplotlib
  import numpy as np
  import pandas as pd
   
  from vframe.settings import app_cfg
  from vframe.utils import log_utils, file_utils
  from vframe.utils.draw_utils import pixels_to_figsize, set_matplotlib_style

  log = app_cfg.LOG

  if not opt_output:
    ext = file_utils.get_ext(opt_input)
    opt_output = opt_input.replace(ext, 'png')
  file_utils.ensure_dir(opt_output)
  if Path(opt_output).is_file() and not opt_force:
    log.error(f'File exists {opt_output}. Use "-f/--force" to overwrite')
    return

  df = pd.read_csv(opt_input)


  # set styles
  set_matplotlib_style(plt)

  # labels
  opencv_ver = df.opencv_version.values[0]
  processor = 'GPU' if df.processor.values[0] == 'gpu' else 'GPU'
  title = f'DNN FPS Benchmark (OpenCV: {opencv_ver} {processor})'
  ylabel ="Frames Per Second"
  xlabel ="DNN Model"

  # fig setup
  fig, ax = plt.subplots()
  fig.dpi = opt_dpi
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)
  
  if opt_plot_type == 'bar':

    # proprocess
    items = {}
    for i, row in df.iterrows():
      label = f'{row.model}: {row.dnn_width}x{row.dnn_height}'
      items[label] = row.fps

    if opt_sort:
      items_sorted = sorted(items.items(), key=itemgetter(1), reverse=True)
      items = {k:v for k,v in items_sorted }

    # plot
    ax.bar(list(items.keys()), list(items.values()))


  elif opt_plot_type == 'line':

    ymax = 0

    # proprocess
    for model_name, group in df.groupby('model'):  
      items = {}
      values = group.fps.values
      ymax = max(ymax, max(values))
      for i, row in group.iterrows():
        label = f'{row.dnn_width}x{row.dnn_height}'
        items[label] = row.fps

      if opt_sort:
        items_sorted = sorted(items.items(), key=itemgetter(1), reverse=True)
        items = {k:v for k,v in items_sorted }
      
      # plot
      ax.plot(list(items.keys()), list(items.values()), label=f'Model: {model_name}', marker="s")
        
    # slightly increase y-margin
    ax.set_ylim([0, 1.05 * ymax])

    # display
    ax.legend(loc='upper right')

  # save figure
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(opt_output, dpi=opt_dpi)
