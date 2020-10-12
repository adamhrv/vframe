############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click


@click.command('')
@click.option('-i', '--input', 'opt_inputs', required=True,
  multiple=True,
  help="Input files to merge")
@click.option('-o', '--output', 'opt_output', required=True,
  help='Output file')
@click.option('--minify', 'opt_minify', is_flag=True,
  default=False,
  help='Minify JSON')
@click.pass_context
def cli(ctx, opt_inputs, opt_output, opt_minify):
  """Merge JSON detections"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.settings import app_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # load first file
  opt_inputs = list(opt_inputs)
  fp_first = opt_inputs.pop(0)
  data =  file_utils.load_json(fp_first)
  results = {}

  # merge 
  for fp_in in tqdm(opt_inputs, desc='Files'):

    # load json
    detections = file_utils.load_json(fp_in)
    detections_lkup = {d['filepath']: d for d in detections}

    # add all the current detections to cumulative detections
    for filepath, result in detections_lkup.items():
      if not filepath in results.keys():
          results[filepath] = {'filepath': filepath}
      for frame_idx, frame_data in result['frames_data'].items():
        if not 'frames_data' in results[filepath].keys():
          results[filepath]['frames_data'] = {}
        if not frame_idx in results[filepath]['frames_data'].keys():
          results[filepath]['frames_data'][frame_idx] = {}
        results[filepath]['frames_data'][frame_idx].update(frame_data)

  # write
  results_out = list(results.values())
  file_utils.write_json(results_out, opt_output, minify=opt_minify)

