#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################


import click

@click.command()
@click.option('-i', '--input', 'opt_fp_cfg', required=True,
  help='Path YAML job config')
@click.pass_context
def cli(ctx, opt_fp_cfg):
  """YOLO PyTorch project"""

  from os.path import join
  import random
  from pathlib import Path
  import operator
  import shutil

  import pandas as pd
  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.utils.dataset_utils import split_train_val_test
  from vframe.models.annotation import Annotation

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import load_yaml
  from vframe.models.training_dataset import YoloPyTorch



  log = app_cfg.LOG
  log.info(f'YOLO PyTorch project from: {opt_fp_cfg}')

  # load config file
  cfg = load_yaml(opt_fp_cfg, data_class=YoloPyTorch)

  # load annotations and convert to BBox class
  df = pd.read_csv(cfg.annotations)
  df_pos = df[df.label_enum != app_cfg.LABEL_BACKGROUND]
  df_neg = df[df.label_enum == app_cfg.LABEL_BACKGROUND]

  # ensure output directory
  file_utils.ensure_dir(cfg.output)

  # images/
  dir_images = join(cfg.output, cfg.images_labels)
  dir_labels = dir_images
  file_utils.ensure_dir(dir_images)

  # create dict of classes, then sort by index, 0 - N
  df_sorted = df_pos.sort_values(by='label_index', ascending=True)
  df_sorted.drop_duplicates(['label_enum'], keep='first', inplace=True)
  class_labels = df_sorted.label_enum.values.tolist()

  # write labels text file
  fp_classes = join(cfg.output, app_cfg.FN_LABELS)
  file_utils.write_txt(class_labels, fp_classes)

  # Generate one label per file with all bboxes and classes
  # <object-class> <x_center> <y_center> <width> <height>
  labels_data = {}
  file_list = []
  df_im_groups = df_pos.groupby('filename')
  for fn, df_im_group in df_im_groups:
    annos = []
    file_list.append(join(dir_images, fn))
    for row_idx, row in df_im_group.iterrows():
      annos.append(Annotation.from_anno_series_row(row).to_darknet_str())
    labels_data.update({fn: annos})

  # write labels and symlink images
  for fn, annos in tqdm(labels_data.items()):
    fp_label = join(dir_labels, file_utils.replace_ext(fn, 'txt'))
    file_utils.write_txt(annos, fp_label)
    fpp_im_dst = Path(join(dir_images, fn))
    fpp_im_src = Path(join(cfg.images, fn))
    if cfg.use_symlinks:
          fpp_im_dst.unlink()
          if fpp_im_dst.is_symlink():
        fpp_im_dst.symlink_to(fpp_im_src)
    else:
      shutil.copy(fpp_im_src, fpp_im_dst)

  # Generate training list of
  splits = split_train_val_test(annos, splits=(0.6, 0.2, 0.2), seed=1)

  # write txt files for train, val
  file_utils.write_txt(splits['train'], join(cfg.output, cfg.train))
  file_utils.write_txt(splits['val'], join(cfg.output, cfg.valid))
  file_utils.write_txt(splits['test'], join(cfg.output, cfg.test))
