#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

# %% codecell
import os
import sys
from os.path import join
from pathlib import Path
import shutil
from pprint import pprint
from dataclasses import asdict
from tqdm import tqdm
import pandas as pd
BASE_DIR = os.getcwd()  # if project loaded in root view
DIR_CLI = join(BASE_DIR, 'vframe_cli')
sys.path.append(DIR_CLI)
from vframe.settings import app_cfg
from vframe.utils.file_utils import ensure_dir, load_yaml, write_yaml
from vframe.utils.file_utils import write_txt, replace_ext, chmod_exec
from vframe.utils.dataset_utils import split_train_val_test
from vframe.models.annotation import Annotation
from vframe.models.training_dataset import YoloPyTorch

# %% codecell
log = app_cfg.LOG

# %% codecell
# load yaml
opt_fp_cfg = '/work/vframe/data/configs/yolo_pytorch/example.yaml'
cfg = load_yaml(opt_fp_cfg, data_class=YoloPyTorch)

# %% codecell
# provision output
ensure_dir(cfg.fp_output)
dir_images = join(cfg.fp_output, cfg.fn_images)
dir_labels = join(cfg.fp_output, cfg.fn_labels)
ensure_dir(dir_images)
ensure_dir(dir_labels)

# %% codecell
# write to yaml
fp_out = join(cfg.fp_output, cfg.fn_hyp)
comment = '\n'.join([app_cfg.LICENSE_HEADER,'# Hyperparameter'])
write_yaml(asdict(cfg.hyperparameters), fp_out, comment=comment)

# %% codecell
# load annos
df = pd.read_csv(cfg.fp_annotations)
df_pos = df[df.label_enum != app_cfg.LABEL_BACKGROUND]
df_neg = df[df.label_enum == app_cfg.LABEL_BACKGROUND]

# %% codecell
# count
print(f'positive annotations: {len(df_pos):,}')
print(f'background annotations: {len(df_neg):,}')
print(f'total annotations: {len(df):,}')
print(f'positive images: {len(df_pos.groupby("filename")):,}')
print(f'negative images: {len(df_neg.groupby("filename")):,}')
print(f'total images: {len(df.groupby("filename")):,}')

# %% codecell
# get class-label list sorted by class index
df_sorted = df_pos.sort_values(by='label_index', ascending=True)
df_sorted.drop_duplicates(['label_enum'], keep='first', inplace=True)
class_labels = df_sorted.label_enum.values.tolist()
# write to txt
write_txt(class_labels, join(cfg.fp_output, app_cfg.FN_LABELS))

# %% codecell
# update config
cfg.classes = class_labels

# %% codecell
# Generate one label per file with all bboxes and classes
# <object-class> <x_center> <y_center> <width> <height>
labels_data = {}
file_list = []
df_groups = df_pos.groupby('filename')
for fn, df_group in df_groups:
  annos = []
  file_list.append(join(dir_images, fn))
  for row_idx, row in df_group.iterrows():
    annos.append(Annotation.from_anno_series_row(row).to_darknet_str())
  labels_data.update({fn: annos})

# %% codecell
# write txt files for train, val
splits = split_train_val_test(file_list, splits=(0.6, 0.2, 0.2), seed=1)
write_txt(splits['train'], join(cfg.fp_output, cfg.fn_train))
write_txt(splits['val'], join(cfg.fp_output, cfg.fn_val))
write_txt(splits['test'], join(cfg.fp_output, cfg.fn_test))

# %% codecell
# write metadata
fp_out = join(cfg.fp_output, cfg.fn_metadata)
comment = '\n'.join([app_cfg.LICENSE_HEADER, '# Metadata'])
write_yaml(cfg.to_metadata(), fp_out, comment=comment)

# %% codecell
# copy postive images
for fn, annos in tqdm(labels_data.items()):
  # write all annos for this image to txt file
  fp_label = join(dir_labels, replace_ext(fn, 'txt'))
  write_txt(annos, fp_label)

# %% codecell
# symlink/copy images
df_groups = df.groupby('filename')
for fn, df_group in tqdm(df_groups):
  fpp_im_dst = Path(join(dir_images, fn))
  fpp_im_src = Path(join(cfg.fp_images, fn))
  if cfg.symlink:
    if fpp_im_dst.is_symlink():
      fpp_im_dst.unlink()
    fpp_im_dst.symlink_to(fpp_im_src)
  else:
    shutil.copy(fpp_im_src, fpp_im_dst)

# %% codecell
# write model yaml, but print k:v pairs instead of dump
model_cfg = load_yaml(cfg.fp_model_cfg)
fp_out = join(cfg.fp_output, cfg.fn_model_cfg)
model_cfg['nc'] = len(cfg.classes)
with open(fp_out, 'w') as f:
  for k,v in model_cfg.items():
   f.write(f'{k}: {v}\n')

# %% codecell
# shell scripts
args = cfg.arguments
sh = []
sh.extend(['python','train.py'])
sh.extend(['--batch', str(args.batch_size)])
sh.extend(['--weights', args.weights])
sh.extend(['--cfg', join(cfg.fp_output, cfg.fn_model_cfg)])
sh.extend(['--data', join(cfg.fp_output, cfg.fn_metadata)])
sh.extend(['--hyp', join(cfg.fp_output, cfg.fn_hyp)])
sh.extend(['--epochs', str(args.epochs)])
sh.extend(['--batch-size', str(args.batch_size)])
sh.extend(['--img-size', str(args.img_size_train)])
if args.rect:
  sh.extend(['--rect', args.rect])
if args.resume:
  sh.extend(['--resume', args.resume])
if args.no_save:
  sh.extend(['--nosave'])
if args.no_test:
  sh.extend(['--notest'])
if args.no_autoanchor:
  sh.extend(['--noautoanchor'])
if args.evolve:
  sh.extend(['--evolve'])
sh.extend(['--bucket', args.local_rank])
if args.cache_images:
  sh.extend(['--cache-images'])
if args.image_weights:
  sh.extend(['--image-weights'])
sh.extend(['--name', args.name])
if args.device:
  sh.extend(['--device', str(args.device)])
if args.multi_scale:
  sh.extend(['--multi-scale'])
if args.adam:
  sh.extend(['--adam'])
if args.single_cls:
  sh.extend(['--single-cls'])
if args.sync_bn:
  sh.extend(['--sync-bn'])
sh.extend(['--local_rank', args.local_rank])
sh.extend(['--logdir', args.logdir])
sh.extend(['--log-imgs', args.log_imgs])
sh.extend(['--workers', args.workers])
# join strings
sh_str = '\n'.join(['#!/bin/bash','','# training', ''])
sh = list(map(str, sh))
sh_str += ' '.join(sh)
fp_sh = join(cfg.fp_output, app_cfg.FN_TRAIN_INIT)
# write
write_txt(sh_str, fp_sh)
# make executable
chmod_exec(fp_sh)
