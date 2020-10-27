import sys
import os
from os.path import join
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import operator
import re
import math
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import dacite
import numpy as np

from vframe.utils import file_utils, log_utils
from vframe.models.geometry import BBox
from vframe.models.annotation import Annotation
from vframe.settings import app_cfg


class YoloProject:

  bboxes = []

  log = log_utils.Logger.getLogger()

  @classmethod
  def from_cfg(cls, cfg):

    # load annotations and convert to BBox class
    df = pd.read_csv(cfg.annotations)

    # class indices must start from zero
    df.label_index -= df.label_index.min()
    n_bg_annos = len(df[df.label_enum == 'background'])
    if n_bg_annos > 0:
      cls.log.debug(f'Annotations contain {n_bg_annos} negative images. Removing 0th index')
      # subtract the 0th index if Background (negative data is used)
      df.label_index -= 1  
    
    # ensure output directory
    file_utils.ensure_dir(cfg.output)

    # .sh train script
    fp_sh_train = join(cfg.output, app_cfg.FN_TRAIN_INIT)

    # .sh testing script
    fp_sh_test = join(cfg.output, app_cfg.FN_TEST_INIT)

    # .sh resume [multi] GPU script
    fp_sh_resume = join(cfg.output, app_cfg.FN_TRAIN_RESUME)

    # .data file
    fp_metadata = join(cfg.output, app_cfg.FN_META_DATA)

    # .cfg file
    fp_cfg_train = join(cfg.output, 'yolov4.cfg')
    fp_cfg_deploy = join(cfg.output, 'yolov4_deploy.cfg')

    # images/
    dir_images = join(cfg.output, cfg.images_labels)
    dir_labels = dir_images
    file_utils.ensure_dir(dir_images)

    # labels/

    # .data and deps filepaths
    fp_classes = join(cfg.output, app_cfg.FN_CLASSES)
    fp_valid_list = join(cfg.output, app_cfg.FN_VALID)
    fp_train_list = join(cfg.output, app_cfg.FN_TRAIN)
    dir_backup = join(cfg.output, app_cfg.DN_BACKUP)
    file_utils.ensure_dir(dir_backup)

    # create dict of classes, then sort by index, 0 - N
    class_labels = {}
    for idx, record in df.iterrows():
      if record.label_enum == app_cfg.LABEL_BACKGROUND:
        continue
      if record.label_enum not in class_labels.keys():
        class_labels.update({record.label_enum: record.label_index})
    class_labels = sorted(class_labels.items(), key=operator.itemgetter(1))
    class_labels = [x[0] for x in class_labels]

    # Create training ".cfg"
    num_classes = len(class_labels)  # class in annotation file
    num_masks = 3  # assuming 3
    num_filters = (num_classes + 5) * num_masks
    
    """
    change line max_batches to (classes*2000 but not less than 
    number of training images, but not less than number of training images and not less than 6000), 
    f.e. max_batches=6000 if you train for 3 classes
    """
    max_batches = max(6000, max(len(df), 2000 * num_classes))
    max_batches = min(cfg.batch_ceiling, max_batches)
    # change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
    batch_steps = (int(0.8 * max_batches), int(0.9 * max_batches))

    # Generate meta.data file
    data = []
    data.append(f'classes = {num_classes}')
    data.append(f'train = {fp_train_list}')
    data.append(f'valid = {fp_valid_list}')
    data.append(f'names = {fp_classes}')
    data.append('backup = {}'.format(dir_backup))
    file_utils.write_txt(data, fp_metadata)

    # Create training .cfg
    subs_all = []
    
    # Sizes, classes
    subs_all.append(('{width}', str(cfg.image_size)))
    subs_all.append(('{height}', str(cfg.image_size)))
    subs_all.append(('{classes}', str(num_classes)))
    
    # Parameters
    subs_all.append(('{num_filters}', str(num_filters)))
    subs_all.append(('{max_batches}', str(max_batches)))
    subs_all.append(('{steps_min}', str(batch_steps[0])))
    subs_all.append(('{steps_max}', str(batch_steps[1])))
    subs_all.append(('{focal_loss}', f'{int(cfg.focal_loss)}'))
    subs_all.append(('{learning_rate}', f'{cfg.learning_rate}'))

    # Data augmentation
    subs_all.append(('{cutmix}', f'{int(cfg.cutmix)}'))
    subs_all.append(('{mosaic}', f'{int(cfg.mosaic)}'))
    subs_all.append(('{mixup}', f'{int(cfg.mixup)}'))
    subs_all.append(('{blur}', f'{int(cfg.blur)}'))
    
    # images per class
    groups = df.groupby('label_enum')
    ipc = ','.join([str(len(groups.get_group(label))) for label in class_labels])
    subs_all.append(('{counters_per_class}', ipc))

    # Test, train unique params
    
    # batch size train
    subs_train = subs_all.copy()
    subs_train.append(('{batch_size}', str(cfg.batch_size)))
    subs_train.append(('{subdivisions}', str(cfg.subdivisions)))

    # batch size test
    subs_test = subs_all.copy()
    subs_test.append(('{batch_size}', '1'))
    subs_test.append(('{subdivisions}', '1'))

    
    # load original cfg into str
    cfg_orig = '\n'.join(file_utils.load_txt(cfg.cfg))

    # search and replace train
    cfg_train = cfg_orig  # str copy
    for placeholder, value in subs_train:
      cfg_train = cfg_train.replace(placeholder, value)

    # search and replace test
    cfg_test = cfg_orig # str copy
    for placeholder, value in subs_test:
      cfg_test = cfg_test.replace(placeholder, value)

    # write .cfg files
    file_utils.write_txt(cfg_train, fp_cfg_train)
    file_utils.write_txt(cfg_test, fp_cfg_deploy)
    
    # write train .sh
    sh_base = []
    sh_base.append('#!/bin/bash')
    sh_base.append(f'DARKNET={cfg.darknet}')
    sh_base.append(f'DIR_PROJECT={cfg.output}')
    sh_base.append(f'FP_META={fp_metadata}')

    sh_train = sh_base.copy()
    sh_train.append(f'FP_CFG={fp_cfg_train}')
    sh_train.append(f'FP_WEIGHTS={cfg.weights}')
    sh_train.append('CMD="detector train"')
    sh_train.append(f'GPUS="-gpus {cfg.gpu_idx_init}"')
    if not cfg.show_output:
      sh_train.append(f'VIZ="-dont_show"')  # don't show viz, if running in docker
    else:
      sh_train.append(f'VIZ=""')

    sh_train.append(f'$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS $VIZ 2>&1 | tee {cfg.logfile}')
    file_utils.write_txt(sh_train, fp_sh_train)
    file_utils.chmod_exec(fp_sh_train)

    # Generate resume .sh
    sh_resume = sh_base.copy()
    sh_resume.append(f'FP_CFG={fp_cfg_train}')
    cfg_name = Path(fp_cfg_train).stem
    sh_resume.append('# Edit path to weights')
    fp_weights_last = join(dir_backup, f'{cfg_name}_last.weights')
    sh_resume.append(f'FP_WEIGHTS={fp_weights_last}')
    sh_resume.append('CMD="detector train"')
    gpus_resume_str = ','.join(list(map(str, cfg.gpu_idxs_resume)))
    sh_resume.append(f'GPUS="-gpus {gpus_resume_str}"')
    if not cfg.show_output:
      sh_resume.append(f'VIZ="-dont_show"')  # don't show viz, if running in docker
    else:
      sh_resume.append(f'VIZ=""')

    sh_resume.append(f'$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS $VIZ 2>&1 | tee -a {cfg.logfile}')
    file_utils.write_txt(sh_resume, fp_sh_resume)
    file_utils.chmod_exec(fp_sh_resume)

    # Generate test .sh
    sh_test = sh_base.copy()
    sh_test.append(f'FP_CFG={fp_cfg_deploy}')
    cfg_name = Path(fp_cfg_train).stem
    sh_test.append('# Edit path to weights')
    fp_weights_best = join(dir_backup, f'{cfg_name}_best.weights')
    sh_test.append(f'FP_WEIGHTS={fp_weights_best}')
    sh_test.append('CMD="detector test"')
    sh_test.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $1')
    file_utils.write_txt(sh_test, fp_sh_test)
    file_utils.chmod_exec(fp_sh_test)

    # Generate classes.txt
    file_utils.write_txt(class_labels, fp_classes)

    # Generate the labels data
    # one label per file with all bboxes and classes
    # <object-class> <x_center> <y_center> <width> <height>
    # color,filename,label,label_index,x1,x2,y1,y2
    # 0xcbff00,emitter_0000_cam_0000_0002.png,ao25rt_arming_vanes,2,0.521875,0.6,0.7055555555555556,0.827
    labels_data = {}
    file_list = []
    df_im_groups = df.groupby('filename')
    for fn, df_im_group in df_im_groups:
      darknet_annos = []
      file_list.append(join(dir_images, fn))
      for row_idx, row in df_im_group.iterrows():
        #bbox = dacite.from_dict(data_class=BBoxNormLabelColor, data=row.to_dict())
        anno = Annotation.from_anno_series_row(row)
        #bbox = dacite.from_dict(data_class=BBoxNormLabel, data=row.to_dict())

        #if bbox.label_index == 0 and bbox.w == 0 and bbox.h == 0:
        # cls.log.debug(f'{bbox.label}, {bbox.label_index}')
        #if anno.label_enum == 'background' and int(anno.label_index) == -1:
        if anno.label_enum == 'background':
          #cls.log.debug(f'Negative data: {bbox.filename}')
          darknet_anno = ''  #empty entry for negative data
        else:
          darknet_anno = anno.to_darknet_str()
        darknet_annos.append(darknet_anno)
      labels_data.update({fn: darknet_annos})
    
    # write labels and symlink images
    for fn, darknet_annos in tqdm(labels_data.items()):
      fp_label = join(dir_labels, file_utils.replace_ext(fn, 'txt'))
      file_utils.write_txt(darknet_annos, fp_label)
      fpp_im_dst = Path(join(dir_images, fn))
      fpp_im_src = Path(join(cfg.images, fn))
      if cfg.use_symlinks:
          if fpp_im_dst.is_symlink():
            fpp_im_dst.unlink()
          fpp_im_dst.symlink_to(fpp_im_src)
      else:
        shutil.copy(fpp_im_src, fpp_im_dst)


    
    # Generate training list of images
    random.shuffle(file_list)
    n_train = int(0.8 * len(file_list))
    n_test = len(file_list) - n_train
    training_list = file_list[:n_train]
    validation_list = file_list[n_train:]
    file_utils.write_txt(training_list, fp_train_list)
    file_utils.write_txt(validation_list, fp_valid_list)




  # ------------------------------------------
  # Plot YOLO Loss Data
  # ------------------------------------------

  @classmethod
  def create_plot(cls, fp_log, xmin_max=None, ymin_max=None):
    """Create plot of Yolo Loss Avg"""
    # can stop training when it reaches ~0.067
    # TODO make dynamic plot

    #f = input']
    #lines  = [line.rstrip("\n") for line in f.readlines()]
    plt.style.use('ggplot')

    lines = file_utils.load_txt(fp_log)
    print(len(lines))
    numbers = {'1','2','3','4','5','6','7','8','9'}

    iters = []
    losses = []
    
    fig,ax = plt.subplots()
    
    for line in lines:
      result = re.match(r'^[\s]?([0-9]+):', line)
      if result:
        n_iter = int(result[1])
        loss_match = re.search(r'([0-9.]+)\s\bavg loss\b', line)  # 0.095066 avg loss
        if loss_match:
          loss = float(loss_match[1])
          iters.append(n_iter)
          losses.append(loss)
             
    ax.plot(iters,losses)
    cls.log.debug(f'xmin_max: {xmin_max}, ymin_max: {ymin_max}')
    cls.log.debug(f'min/max loss: {min(losses)}, {max(losses)}')
    if xmin_max and ymin_max:
      ax.set_xlim(xmin_max)
      ax.set_ylim(ymin_max)
    project_name = Path(fp_log).parent.name
    plt.title('YoloV3: {}'.format(project_name))
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.grid()
    
    plt.show()

