#!/bin/bash
# source environment variables for example scripts

# base
DIR_EXAMPLES="../data/examples"
DIR_IMAGES=${DIR_EXAMPLES}"/images"
DIR_IMAGES_OUT=${DIR_IMAGES}"/output"
DIR_IMAGES_3D=${DIR_IMAGES}"/3d"
DIR_VIDEOS=${DIR_EXAMPLES}"/videos"

# face images
FP_SNOWDEN_X1=${DIR_IMAGES_3D}"/face-snowden-x1.png"
FP_SNOWDEN_X3=${DIR_IMAGES_3D}"/face-snowden-x3.png"

# face videos
FP_SNOWDEN_X1_VIDEO=${DIR_IMAGES_3D}"/face-snowden-x1.mp4"

echo "Added VFRAME example filepaths to environment"