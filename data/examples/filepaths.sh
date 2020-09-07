#!/bin/bash
# source environment variables for example scripts

# base
DIR_EXAMPLES="../data/examples"
DIR_IMAGES=${DIR_EXAMPLES}"/images"
DIR_IMAGES_OUT=${DIR_IMAGES}"/output"
DIR_VIDEOS=${DIR_EXAMPLES}"/videos"
DIR_VIDEO_OUT=${DIR_VIDEOS}"/output"

# face images
SNOWDEN_X1=${DIR_IMAGES}"/face-snowden-x1.png"
SNOWDEN_X3=${DIR_IMAGES}"/face-snowden-x3.png"

# face videos
SNOWDEN_X1_VIDEO=${DIR_VIDEOS}"/face-snowden-x1.mp4"

echo "Added VFRAME example filepaths to environment"