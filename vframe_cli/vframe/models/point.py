############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

"""
VFRAME BBox classes. Used in all VFRAME repos
"""

import logging
import random
import math
from dataclasses import dataclass

import numpy as np

from vframe.models.color import Color


# ---------------------------------------------------------------------------
#
# Point classes
#
# ---------------------------------------------------------------------------

@dataclass
class PointNorm:
  x: float
  y: float
  dim: tuple(0.0, 1.0)

  def distance(self, p2):
    """Calculate distance between 2 points
    """
    dx = self.x - p2.x
    dy = self.y - p2.y
    return int(math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)))

  def _clamp(self, xy):
    return [min(self.dim[i], max(type(self.x)(0), xy[i])) for i in range(len(xy))]

  def move(self, x, y):
    return self.__class__(*self._clamp((self.x + x, self.y + y)))

  def scale(self, scale):
    return self.__class__((self.x * scale, self.y * scale))
  
  def to_point_dim(self, dim):
    return PointDim(int(self.x * dim[0]), int(self.y * dim[1]), dim)

  @property
  def xy(self):
    return (self.x, self.y)


@dataclass
class PointDim(PointNorm):
  x: int
  y: int
  dim: (int, int)

  def to_point_norm(self):
    return PointNorm(self.x / self.dim[0], self.y / self.dim[1])