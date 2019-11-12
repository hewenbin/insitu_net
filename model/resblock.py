# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Residual block architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlockGenerator(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu, upsample=True):
    super(BasicBlockGenerator, self).__init__()

    self.activation = activation
    self.upsample = upsample
    self.conv_res = None
    if self.upsample or in_channels != out_channels:
      self.conv_res = nn.Conv2d(in_channels, out_channels,
                                1, 1, 0, bias=False)

    self.bn0 = nn.BatchNorm2d(in_channels)
    self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

  def forward(self, x):
    residual = x
    if self.upsample:
      residual = F.interpolate(residual, scale_factor=2)
    if self.conv_res is not None:
      residual = self.conv_res(residual)

    out = self.bn0(x)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, scale_factor=2)
    out = self.conv0(out)

    out = self.bn1(out)
    out = self.activation(out)
    out = self.conv1(out)

    return out + residual

class BasicBlockDiscriminator(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu, downsample=True):
    super(BasicBlockDiscriminator, self).__init__()

    self.activation = activation
    self.downsample = downsample
    self.conv_res = None
    if self.downsample or in_channels != out_channels:
      self.conv_res = nn.Conv2d(in_channels, out_channels,
                                1, 1, 0, bias=True)

    self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=True)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=True)

  def forward(self, x):
    residual = x
    if self.conv_res is not None:
      residual = self.conv_res(residual)
    if self.downsample:
      residual = F.avg_pool2d(residual, kernel_size=2)

    out = self.activation(x)
    out = self.conv0(out)

    out = self.activation(out)
    out = self.conv1(out)

    if self.downsample:
      out = F.avg_pool2d(out, kernel_size=2)

    return out + residual

class FirstBlockDiscriminator(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu):
    super(FirstBlockDiscriminator, self).__init__()

    self.activation = activation

    self.conv_res = nn.Conv2d(in_channels, out_channels,
                              1, 1, 0, bias=True)
    self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=True)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=True)

  def forward(self, x):
    residual = self.conv_res(x)
    residual = F.avg_pool2d(residual, kernel_size=2)

    out = self.conv0(x)
    out = self.activation(out)
    out = self.conv1(out)

    out = F.avg_pool2d(out, kernel_size=2)

    return out + residual
