# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Self-attention block architecture

# Important!!!! The self-attention is not used in the paper because we cannot
# get good results out from it. Hence, the following code is not guaranteed to
# be correct. Users interested in this could try and see how the attention
# mechanisms make the differences, and if you identify any bugs in the following
# code, please let us know.

import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
  def __init__(self, in_channels):
    super(SelfAttention, self).__init__()

    self.conv_theta = nn.Conv2d(in_channels, in_channels // 8,
                                kernel_size=1, stride=1, padding=0)
    self.conv_phi = nn.Conv2d(in_channels, in_channels // 8,
                              kernel_size=1, stride=1, padding=0)
    self.conv_g = nn.Conv2d(in_channels, in_channels // 2,
                            kernel_size=1, stride=1, padding=0)
    self.conv_o = nn.Conv2d(in_channels // 2, in_channels,
                            kernel_size=1, stride=1, padding=0)

    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.softmax = nn.Softmax(dim=-1)

    self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    batch_size, c, h, w = x.size()

    theta = self.conv_theta(x).view(batch_size, -1, h * w).permute(0, 2, 1)
    phi = self.maxpool(self.conv_phi(x)).view(batch_size, -1, h * w // 4)

    attn = torch.bmm(theta, phi)
    attn = self.softmax(attn)

    g = self.maxpool(self.conv_g(x)).view(batch_size, -1, h * w // 4)

    o = torch.bmm(g, attn.permute(0, 2, 1))
    o = o.view(batch_size, -1, h, w)
    o = self.conv_o(o)

    o = self.gamma * o + x

    return o
