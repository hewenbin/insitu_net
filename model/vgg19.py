# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# VGG19 network for perceptual loss computation

import torch.nn as nn
from torchvision import models

class VGG19(nn.Module):
  def __init__(self, layer="relu1_2"):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features

    self.layer_dict = {"relu1_1": 2, "relu1_2": 4,
                       "relu2_1": 7, "relu2_2": 9,
                       "relu3_1": 12, "relu3_2": 14}

    self.layer = layer
    self.subnet = nn.Sequential()
    for i in range(self.layer_dict[self.layer]):
      self.subnet.add_module(str(i), features[i])

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    out = self.subnet(x)
    return out
