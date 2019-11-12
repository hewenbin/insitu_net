# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator
from self_attention import SelfAttention

class Generator(nn.Module):
  def __init__(self, dsp=1, dvo=3, dvp=3, dspe=512, dvoe=512, dvpe=512, ch=64):
    super(Generator, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.dvo, self.dvoe = dvo, dvoe
    self.dvp, self.dvpe = dvp, dvpe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # visualization operations subnet
    self.vops_subnet = nn.Sequential(
      nn.Linear(dvo, dvoe), nn.ReLU(),
      nn.Linear(dvoe, dvoe), nn.ReLU()
    )

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe + dvoe + dvpe, ch * 16 * 4 * 4, bias=False)
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(ch),
      nn.ReLU(),
      nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
      nn.Tanh()
    )

  def forward(self, sp, vo, vp):
    sp = self.sparams_subnet(sp)
    vo = self.vops_subnet(vo)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp, vo, vp), 1)
    mp = self.mparams_subnet(mp)

    x = mp.view(mp.size(0), self.ch * 16, 4, 4)
    x = self.img_subnet(x)

    return x
