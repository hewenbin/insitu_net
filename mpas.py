# Copyright 2019 The IEVA-DGM Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# mpas dataset

from __future__ import absolute_import, division, print_function

import os

import pandas as pd
import numpy as np
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MPASDataset(Dataset):
  def __init__(self, root, train=True, data_len=0, transform=None):
    self.root = root
    self.train = train
    self.data_len = data_len
    self.transform = transform
    if self.train:
      self.filenames = pd.read_csv(os.path.join(root, "train/filenames.txt"),
                                   sep=" ", header=None)
      self.params = np.load(os.path.join(root, "train/params.npy"))
    else:
      self.filenames = pd.read_csv(os.path.join(root, "test/filenames.txt"),
                                   sep=" ", header=None)
      self.params = np.load(os.path.join(root, "test/params.npy"))

  # TODO(wenbin): deal with data_len correctly.
  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.params)

  def __getitem__(self, index):
    if type(index) == torch.Tensor:
      index = index.item()

    params = self.params[index]
    sparams = np.copy(params[1:2])
    vops = np.copy(params[2:5])

    vparams = np.zeros(3, dtype=np.float32)
    vparams[0] = np.cos(np.deg2rad(params[5]))
    vparams[1] = np.sin(np.deg2rad(params[5]))
    vparams[2] = params[6] / 90.

    if self.train:
      img_name = os.path.join(self.root, "train/" + self.filenames.iloc[index][0])
    else:
      img_name = os.path.join(self.root, "test/" + self.filenames.iloc[index][0])

    image = io.imread(img_name)[:, :, 0:3]

    sample = {"image": image, "sparams": sparams, "vops": vops, "vparams": vparams}

    if self.transform:
      sample = self.transform(sample)

    return sample

# utility functions
def imshow(image):
  plt.imshow(image.numpy().transpose((1, 2, 0)))

# data transformation
class Resize(object):
  def __init__(self, size):
    assert isinstance(size, (int, tuple))
    self.size = size

  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vops = sample["vops"]
    vparams = sample["vparams"]

    h, w = image.shape[:2]
    if isinstance(self.size, int):
      if h > w:
        new_h, new_w = self.size * h / w, self.size
      else:
        new_h, new_w = self.size, self.size * w / h
    else:
      new_h, new_w = self.size

    new_h, new_w = int(new_h), int(new_w)

    image = transform.resize(
        image, (new_h, new_w), order=1, mode="reflect",
        preserve_range=True, anti_aliasing=True).astype(np.float32)

    return {"image": image, "sparams": sparams, "vops": vops, "vparams": vparams}

class Normalize(object):
  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vops = sample["vops"]
    vparams = sample["vparams"]

    image = (image.astype(np.float32) - 127.5) / 127.5

    # sparams min [1.]
    #         max [4.]

    sparams = (sparams - np.array([2.5], dtype=np.float32)) / \
              np.array([1.5], dtype=np.float32)

    return {"image": image, "sparams": sparams, "vops": vops, "vparams": vparams}

class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vops = sample["vops"]
    vparams = sample["vparams"]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {"image": torch.from_numpy(image),
            "sparams": torch.from_numpy(sparams),
            "vops": torch.from_numpy(vops),
            "vparams": torch.from_numpy(vparams)}

# # data verification
# import matplotlib.pyplot as plt

# dataset = MPASDataset(
#     root="/Users/rhythm/Desktop/mpas",
#     train=False,
#     transform=transforms.Compose([Resize(64), Normalize(), ToTensor()]))

# loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# samples = iter(loader).next()

# print(samples)

# # fig = plt.figure()
# # imshow(utils.make_grid(((samples["image"] + 1.) * .5)))
# # plt.show()
