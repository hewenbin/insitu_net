# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# main file for training

from __future__ import absolute_import, division, print_function

import os
import argparse
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import sys
sys.path.append("../datasets")
from mpas import *

from generator import Generator
from discriminator import Discriminator
from vgg19 import VGG19

# parse arguments
def parse_args():
  parser = argparse.ArgumentParser(description="InSituNet")

  parser.add_argument("--no-cuda", action="store_true", default=False,
                      help="disables CUDA training")
  parser.add_argument("--data-parallel", action="store_true", default=False,
                      help="enable data parallelism")
  parser.add_argument("--seed", type=int, default=1,
                      help="random seed (default: 1)")

  parser.add_argument("--root", required=True, type=str,
                      help="root of the dataset")
  parser.add_argument("--resume", type=str, default="",
                      help="path to the latest checkpoint (default: none)")

  parser.add_argument("--dsp", type=int, default=3,
                      help="dimensions of the simulation parameters (default: 3)")
  parser.add_argument("--dvo", type=int, default=3,
                      help="dimensions of the visualization operations (default: 3)")
  parser.add_argument("--dvp", type=int, default=3,
                      help="dimensions of the view parameters (default: 3)")
  parser.add_argument("--dspe", type=int, default=512,
                      help="dimensions of the simulation parameters' encode (default: 512)")
  parser.add_argument("--dvoe", type=int, default=512,
                      help="dimensions of the visualization operations' encode (default: 512)")
  parser.add_argument("--dvpe", type=int, default=512,
                      help="dimensions of the view parameters' encode (default: 512)")
  parser.add_argument("--ch", type=int, default=64,
                      help="channel multiplier (default: 64)")

  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")

  parser.add_argument("--mse-loss", action="store_true", default=False,
                      help="enable mse loss")
  parser.add_argument("--perc-loss", type=str, default="relu1_2",
                      help="layer that perceptual loss is computed on (default: relu1_2)")
  parser.add_argument("--gan-loss", type=str, default="none",
                      help="gan loss (default: none)")
  parser.add_argument("--gan-loss-weight", type=float, default=0.,
                      help="weight of the gan loss (default: 0.)")

  parser.add_argument("--lr", type=float, default=1e-3,
                      help="learning rate (default: 1e-3)")
  parser.add_argument("--d-lr", type=float, default=1e-3,
                      help="learning rate of the discriminator (default: 1e-3)")
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 of Adam (default: 0.9)")
  parser.add_argument("--beta2", type=float, default=0.999,
                      help="beta2 of Adam (default: 0.999)")
  parser.add_argument("--batch-size", type=int, default=50,
                      help="batch size for training (default: 50)")
  parser.add_argument("--start-epoch", type=int, default=0,
                      help="start epoch number (default: 0)")
  parser.add_argument("--epochs", type=int, default=10,
                      help="number of epochs to train (default: 10)")

  parser.add_argument("--log-every", type=int, default=10,
                      help="log training status every given number of batches (default: 10)")
  parser.add_argument("--check-every", type=int, default=20,
                      help="save checkpoint every given number of epochs (default: 20)")

  return parser.parse_args()

# the main function
def main(args):
  # log hyperparameters
  print(args)

  # select device
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda:0" if args.cuda else "cpu")

  # set random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # data loader
  train_dataset = MPASDataset(
      root=args.root,
      train=True,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  test_dataset = MPASDataset(
      root=args.root,
      train=False,
      data_len=1000,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=True, **kwargs)

  # model
  def weights_init(m):
    if isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m

  g_model = Generator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                      dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                      ch=args.ch)
  g_model.apply(weights_init)
  # if args.sn:
  #   g_model = add_sn(g_model)

  if args.data_parallel and torch.cuda.device_count() > 1:
    g_model = nn.DataParallel(g_model)
  g_model.to(device)

  if args.gan_loss != "none":
    d_model = Discriminator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                            dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                            ch=args.ch)
    d_model.apply(weights_init)
    if args.sn:
      d_model = add_sn(d_model)

    if args.data_parallel and torch.cuda.device_count() > 1:
      d_model = nn.DataParallel(d_model)
    d_model.to(device)

  # loss
  if args.perc_loss != "none":
    norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
    norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
    vgg = VGG19(args.perc_loss).eval()
    if args.data_parallel and torch.cuda.device_count() > 1:
      vgg = nn.DataParallel(vgg)
    vgg.to(device)

  mse_criterion = nn.MSELoss()
  train_losses, test_losses = [], []
  d_losses, g_losses = [], []

  # optimizer
  g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))
  if args.gan_loss != "none":
    d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                             betas=(args.beta1, args.beta2))

  # load checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint {}".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint["epoch"]
      g_model.load_state_dict(checkpoint["g_model_state_dict"])
      g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
      if args.gan_loss != "none":
        d_model.load_state_dict(checkpoint["d_model_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        d_losses = checkpoint["d_losses"]
        g_losses = checkpoint["g_losses"]
      train_losses = checkpoint["train_losses"]
      test_losses = checkpoint["test_losses"]
      print("=> loaded checkpoint {} (epoch {})"
          .format(args.resume, checkpoint["epoch"]))

  # main loop
  for epoch in tqdm(range(args.start_epoch, args.epochs)):
    # training...
    g_model.train()
    if args.gan_loss != "none":
      d_model.train()
    train_loss = 0.
    for i, sample in enumerate(train_loader):
      image = sample["image"].to(device)
      sparams = sample["sparams"].to(device)
      vops = sample["vops"].to(device)
      vparams = sample["vparams"].to(device)
      g_optimizer.zero_grad()
      fake_image = g_model(sparams, vops, vparams)

      loss = 0.

      # gan loss
      if args.gan_loss != "none":
        # update discriminator
        d_optimizer.zero_grad()
        decision = d_model(sparams, vops, vparams, image)

        if args.gan_loss == "vanilla":
          d_loss_real = torch.mean(F.softplus(-decision))
        elif args.gan_loss == "hinge":
          d_loss_real = torch.mean(F.relu(1. - decision))

        fake_decision = d_model(sparams, vops, vparams, fake_image.detach())

        if args.gan_loss == "vanilla":
          d_loss_fake = torch.mean(F.softplus(fake_decision))
        elif args.gan_loss == "hinge":
          d_loss_fake = torch.mean(F.relu(1. + fake_decision))

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()

        d_optimizer.step()

        # loss of generator
        g_optimizer.zero_grad()
        fake_decision = d_model(sparams, vops, vparams, fake_image)

        if args.gan_loss == "vanilla":
          g_loss = args.gan_loss_weight * torch.mean(F.softplus(-fake_decision))
        elif args.gan_loss == "hinge":
          g_loss = -args.gan_loss_weight * torch.mean(fake_decision)
        loss += g_loss

      # mse loss
      if args.mse_loss:
        mse_loss = mse_criterion(image, fake_image)
        loss += mse_loss

      # perceptual loss
      if args.perc_loss != "none":
        # normalize
        image = ((image + 1.) * .5 - norm_mean) / norm_std
        fake_image = ((fake_image + 1.) * .5 - norm_mean) / norm_std

        features = vgg(image)
        fake_features = vgg(fake_image)

        perc_loss = mse_criterion(features, fake_features)
        loss += perc_loss

      loss.backward()
      g_optimizer.step()
      train_loss += loss.item() * len(sparams)

      # log training status
      if i % args.log_every == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
          epoch, i * len(sparams), len(train_loader.dataset),
          100. * i / len(train_loader),
          loss.item()))
        if args.gan_loss != "none":
          print("DLoss: {:.6f}, GLoss: {:.6f}".format(
            d_loss.item(), g_loss.item()))
          d_losses.append(d_loss.item())
          g_losses.append(g_loss.item())
        train_losses.append(loss.item())

    print("====> Epoch: {} Average loss: {:.4f}".format(
      epoch, train_loss / len(train_loader.dataset)))

    # testing...
    g_model.eval()
    if args.gan_loss != "none":
      d_model.eval()
    test_loss = 0.
    with torch.no_grad():
      for i, sample in enumerate(test_loader):
        image = sample["image"].to(device)
        sparams = sample["sparams"].to(device)
        vops = sample["vops"].to(device)
        vparams = sample["vparams"].to(device)
        fake_image = g_model(sparams, vops, vparams)
        test_loss += mse_criterion(image, fake_image).item() * len(sparams)

        if i == 0:
          n = min(len(sparams), 8)
          comparison = torch.cat(
              [image[:n], fake_image.view(len(sparams), 3, 256, 256)[:n]])
          save_image(((comparison.cpu() + 1.) * .5),
                     "../tmp/" + str(args.mse_loss) + "_" + args.perc_loss + \
                     "_" + str(args.gan_loss) + "_" + str(epoch) + ".png", nrow=n)

    test_losses.append(test_loss / len(test_loader.dataset))
    print("====> Epoch: {} Test set loss: {:.4f}".format(
      epoch, test_losses[-1]))

    # saving...
    if epoch % args.check_every == 0:
      print("=> saving checkpoint at epoch {}".format(epoch))
      if args.gan_loss != "none":
        torch.save({"epoch": epoch + 1,
                    "g_model_state_dict": g_model.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_model_state_dict": d_model.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "d_losses": d_losses,
                    "g_losses": g_losses,
                    "train_losses": train_losses,
                    "test_losses": test_losses},
                   os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
                                args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                                str(epoch) + ".pth.tar"))
      else:
        torch.save({"epoch": epoch + 1,
                    "g_model_state_dict": g_model.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses},
                   os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
                                args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                                str(epoch) + ".pth.tar"))

      torch.save(g_model.state_dict(),
                 os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
                              args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                              str(epoch) + ".pth"))

if __name__ == "__main__":
  main(parse_args())
