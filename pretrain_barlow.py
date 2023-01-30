import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import tqdm

from my_dataset import MyTLLDataset
from my_models import SiameseNetTriplet
from arg_parser import build_arg_parser

# Computes the DxD cross-correlation matrix between the normalized vectors of the 2 views of each image.
# Then it splits this cross-correlation matrix into two parts. The first part, the diagonal of this matrix
#  is brought closer to 1, which pushes up the cosine similarity between the latent vectors of two views
#  of each image, thus making the backbone invariant to the transformations applied to the views.
#  The second part of the loss pushes the non-diagonal elements of the cross-corrlelation matrix closes to 0.
#  This reduces the redundancy between the different dimensions of the latent vector.
class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()

        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        batch_size = z1.size(0)
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

# we use pipeline proposed in SimCLR, which generates two copies/views
#  of an input image by applying the following transformations in a sequence.
class BarlowTwinsTransform:
    def __init__(self, normalize, input_height=224, gaussian_blur=True, jitter_strength=1.0,):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, sample):
        tf1 = self.transform(sample)
        tf2 = self.transform(sample)
        return tf1, tf2

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=128):
        super().__init__()

        self.output_dim = output_dim

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)

class BarlowTwinsModule(nn.Module):
    def __init__(self):
        super(BarlowTwinsModule, self).__init__()

        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.out_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity() # remove last layer
        self.projection_head = ProjectionHead(input_dim=self.out_ftrs)

    def forward(self, x1, x2):
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return z1, z2

