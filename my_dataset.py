
import polars
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
import torchvision
import pandas as pd
import random
import numpy

# ensure you have torch 1.12.0+cu113 and not torch 1.12.0+cpu since read_image fails otherwise
print(torch.__version__)

# Returns triplet images from Totally-Looks-Like-Data transformed by `transform` param
class MyTLLDataset(Dataset):
    def __init__(self, ttl_dataset_path, transform, no_faces = True):
        self.transform = transform

        # (3, 245, 200)
        self.metadata = pd.read_pickle(os.path.join(ttl_dataset_path, "metadata.pkl"))
        # the images in left/ dir have the same names as in right/ dir
        if no_faces:
            self.left_imgs_paths = []
            self.right_imgs_paths = []
            for i, img_name in enumerate(os.listdir(os.path.join(ttl_dataset_path, "left"))):
                if self.metadata['no_faces'][i]:
                    self.left_imgs_paths.append(os.path.join(ttl_dataset_path, "left", img_name))
                    self.right_imgs_paths.append(os.path.join(ttl_dataset_path, "right", img_name))
        else:
            self.left_imgs_paths = [os.path.join(ttl_dataset_path, "left", img_name)
                for img_name in os.listdir(os.path.join(ttl_dataset_path, "left"))]
            self.right_imgs_paths = [os.path.join(ttl_dataset_path, "right", img_name)
                for img_name in os.listdir(os.path.join(ttl_dataset_path, "right"))]
      
    def __len__(self):
        return len(self.left_imgs_paths)

    def __getitem__(self, idx):
        def read_transform_img(img_path):
            img = torchvision.io.read_image(img_path)
            img = self.transform(img)

            return img

        left_img = read_transform_img(self.left_imgs_paths[idx])
        right_img = read_transform_img(self.right_imgs_paths[idx])

        rand_idx = numpy.random.randint(len(self))
        while rand_idx == idx:
            rand_idx = numpy.random.randint(len(self))
        rand_img_path = self.left_imgs_paths[idx] if random.choice([True, False]) else self.right_imgs_paths[idx]
        rand_img = read_transform_img(rand_img_path)

        # return 2 similar and 1 dissimilar for triplet loss
        return left_img, right_img, rand_img
