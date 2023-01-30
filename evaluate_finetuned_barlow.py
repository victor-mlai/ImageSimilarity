import polars
import os
import torch
from torch.utils.data import Dataset
from torch import nn
import torchvision
import pandas as pd
import random
import numpy
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

from my_dataset import MyAllImagesTLLDataset
from my_models import SiameseNetTriplet
from arg_parser import build_arg_parser

args = build_arg_parser().parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset = MyAllImagesTLLDataset(transform)

def get_model():
    feat_extractor = models.resnet50()
    features = feat_extractor.fc.in_features
    feat_extractor.fc = torch.nn.Identity()

    model = SiameseNetTriplet(feat_extractor, features, args)
    
    checkpoint = torch.load(args.checkpoint_dir / args.model_filename)
    model.load_state_dict(checkpoint)
    model = model.eval()

    return model

model = get_model().to(device)

print("Calculating embeddings")
with torch.no_grad():
    left_embeddings = [
        model._forward(transform(torchvision.io.read_image(path)).unsqueeze(0).to(device)).squeeze().cpu()
        for path in dataset.left_imgs_paths
    ]
    right_embeddings = [
        model._forward(transform(torchvision.io.read_image(path)).unsqueeze(0).to(device)).squeeze().cpu()
        for path in dataset.right_imgs_paths
    ]
print("Done")
print("Finding matches...")

topk = 5
distrib = [0 for _ in range(topk)]

correct = 0
for left_idx, left_emb in enumerate(left_embeddings):
    pred_sim = []
    for right_emb in right_embeddings:
        sim = nn.functional.pairwise_distance(left_emb, right_emb)
        pred_sim.append(sim.item())

    pred_probs_idxs = np.array(pred_sim).argsort()[:topk]
    for i, idx in enumerate(pred_probs_idxs):
        if idx == left_idx:
            correct += 1
            distrib[i] += 1
            print(f"Found the match for img {idx} Placed at pos {i}")
            break

    if False:
        plt.figure(figsize=((topk+2) * 10, 10))
        plt.subplot(1, topk+2, 1)
        plt.imshow(np.transpose(inv_norm_transform(img_ref.squeeze()).cpu().numpy(), (1, 2, 0)))
        plt.subplot(1, topk+2, 2)
        plt.imshow(np.transpose(inv_norm_transform(img_target).numpy(), (1, 2, 0)))
        #plt.xticks([])
        #plt.yticks([])
        for k_idx, k in enumerate(pred_probs_args):
            plt.subplot(1, topk+2, k_idx+3)
            img = inv_norm_transform(transform(torchvision.io.read_image(dataset.right_imgs_paths[k])))
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            #plt.xticks([])
            #plt.yticks([])

        plt.show()

print(f"Found {correct}/{len(dataset)} in the top-{topk}")
plt.bar(range(topk), distrib)
plt.title(f'Rank frequency for Top-{topk} matches')
plt.ylabel('Frequency')
plt.xlabel('Rank')
plt.legend()
plt.show()
