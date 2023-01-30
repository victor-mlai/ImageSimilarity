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
from my_models import SiameseNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
inv_norm_transform = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

dataset = MyAllImagesTLLDataset(transform)

def get_model():
    feat_extractor = models.resnet18(pretrained=False)
    features = feat_extractor.fc.in_features
    feat_extractor = torch.nn.Sequential(*list(feat_extractor.children())[:-1])

    model = SiameseNet(feat_extractor, features)
    
    checkpoint = torch.load("siamese_network.pt")
    model.load_state_dict(checkpoint)
    model = model.eval()

    return model

model = get_model().to(device)

print("Calculating embeddings")
with torch.no_grad():
    left_embeddings = [model.feature_extractor.forward(transform(torchvision.io.read_image(path)).unsqueeze(0).to(device)).cpu() for path in dataset.left_imgs_paths]
    right_embeddings = [model.feature_extractor.forward(transform(torchvision.io.read_image(path)).unsqueeze(0).to(device)).cpu() for path in dataset.right_imgs_paths]
print("Done")
print("Finding matches...")


correct = 0
for left_idx, left_emb in enumerate(left_embeddings):
    #img_ref, img_target = dataset[i]
    #img_ref = img_ref.unsqueeze(0).to(device)

    pred_probs = []
    for right_emb in right_embeddings:
        left_emb = left_emb.view(left_emb.size()[0], -1).to(device)
        right_emb = right_emb.view(right_emb.size()[0], -1).to(device)
        emb = torch.cat((left_emb, right_emb), 1)
        with torch.no_grad():
            output = model.adaptive_func(emb)
        pred_prob = nn.functional.softmax(output, dim=1).squeeze()

        pred_probs.append(pred_prob[0].item())

    topk = 25
    pred_probs_args = np.array(pred_probs).argsort()[:topk]
    for arg, idx in enumerate(pred_probs_args):
        if idx == left_idx:
            correct += 1
            print(f"Found the match for img {idx} Placed at pos {arg}")
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