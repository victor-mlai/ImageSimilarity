
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import torchvision.models as models
import torch
import torch.nn as nn

from my_dataset import MyTLLDataset


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
transform_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

dataset = MyTLLDataset(transform)
ref, pos, neg = dataset[39]
transf_ref = transform_normalize(ref)
transf_pos = transform_normalize(pos)

def get_feat_extr():
    #feat_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feat_extractor = models.resnet18(pretrained=True)

    # remove last fc layer which outputs class probabilities
    # and only keep the feature extractor
    features = feat_extractor.fc.in_features
    feat_extractor = torch.nn.Sequential(*list(feat_extractor.children())[:-1])
    feat_extractor = feat_extractor.eval()

    #we don't need gradients w.r.t. weights for a trained model
    for param in feat_extractor.parameters():
        param.requires_grad = False
    return feat_extractor

feat_extractor = get_feat_extr()


#we want to calculate gradient of higest score w.r.t. input
#so set requires_grad to True for input 
transf_ref.requires_grad = True
transf_pos.requires_grad = True

emb1 = feat_extractor.forward(transf_ref.unsqueeze(0))
emb2 = feat_extractor.forward(transf_pos.unsqueeze(0))

criterion = nn.MSELoss()
diff = criterion(emb2, emb1)
diff.backward()



#get max along channel axis
slc, _ = torch.max(torch.abs(transf_ref.grad), dim=0)
#normalize to [0..1]
slc = (slc - slc.min())/(slc.max()-slc.min())

#get max along channel axis
slc_pos, _ = torch.max(torch.abs(transf_pos.grad), dim=0)
#normalize to [0..1]
slc_pos = (slc_pos - slc_pos.min())/(slc_pos.max()-slc_pos.min())

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(np.transpose(ref.numpy(), (1, 2, 0)))
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(slc.numpy(), cmap=plt.cm.hot)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(np.transpose(pos.numpy(), (1, 2, 0)))
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(slc_pos.numpy(), cmap=plt.cm.hot)
plt.xticks([])
plt.yticks([])

plt.show()