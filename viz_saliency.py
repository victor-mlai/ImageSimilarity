
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, Saliency, FeatureAblation
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import torchvision.models as models
import torchvision
import torch
import torch.nn as nn
import numpy as np
import os

from my_dataset import MyTLLDataset
from my_models import SiameseNetTriplet
from arg_parser import build_arg_parser

@torch.no_grad()
def search(model: SiameseNetTriplet, valid_ds: MyTLLDataset, device):
    correct_idxs_file = "Correct_idxs.npy"
    if os.path.isfile(correct_idxs_file):
        return np.load(correct_idxs_file).tolist()

    print("Calculating embeddings")
    left_embeddings = [
        model._forward(left.unsqueeze(0).to(device)).squeeze().cpu()
        for left, _, _ in valid_ds
    ]
    right_embeddings = [
        model._forward(right.unsqueeze(0).to(device)).squeeze().cpu()
        for _, right, _ in valid_ds
    ]
    print("Done")
    print("Finding matches...")

    correct_idxs = []
    correct = 0
    for left_idx, left_emb in enumerate(left_embeddings):
        pred_sim = [nn.functional.pairwise_distance(left_emb, right_emb).item() for right_emb in right_embeddings]
        closest_right_emb_idx = np.array(pred_sim).argmin()
        if closest_right_emb_idx == left_idx:
            correct += 1
            correct_idxs.append(left_idx)

    print(f"Found {correct}/{len(valid_ds)} in the top-{1}")


    np.save(correct_idxs_file, correct_idxs)
    print(f"Saved correct_idxs to {correct_idxs_file}")

    return correct_idxs

def get_model(args):
    encoder = models.resnet50()
    num_ftrs = encoder.fc.in_features
    encoder.fc = torch.nn.Identity()

    model = SiameseNetTriplet(encoder, num_ftrs, args)
    
    checkpoint = torch.load(args.checkpoint_dir / "SiameseNetTriplet.pch")
    model.load_state_dict(checkpoint)
    model = model.eval()

    return model


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    dataset = MyTLLDataset(args.dataset_dir, transform, no_faces=True)

    siam_model = get_model(args)

    matched_idxs = search(siam_model, dataset, device)

    #for idx in range(50, len(matched_idxs)):
    if True:
        idx = matched_idxs[95]
        ref, pos, neg = dataset[idx]
        transf_ref = transform_normalize(ref)
        transf_pos = transform_normalize(pos)
        transf_neg = transform_normalize(neg)

        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input 
        transf_ref.requires_grad = True
        transf_pos.requires_grad = True

        #out_ref = siam_model._forward(transf_ref.unsqueeze(0).to(device))
        #out_pos = siam_model._forward(transf_pos.unsqueeze(0).to(device))
        #out_neg = siam_model._forward(transf_pos.unsqueeze(0).to(device))

        def forward_func(ref, pos):
            sentence_vector_1 = siam_model._forward(ref.unsqueeze(0))
            sentence_vector_2 = siam_model._forward(pos.unsqueeze(0))
            return nn.functional.cosine_similarity(sentence_vector_1, sentence_vector_2)

        ig = FeatureAblation(forward_func)
        attr_ref : torch.Tensor = ig.attribute((transf_ref, transf_pos))

        inp_ref_image = torchvision.io.read_image(dataset.left_imgs_paths[idx])
        inp_pos_image = torchvision.io.read_image(dataset.right_imgs_paths[idx])
        
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
        _ = viz.visualize_image_attr_multiple(np.transpose(attr_ref.cpu().detach().numpy(), (1, 2, 0)),
                                            np.transpose(inp_ref_image.cpu().detach().numpy(), (1, 2, 0)),
                                            ["original_image", "heat_map"],
                                            ["all", "absolute_value"],
                                            cmap=default_cmap,
                                            show_colorbar=True)

        if False:                             
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