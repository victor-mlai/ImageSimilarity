
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tqdm

from my_models import SiameseNetAdaptive
from my_dataset import MyTLLDataset
from arg_parser import build_arg_parser


@torch.no_grad()
def eval(model: SiameseNetAdaptive, valid_ds, device):
    print("Calculating embeddings")
    left_embeddings = [
        model._forward(left.unsqueeze(0).to(device)).cpu()
        for left, _, _ in valid_ds
    ]
    right_embeddings = [
        model._forward(right.unsqueeze(0).to(device)).cpu()
        for _, right, _ in valid_ds
    ]
    print("Done")
    print("Finding matches...")

    topk = 100
    distrib = [0 for _ in range(topk)]

    correct = 0
    for left_idx, left_emb in enumerate(left_embeddings):
        pred_sim = []
        for right_emb in right_embeddings:
            sim = model.adaptive_func(torch.cat((left_emb.to(device), right_emb.to(device)), 1))

            pred_sim.append(sim.item())

        pred_probs_idxs = np.array(pred_sim).argsort()[::-1][:topk]
        for i, idx in enumerate(pred_probs_idxs):
            if idx == left_idx:
                correct += 1
                distrib[i] += 1
                print(f"Found the match for img {idx} Placed at pos {i}")
                break

            if False:
                plt.figure(figsize=((topk+2) * 10, 10))
                plt.subplot(1, topk+2, 1)
                anchor, _, _ = valid_ds[idx]
                plt.imshow(np.transpose(valid_ds[idx].numpy(), (1, 2, 0)))
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

    print(f"Found {correct}/{len(valid_ds)} in the top-{topk}")
    plt.bar(range(topk), distrib)
    plt.title(f'Rank frequency for Top-{topk} matches')
    plt.ylabel('Frequency')
    plt.xlabel('Rank')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    print("Loading dataset..")
    normalize_tf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = MyTLLDataset(args.dataset_dir, transform, no_faces=True)
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_ds, test_ds = random_split(dataset, [train_set_size, valid_set_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print("Dataset loaded.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #encoder = models.resnet50()
    encoder = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    num_ftrs = encoder.fc.in_features
    encoder.fc = torch.nn.Identity()

    # TODO Freeze rest of net
    for name, param in encoder.named_parameters():
    # TODO Unfreeze BN layers
        #if not 'bn' in name:
            param.requires_grad = False
    encoder.eval()

    model = SiameseNetAdaptive(encoder, num_ftrs, args).to(device)

    since = time.time()

    best_valid_loss = 9999999.
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        print('-' * 10)

        # Train phase
        model.adaptive_func.train()

        running_loss = 0.0
        for anchor, pos, neg in tqdm.tqdm(train_loader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            anchor = model._forward(anchor)
            pos = model._forward(pos)
            neg = model._forward(neg)

            model.optimizer.zero_grad()
            output_pos = model.adaptive_func(torch.cat((anchor, pos), 1))
            target = torch.ones_like(output_pos, device=device)
            loss_pos = nn.functional.binary_cross_entropy_with_logits(output_pos, target)
            loss_pos.backward()
            model.optimizer.step()

            model.optimizer.zero_grad()
            output_neg = model.adaptive_func(torch.cat((anchor, neg), 1))
            target = torch.zeros_like(output_neg, device=device)
            loss_neg = nn.functional.binary_cross_entropy_with_logits(output_neg, target)
            loss_neg.backward()
            model.optimizer.step()

            # statistics
            running_loss += (loss_pos.item() + loss_neg.item()) * args.batch_size
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        train_losses_per_epoch.append(epoch_loss)


        # Validation phase
        model.adaptive_func.eval()

        running_loss = 0.0
        for anchor, pos, neg in tqdm.tqdm(test_loader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            anchor = model._forward(anchor)
            pos = model._forward(pos)
            neg = model._forward(neg)

            with torch.no_grad():
                output_pos = model.adaptive_func(torch.cat((anchor, pos), 1))
            target = torch.ones_like(output_pos, device=device)
            loss_pos = nn.functional.binary_cross_entropy_with_logits(output_pos, target)

            with torch.no_grad():
                output_neg = model.adaptive_func(torch.cat((anchor, neg), 1))
            target = torch.zeros_like(output_neg, device=device)
            loss_neg = nn.functional.binary_cross_entropy_with_logits(output_neg, target)

            # statistics
            running_loss += (loss_pos.item() + loss_neg.item()) * args.batch_size
        
        epoch_loss = running_loss / len(test_loader.dataset)
        print(f'Valid Loss: {epoch_loss:.4f}')

        test_losses_per_epoch.append(epoch_loss)

        # Save checkpoint
        if epoch_loss < best_valid_loss:
            best_valid_loss = epoch_loss
            best_clsf_wts = copy.deepcopy(model.state_dict())

        model.lr_scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_valid_loss:4f}')

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    torch.save(best_clsf_wts, args.checkpoint_dir / args.model_filename)
    print(f"Saved best model to {args.checkpoint_dir}")

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    time_stamp = datetime.datetime.today().strftime("[%Y-%m-%d-%H-%M-%S]")
    np.savez(args.logs_dir / time_stamp,
        train_losses_per_epoch = train_losses_per_epoch,
        test_losses_per_epoch = test_losses_per_epoch,
    )
    print(f"Saved stats to {args.logs_dir}")

    eval(model, test_ds, device)