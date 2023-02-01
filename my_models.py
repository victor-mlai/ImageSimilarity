import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

class SiameseNetAdaptive(nn.Module):
    def __init__(self, encoder, enc_out_dim, args):
        super(SiameseNetAdaptive, self).__init__()
        self.encoder = encoder
        self.enc_out_dim = enc_out_dim
        self.args = args

        self.adaptive_func = nn.Sequential(
            nn.Linear(enc_out_dim * 2, 256), # *2 for the 2 input images
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam(
            self.adaptive_func.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)

        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)


    def _forward(self, x):
        with torch.no_grad():
            out = self.encoder(x)

        out = out.view(out.size()[0], -1)

        return out

    def forward(self, anchor, pos, neg, device):
        anchor = self._forward(anchor)
        pos = self._forward(pos)
        neg = self._forward(neg)

        # concatenate both images' features
        output_pos = torch.cat((anchor, pos), 1)
        output_pos = self.adaptive_func(output_pos)

        target = torch.ones_like(output_pos, device=device)
        loss_pos = nn.functional.binary_cross_entropy_with_logits(output_pos, target)

        # negative pair
        output_neg = torch.cat((anchor, neg), 1)
        output_neg = self.adaptive_func(output_neg)

        target = torch.zeros_like(output_neg, device=device)
        loss_neg = nn.functional.binary_cross_entropy_with_logits(output_neg, target)

        return anchor, pos, neg, loss_pos + loss_neg


class SiameseNetTriplet(nn.Module):
    def __init__(self, encoder, enc_out_dim, args):
        super(SiameseNetTriplet, self).__init__()
        self.encoder = encoder
        self.enc_out_dim = enc_out_dim
        self.args = args

        self.proj_head = nn.Sequential(
            nn.Linear(enc_out_dim, self.args.hidden_dim, bias=True),
            nn.BatchNorm1d(self.args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.args.hidden_dim, self.args.output_dim, bias=False),
        )

        self.optimizer = optim.Adam(
            self.proj_head.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)

        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)


    def _forward(self, x):
        out = self.encoder(x)

        out = out.view(out.size()[0], -1)

        out = self.proj_head(out)

        return out

    def forward(self, anchor, pos, neg):
        anchor = self._forward(anchor)
        pos = self._forward(pos)
        neg = self._forward(neg)

        loss = nn.functional.triplet_margin_loss(anchor, pos, neg)

        return anchor, pos, neg, loss

