from models.losses.metrics import l2_dist_sum_weighted_per_batch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.5, reduction_mode="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction_mode = reduction_mode

    def forward(self, encodings):

        # get anchor, pos, negs from encodings
        anchor = [encodings[i]["anchor"] for i in range(len(encodings))]
        pos = [encodings[i]["pos"] for i in range(len(encodings))]
        negs = [encodings[i]["neg"] for i in range(len(encodings))]
        if len(negs[0]) != 1:
            raise ValueError(f"Only one negative is allowed for TripletLoss, but got: {len(negs[0])}")
        negs = [negs[k][0] for k in range(len(negs))]

        # calculate anchor->pos dist
        pos_dist = l2_dist_sum_weighted_per_batch(anchor, pos)

        # calculate anchor->neg dist for all negs
        neg_dist = l2_dist_sum_weighted_per_batch(anchor, negs)

        losses = F.relu(pos_dist - neg_dist + self.margin)

        # calculate triplet loss
        if self.reduction_mode == "mean":
            return F.relu(pos_dist - neg_dist + self.margin).mean(), pos_dist.mean().data.cpu().numpy(), neg_dist.mean().data.cpu().numpy()
        elif self.reduction_mode == "sum":
            return F.relu(pos_dist - neg_dist + self.margin).sum(), pos_dist.sum().data.cpu().numpy(), neg_dist.sum().data.cpu().numpy()
        else:
            raise ValueError(f"Unsupported reduction_mode:{self.reduction_mode}")

