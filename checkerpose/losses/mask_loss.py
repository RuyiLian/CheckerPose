import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLoss_interpolate(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, pred_mask, groundtruth_mask):
        b, c, h, w = pred_mask.shape
        pred_mask = pred_mask[:, 0, :, :]
        pred_mask = torch.sigmoid(pred_mask)
        resized_mask = F.interpolate(groundtruth_mask[:, None], size=(h, w), mode="nearest")
        resized_mask = resized_mask[:, 0, :, :]
        return self.loss(pred_mask, resized_mask)
