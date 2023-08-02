import torch
import torch.nn as nn
import torch.nn.functional as F


class UnmaskedCodeLoss(nn.Module):
    def __init__(self, loss_type="BCE"):
        super(UnmaskedCodeLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "BCE":
            self.code_loss_func = nn.BCEWithLogitsLoss(reduction="mean")
        elif loss_type == "L1":
            self.code_loss_func = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("loss_type {} not supported in MaskedCodeLoss".format(loss_type))

    def forward(self, pred_code_prob, gt_code):
        ''' Args:
        pred_code_prob: shape: (batch, #bits, #keypoints)
        gt_code: shape: (batch, #bits, #keypoints)
        '''
        if self.loss_type == "L1":
            loss = self.code_loss_func(torch.sigmoid(pred_code_prob), gt_code)
        else:
            loss = self.code_loss_func(pred_code_prob, gt_code)
        return loss


class MaskedCodeLoss(nn.Module):
    def __init__(self, loss_type="BCE"):
        super(MaskedCodeLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "BCE":
            self.code_loss_func = nn.BCEWithLogitsLoss(reduction="none")
        elif loss_type == "L1":
            self.code_loss_func = nn.L1Loss(reduction="none")
        elif loss_type == "CE":  # for multi-class classification
            self.code_loss_func = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError("loss_type {} not supported in MaskedCodeLoss".format(loss_type))

    def forward(self, pred_code_prob, gt_code, gt_mask):
        ''' Args:
        pred_code_prob: shape: (batch, #bits, #keypoints)
        gt_code: shape: (batch, #bits, #keypoints) or (batch, 1, #keypoints) for multi-class classification
        gt_mask: shape: (batch, 1, #keypoints)
        '''
        if self.loss_type == "CE":
            num_bits = 1
        else:
            num_bits = pred_code_prob.shape[1]
        if self.loss_type == "L1":
            raw_loss = self.code_loss_func(torch.sigmoid(pred_code_prob), gt_code)
        elif self.loss_type == "CE":
            raw_loss = self.code_loss_func(pred_code_prob, gt_code[:, 0, :])  # shape: (batch, #keypoint)
            raw_loss = raw_loss.unsqueeze(dim=1)  # shape: (batch, 1, #keypoints)
        else:
            raw_loss = self.code_loss_func(pred_code_prob, gt_code)
        raw_loss = raw_loss * gt_mask  # shape: (B, #bits, #keypoints)
        mask_sum = gt_mask.sum().float().clamp(min=1.0) * num_bits
        loss = raw_loss.sum() / mask_sum
        return loss
