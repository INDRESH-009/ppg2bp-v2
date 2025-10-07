import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1(nn.Module):
    def __init__(self, sbp_weight=1.5, dbp_weight=1.0, beta=1.0):
        super().__init__()
        self.sbp_w = sbp_weight
        self.dbp_w = dbp_weight
        self.beta = beta

    def forward(self, pred, target):
        # pred/target: [B,2] (SBP, DBP)
        l_sbp = F.smooth_l1_loss(pred[:,0], target[:,0], beta=self.beta)
        l_dbp = F.smooth_l1_loss(pred[:,1], target[:,1], beta=self.beta)
        return self.sbp_w * l_sbp + self.dbp_w * l_dbp
