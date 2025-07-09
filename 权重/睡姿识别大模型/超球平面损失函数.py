import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineMarginProduct(nn.Module):
    def __init__(self, feat_dim=2048, class_num=21, s=6.0, m=0.2):
        super(CosineMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.score = nn.Linear(feat_dim, class_num, bias=False)
        self.score.weight.data.normal_(0, 0.01)

    def forward(self, features, labels):
        if labels is not None:
            cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.score.weight.data, dim=0))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            roi_scores = self.s * (cosine - one_hot * self.m)
        else:
            roi_scores = self.score(features)
        return roi_scores