# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax Loss as described in the paper.
    """

    def __init__(self, in_features, num_classes, margin=0.2, scale=32.0):
        super(AAMSoftmax, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = margin
        self.s = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embedding, label):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, label)