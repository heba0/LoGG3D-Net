import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def similarity_downgrade(similarity):
    """
    for a input self-similarity matrix, higher similarity returns lower scores
    similarity range: 0 ~ 1 after softmax
    y = -log(x)
    """
    return -torch.log(similarity)

def attention_loss(weighted_local_desc, org_local_desc):
    """
    org_local_desc: the output from spvcnn in LOGG3D Net as local descriptor; dim: batch, N, d
    weight_local_desc: the weighted org_local_desc; dim: batch, N, d

    following self-attention scheme:
        softmax((X @ X.T) / d_k) * X

    """
    # TODO: when and where to layer normalize the matrix? normalize the local descriptor?

    softmax = nn.Softmax(dim=-1)
    d_k = math.sqrt(org_local_desc.shape[-1])
    
    # (X @ X.T) / d_k
    self_similarity = torch.bmm(org_local_desc, org_local_desc.permute(0, 2, 1)) / d_k # batch, N, N
    
    score = softmax(self_similarity)
    score = similarity_downgrade(score) 

    self_attention_desc = torch.bmm(score, org_local_desc) # batch, N, d 
    
    # layer norm?

    # calculatet the loss on the feature dimension
    loss = (weighted_local_desc - self_attention_desc).pow(2).sum(-1)

    return loss.mean()





