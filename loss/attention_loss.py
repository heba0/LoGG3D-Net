import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pipelines.attention_weight import self_attention


def similarity_downgrade(similarity):
    """
    for a input self-similarity matrix, higher similarity returns lower scores
    similarity range: 0 ~ 1 after softmax
    y = -log(x)
    """
    return -torch.log(similarity)

def attention_loss(weighted_local_desc, org_local_desc):
    """
    org_local_desc: the output from spvcnn in LOGG3D Net as local descriptor; dim:  [N, d] x 2
    weight_local_desc: the weighted org_local_desc; dim: [N, d] x 2

    """
    self_attention_desc = []
    org_local_desc = list(org_local_desc)
    
    weights_0 = self_attention(org_local_desc)
    self_attention_desc.append(weights_0 * org_local_desc[0])

    # org_local_desc[0], org_local_desc[1] = org_local_desc[1], org_local_desc[0]
    # weights_1 = self_attention(org_local_desc)
    # self_attention_desc.append(weights_1 * org_local_desc[1])

    # org_local_desc[0], org_local_desc[1] = org_local_desc[1], org_local_desc[0]
    

    # softmax = nn.Softmax(dim=-1)
    # d_k = math.sqrt(org_local_desc.shape[-1])
    
    # # (X @ X.T) / d_k
    # self_similarity = torch.bmm(org_local_desc, org_local_desc.permute(0, 2, 1)) / d_k # batch, N, N
    
    # score = softmax(self_similarity)
    # score = similarity_downgrade(score) 

    # self_attention_desc = torch.bmm(score, org_local_desc) # batch, N, d 
    
    weighted_local_desc = torch.stack([weighted_local_desc[0]])
    self_attention_desc = torch.stack(self_attention_desc)
    # calculatet the loss on the feature dimension
    loss = (weighted_local_desc - self_attention_desc).pow(2).sum(-1)

    return loss.mean()





