import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def similarity_downgrade(similarity):
    """
    for a input self-similarity score, higher similarity returns lower scores
    """
    return torch.exp(-similarity)


def self_attention(local_feat):
    """
    local_feat: the output from spvcnn in LOGG3D Net as local descriptor; 
                dim: batch, N, d
    
    """
    self_sim_weight = None
    
    with torch.no_grad():
        local_feat_c = local_feat.clone()
        dim_feat     = local_feat_c.data.shape[-1] # dim_feat: 16
        batch_size   = local_feat_c.data.shape[0]
        num_feat     = local_feat_c.data.shape[1]
        softmax = nn.Softmax(dim=-1)
        # d_k = math.sqrt(local_feat.shape[-1])

        norm_local_feat = nn.functional.normalize(local_feat_c, p=2, dim=-1)

        concat_local_feat = torch.reshape(norm_local_feat, (-1, dim_feat)) # dim: (B*N, dim_feat)
        
        # calculate simialrity between each feature across all samples in the minibatch
        self_sim = torch.matmul(concat_local_feat, concat_local_feat.T)
        
        # the value range from -1 to 1
        self_sim_score = self_sim.sum(-1) # dim: (B*N, 1)
        
        ## OPTION 1
        # standardization, scale it to zero mean and unit standard deviation
#         self_sim_score_stand = (self_sim_score - torch.mean(self_sim_score)) / torch.std(self_sim_score)
        
#         self_sim_weight = torch.exp(-self_sim_score_stand)
        
        
        ## OPTION 2
        # "idf" score
        self_sim_weight = torch.log(self_sim_score.shape[0] / self_sim_score)
        
        
        self_sim_weight = self_sim_weight.reshape(batch_size, num_feat, 1)
        
    
    return self_sim_weight



