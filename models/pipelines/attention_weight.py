import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


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
        anchor_pc = norm_local_feat[0]
        self_sim_score = torch.zeros(num_feat).to(anchor_pc.device)
    
        for i in range(batch_size):
            target_pc = norm_local_feat[i]
            self_sim_matrix = torch.matmul(anchor_pc, target_pc.T) # output dim: N x N
            
            self_sim_sub    = self_sim_matrix.mean(-1)
            del self_sim_matrix
            self_sim_score += self_sim_sub
            del self_sim_sub
        
        self_sim_score = self_sim_score / batch_size
        
        # Calculate the weight based on the idea: higher similarity deserve less score
        ## OPTION 1
        # "idf" score
        # self_sim_weight = torch.log(self_sim_score.shape[0] / (self_sim_score + 0.1))
        # norm_sim_score = nn.functional.normalize(self_sim_score.unsqueeze(0), p=2)
        # self_sim_softmax = softmax(norm_sim_score).squeeze(0)
        # self_sim_weight = torch.log(1 / (self_sim_softmax + 0.1))
        # print('****** self_sim_weight: ', self_sim_weight)
    
        ## OPTION 2
        # standardization, scale it to zero mean and unit standard deviation
        self_sim_score_stand = (self_sim_score - torch.mean(self_sim_score)) / torch.std(self_sim_score)
        
        self_sim_weight = torch.exp(-self_sim_score_stand)      

        ## OPTION 3
        # normalize + softmax
#         norm_sim_score = nn.functional.normalize(self_sim_score.unsqueeze(0), p=2)
#         self_sim_weight = softmax(1 - norm_sim_score).squeeze(0)

        self_sim_weight = self_sim_weight.reshape(num_feat, 1)
        
    
    return self_sim_weight



