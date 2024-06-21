import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.SOP import *
from models.backbones.spvnas.model_zoo import spvcnn
from models.pipelines.pipeline_utils import *
from models.pipelines.attention_weight import self_attention

__all__ = ['LOGG3D']


class LOGG3D_ATTN(nn.Module):
    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()
        
        self.spvcnn = spvcnn(output_dim=feature_dim)
        self.sop = SOP(
            signed_sqrt=False, do_fc=False, input_dim=feature_dim, is_tuple=False)

    def forward(self, x, topK=1):
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.spvcnn(x)
        y = torch.split(x, list(counts))
        
        # y = list(y)
        # weights = self_attention(y)
        # y[0] = weights * y[0]
        
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        
        if not self.training and topK > 0:
            y = list(y)
            weights = self_attention(y)
            weighted_feat = x.squeeze(0) * weights
            topK = min(topK, weights.shape[0]) 
            top_indices = torch.topk(weights, k=int(len(weighted_feat) * topK) , dim=0).indices # int(len(weighted_feat) * (topK / 10))
            x = weighted_feat[top_indices] # dim: (N //topK, 1, feature_dim)
            x = x.permute(1, 0, 2)
        x = self.sop(x)
        return x, y[:2] # slice to 2 is to keep only anchor and positive pair, ignore negative sample and others


if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir)
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)
    lidar_pc = lidar_pc.reshape(-1, 4)
    print('==== lidar_pc: ', lidar_pc.shape)
    input = make_sparse_tensor(lidar_pc, 0.05).cuda()

    model = LOGG3D().cuda()
    model.train()
    output = model(input)
    print('output size: ', output[0].size())
