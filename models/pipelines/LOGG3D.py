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


class LOGG3D(nn.Module):
    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()

        self.spvcnn = spvcnn(output_dim=feature_dim)
        self.sop = SOP(
            signed_sqrt=False, do_fc=False, input_dim=feature_dim, is_tuple=False)
        # self.mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Softmax(dim=-1)) # input (B, N, feature_dim) -> output (B, N, feature_dim)

    def forward(self, x):
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.spvcnn(x)
        y = torch.split(x, list(counts))
        
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        
        weights = self_attention(x)
        x[0] = x[0] * weights
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
