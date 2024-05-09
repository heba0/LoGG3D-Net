#!/bin/bash

torchpack dist-run -np "${_NGPU}" python training/train.py \
    --wildplaces_dir "/home/heba/Documents/UZH/PerceptionforRobotics/experiments/log3d_self_attention/LoGG3D-Net/data" \
    --dataset GeneralPointSparseTupleDataset \
    --positives_per_query 1 \
    --negatives_per_query 1 \
    --point_loss_weight 0 \
    --resume_checkpoint "{PATH_TO_PRETRAINED_CHECKPOINT}" \
    --resume_training True