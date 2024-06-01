import os
import sys
import random
import numpy as np
import json
import csv
import pandas as pd

def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Rotation matrix
    rotation_matrix = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

    return rotation_matrix

def pose_to_matrix(self, x, y, z, qx, qy, qz, qw):
    rotation_matrix = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
    translation_vector = np.array([[x], [y], [z]])

    # Assemble transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector.flatten()

    return transformation_matrix

#####################################################################################
# Load poses
#####################################################################################

def wp_load_poses_from_csv(file_name):
    df = pd.read_csv(file_name, delimiter = ',', dtype = str)
    df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, 'timestamp': float})
    
    position = df[['x', 'y', 'z']].to_numpy()
    return position
    
    
    
#     with open(file_name, newline='') as f:
#         reader = csv.reader(f)
#         data_poses = list(reader)

#     transforms = []
#     positions = []
#     for cnt, line in enumerate(data_poses):
#         line_f = [float(i) for i in line]
#         P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
#         transforms.append(P)
#         positions.append([P[0, 3], P[1, 3], P[2, 3]])
    # return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def wp_load_timestamps_csv(file_name):
    df = pd.read_csv(file_name, delimiter = ',', dtype = str)
    df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, 'timestamp': float})
    data_poses_ts = df['timestamp'].values
    # with open(file_name, newline='') as f:
    #     reader = csv.reader(f)
    #     data_poses = list(reader)
    # data_poses_ts = np.asarray(
    #     [float(t)/1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts
