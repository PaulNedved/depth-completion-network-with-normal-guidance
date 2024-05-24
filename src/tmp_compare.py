import numpy as np
from PIL import Image
import data_utils
import torch
import datasets


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list


train_image_path = 'training/kitti/kitti_train_image-clean.txt'
train_sparse_depth_path = 'training/kitti/kitti_train_sparse_depth-clean.txt'
train_intrinsics_path = 'training/kitti/kitti_train_intrinsics-clean.txt'
dense_normal_path = 'training/kitti/kitti_train_dense_normal-clean.txt'

train_image_paths = data_utils.read_paths(train_image_path)
train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)
train_dense_normal_paths = data_utils.read_paths(dense_normal_path)

train_dataloader = torch.utils.data.DataLoader(
    datasets.KBNetTrainingDataset(
        image_paths=train_image_paths,
        sparse_depth_paths=train_sparse_depth_paths,
        intrinsics_paths=train_intrinsics_paths,
        normal_paths=train_dense_normal_paths,
        shape=(320, 768),
        random_crop_type=['horizontal', 'vertical', 'anchored', 'bottom']),
    batch_size=8,
    shuffle=True,
    num_workers=8,
    drop_last=False)

print(type(train_dataloader))

for inputs in train_dataloader:
    image0, image1, image2, sparse_depth0, normal0, intrinsics = inputs

