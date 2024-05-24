#!/bin/bash

mkdir -p data0/kitti_depth_completion
mkdir -p data0/kitti_depth_completion/train_val_split
mkdir -p data0/kitti_depth_completion/train_val_split/sparse_depth
mkdir -p data0/kitti_depth_completion/train_val_split/ground_truth
mkdir -p data0/kitti_depth_completion/validation
mkdir -p data0/kitti_depth_completion/testing
mkdir -p data0/kitti_depth_completion/tmp

unzip data0/data_depth_velodyne.zip -d data0/kitti_depth_completion/train_val_split/sparse_depth
unzip data0/data_depth_annotated.zip -d data0/kitti_depth_completion/train_val_split/ground_truth
unzip data0/data_depth_selection.zip -d data0/kitti_depth_completion/tmp

mv data0/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/image data0/kitti_depth_completion/validation/image
mv data0/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/velodyne_raw data0/kitti_depth_completion/validation/sparse_depth
mv data0/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/groundtruth_depth data0/kitti_depth_completion/validation/ground_truth
mv data0/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/intrinsics data0/kitti_depth_completion/validation/intrinsics

mv data0/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/image data0/kitti_depth_completion/testing/image
mv data0/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/velodyne_raw data0/kitti_depth_completion/testing/sparse_depth
mv data0/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/intrinsics data0/kitti_depth_completion/testing/intrinsics

rm -r data0/kitti_depth_completion/tmp

