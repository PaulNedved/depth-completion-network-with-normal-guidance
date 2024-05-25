# Self-supervised Depth Completion with Calibration Enhancement and Normal Guidance

PyTorch implementation of *Self-supervised Depth Completion with Calibration Enhancement and Normal Guidance*

## Setting up virtual environment <a name="setting-up"></a>
 Create a virtual environment with the necessary dependencies.

CUDA 10.1
```
virtualenv -p /usr/bin/python3.7 kbnet-py37env
source kbnet-py37env/bin/activate
pip install opencv-python scipy scikit-learn scikit-image matplotlib gdown numpy gast Pillow pyyaml
pip install torch==1.3.0 torchvision==0.4.1 tensorboard==2.3.0
```

CUDA 11.1
```
virtualenv -p /usr/bin/python3.7 kbnet-py37env
source kbnet-py37env/bin/activate
pip install opencv-python scipy scikit-learn scikit-image matplotlib gdown numpy gast Pillow pyyaml
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install tensorboard==2.3.0
```

## Setting up datasets
For datasets, we will use KITTI for outdoors, here is the download script:
```
mkdir data
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_depth_completion data/
bash bash/setup_dataset_kitti.sh
```
## Training and Testing KBNet <a name="training-kbnet"></a>
To train KBNet on the KITTI dataset, run
```
bash bash/kitti/train_kbnet_kitti.sh
```
To run the model on the KITTI validation set or KITTI test set, you can use
```
bash bash/kitti/run_kbnet_kitti_validation.sh

bash bash/kitti/run_kbnet_kitti_testing.sh
```