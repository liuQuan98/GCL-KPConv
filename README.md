# Density-invariant Features for Distant Point Cloud Registration (ICCV 2023)

Registration of distant outdoor LiDAR point clouds is crucial to extending the 3D vision of collaborative autonomous vehicles, and yet is challenging due to small overlapping area and a huge disparity between observed point densities. In this paper, we propose Group-wise Contrastive Learning (GCL) scheme to extract density-invariant geometric features to register distant outdoor LiDAR point clouds. We mark through theoretical analysis and experiments that, contrastive positives should be independent and identically distributed (i.i.d.), in order to train density-invariant feature extractors. We propose upon the conclusion a simple yet effective training scheme to force the feature of multiple point clouds in the same spatial location (referred to as positive groups) to be similar, which naturally avoids the sampling bias introduced by a pair of point clouds to conform with the i.i.d. principle. The resulting fully-convolutional feature extractor is more powerful and density-invariant than state-of-the-art methods, improving the registration recall of distant scenarios on KITTI and nuScenes benchmarks by 40.9% and 26.9%, respectively.

Link to the conference version: [Link](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Density-invariant_Features_for_Distant_Point_Cloud_Registration_ICCV_2023_paper.html)

Link to the arxiv version: [Link](https://arxiv.org/abs/2307.09788)

This repository is the implementation of GCL-KPConv upon a tweaked version of [Predator](https://github.com/prs-eth/OverlapPredator). Another implementation with Minkowski Convolution is available [here](https://github.com/liuQuan98/GCL).

## News

20231127 - Code and pretrained weights are released [here](https://drive.google.com/file/d/17rt_eNBiLdOr5WxxYz8rOuUDwGsnDTXZ/view?usp=sharing). We recommend unzipping it under the project directory.

20230713 - Our paper has been accepted by ICCV'23!

## Overview of Group-wise Contrastive Learning (GCL)

<div align="center">
<img src=assets\overview.png>
</div>

## Results

All results below are tested at a metric of TE<2m, RE<5°.

KITTI:

| Method       |  RR  | RRE  | RTE  |
| :----------- | :---: | ---- | ---- |
| FCGF         | 97.8 | 0.35 | 12.6 |
| FCGF+APR     | 98.2 | 0.34 | 9.6  |
| Predator     | 100.0 | 0.31 | 7.4  |
| Predator+APR | 100.0 | 0.30 | 7.3  |
| GCL+Conv     | 98.6 | 0.26 | 6.62 |
| GCL+KPConv   | 99.2 | 0.25 | 7.50 |

LoKITTI:

| Method       |  RR  | RRE  | RTE  |
| :----------- | :--: | ---- | ---- |
| FCGF         | 22.2 | 2.02 | 55.2 |
| FCGF+APR     | 32.7 | 1.74 | 51.9 |
| Predator     | 42.4 | 1.75 | 43.4 |
| Predator+APR | 50.8 | 1.64 | 39.5 |
| GCL+Conv     | 72.3 | 1.03 | 25.9 |
| GCL+KPConv   | 55.4 | 1.28 | 27.8 |

nuScenes:

| Method       |  RR  | RRE  | RTE  |
| :----------- | :--: | ---- | ---- |
| FCGF         | 93.6 | 0.46 | 50.0 |
| FCGF+APR     | 94.5 | 0.45 | 37.0 |
| Predator     | 97.8 | 0.58 | 20.2 |
| Predator+APR | 99.5 | 0.47 | 19.1 |
| GCL+Conv     | 99.2 | 0.30 | 16.7 |
| GCL+KPConv   | 99.7 | 0.35 | 15.9 |

LoNuScenes:

| Method       |  RR  | RRE  | RTE  |
| :----------- | :--: | ---- | ---- |
| FCGF         | 49.1 | 1.30 | 60.9 |
| FCGF+APR     | 51.8 | 1.40 | 62.0 |
| Predator     | 50.4 | 1.47 | 54.5 |
| Predator+APR | 62.7 | 1.30 | 51.8 |
| GCL+Conv     | 82.4 | 0.70 | 46.8 |
| GCL+KPConv   | 86.5 | 0.84 | 46.5 |

## Requirements

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher

## Dataset Preparation

### KITTI

For KITTI dataset preparation, please first follow the [KITTI official instructions](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the 'velodyne laser data', 'calibration files', and (optionally) 'ground truth poses'.

Since the GT poses provided in KITTI drift a lot, we recommend using the pose labels provided by [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) instead, as they are more accurate. Please follow the official instruction to download the split called 'SemanticKITTI label data'.

Extract all compressed files in the same folder and we are done. We denote KITTI_ROOT as the directory that have the following structure: `{$KITTI_ROOT}/dataset/poses` and `{$KITTI_ROOT}/dataset/sequences/XX`.

The option to use KITTI original pose is still preserved which can be enabled by setting `use_old_pose` to True in the scripts, although we highly discourage doing so due to performance degredation. Please note that all of the methods reported in our paper are retrained on the label of SemanticKITTI instead of OdometryKITTI.

### nuScenes

The vanilla nuScenes dataset structure is not friendly to the registration task, so we propose to convert the lidar part into KITTI format for ease of development and extension. Thanks to the code provided by nuscenes-devkit, the conversion requires only minimal modification.

To download nuScenes, please follow the [nuscenes official page](https://www.nuscenes.org/nuscenes#download) to obtain the 'lidar blobs' (inside 'file blobs') and 'Metadata' of the 'trainval' and 'test' split in the 'Full dataset (v1.0)' section. Only LiDAR scans and pose annotations are used.

Next, execute the following commands to deploy [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and our conversion script:

```
git clone https://github.com/nutonomy/nuscenes-devkit.git
conda create -n nuscenes-devkit python=3.8
conda activate nuscenes-devkit
pip install nuscenes-devkit
cp ./assets/export_kitti_minimal.py ./nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti_minimal.py
```

Cater the `nusc_dir` and `nusc_kitti_dir` parameter in `./nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti_minimal.py` line 51 & 52 to your preferred path. Parameter `nusc_dir` specifies the path to the nuScenes dataset, and `nusc_kitti_dir` will be the path to store the converted nuScenes LiDAR data. Start conversion by executing the following instructions:

```
cd ./nuscenes-devkit/python-sdk
python nuscenes/scripts/export_kitti_minimal.py
```

The process may be slow (can take hours).

## Installation

We recommend conda for installation. First, we need to create a basic environment to setup MinkowskiEngine:

```
conda create -n gcl-kpconv python=3.7 pip=21.1
conda activate gcl-kpconv
pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```

### Setting the distance between two LiDARs (registration difficulty during testing)

As the major focus of this paper, we divide the PCL registration datasets  into different slices according to the distance $d$ between two LiDARs, both during testing and PCL training. Greater $d$ leads to a smaller overlap and more divergent point density, resulting in a higher registration difficulty. We denote range of $d$ with the parameter `--pair_min_dist` and `--pair_max_dist`, which can be found in `./scripts/train_{$method}_{$dataset}.sh`. For example, setting

```
--pair_min_dist 5 \
--pair_max_dist 20 \
```

will set $d\in [5m,20m]$. In other words, for every pair of point clouds, the ground-truth euclidean distance betwen two corresponding LiDAR positions (i.e., the origins of the two specified point clouds) obeys a uniform distribution between 5m and 20m.

### Launch the training

Notes:

1. Remember to set `--use_old_pose` to true when using the nuScenes dataset.
2. When dealing with GCL training, there is no need to alter the `pair_min_dist, pair_max_dist, min_dist, max_dist` parameters. The former two parameters specifies the dataset split used to assess model performance during validation, which will not affect the model itself; The latter two are used to specify the minimum and maximum range to select neighborhood point clouds in GCL, which is our selected optimal parameter.

To train GCL-KPConv on either dataset, run either of the following command inside conda environment `gcl-kpconv`:

```
python main.py configs/train/kitti.yaml
python main.py configs/train/nuscenes.yaml
```

### Testing

To test GCL-KPConv on either dataset, run the corresponding commands below. Do not forget to set  `pretrain` to the specific model path before running the corresponding script inside conda environment `gcl-kpconv`:

```
python main.py configs/test/kitti.yaml
python main.py configs/test/kitti.yaml
```

### Generalization to ETH

Install pytorch3d by the following command:

```
pip install pytorch3d
```

Then download ETH dataset from the official [website](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration), and organize the gazebo_summer, gazebo_winter, wood_autmn, and wood_summer splits in the following structure right in the GCL-KPConv-master directory:

```
--ETH--gazebo_summer
    |--gazebo_winter
    |--wood_autmn
    |--wood_summer
```

Change the `checkpoint` in `generalization_ETH/evaluate.py`, line 235 to your specified model checkpoint path, then run the following commands:

```
cd generalization_eth
python evaluate.py
```

## Acknowlegdements

We thank [Predator](https://github.com/prs-eth/OverlapPredator) for the wonderful baseline and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for the convenient dataset conversion code.
