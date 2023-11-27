# Basic libs
import os, time, glob, random, pickle, copy, torch
import numpy as np
import open3d
from scipy.spatial.transform import Rotation
from torch.utils import data
from scripts.cal_overlap import get_overlap_ratio

# Dataset parent class
from torch.utils.data import Dataset
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_correspondences


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result


class ETHDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """

    def __init__(self,config,split,data_augmentation=True):
        super(ETHDataset,self).__init__()
        self.voxel_size = config.first_subsampling_dl
        self.rescale_pcd_factor = config.rescale_pcd_factor
        self.config = config

        # Initiate containers
        self.files = []
        self.prepare_eth_ply()
        self.split = split
        assert split == 'test'


    def prepare_eth_ply(self):
        scene_list = [
            'gazebo_summer',
            'gazebo_winter',
            'wood_autmn',
            'wood_summer',
        ]
        for scene in scene_list:
            pcdpath = f"./ETH/{scene}/"
            interpath = f"./ETH/{scene}/01_Keypoints/"
            gtpath = f'./ETH/{scene}/'
            gtLog = loadlog(gtpath)

            # register each pair
            fragments = glob.glob(pcdpath + '*.ply')
            num_frag = len(fragments)
            start_time = time.time()
            for id1 in range(num_frag):
                for id2 in range(id1 + 1, num_frag):
                    self.files.append((pcdpath, id1, id2))
            

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        pcdpath = self.files[idx][0]
        id1, id2 = self.files[idx][1], self.files[idx][2]

        cloud_bin_s = f'Hokuyo_{id1}'
        cloud_bin_t = f'Hokuyo_{id2}'
        write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
        pcd0_original = open3d.io.read_point_cloud(os.path.join(pcdpath, cloud_bin_s + '.ply'), format='ply')
        pcd1_original = open3d.io.read_point_cloud(os.path.join(pcdpath, cloud_bin_t + '.ply'), format='ply')

        xyz0 = np.asarray(pcd0_original.points)
        xyz1 = np.asarray(pcd1_original.points)

        # scale-up points by a factor instead of shrinking the voxel size
        xyz0 = xyz0 * self.rescale_pcd_factor
        xyz1 = xyz1 * self.rescale_pcd_factor
        pcd0 = to_o3d_pcd(xyz0)
        pcd1 = to_o3d_pcd(xyz1)

        # voxelize the point clouds here
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        src_pcd = np.array(pcd0.points)
        tgt_pcd = np.array(pcd1.points)

        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        rot = np.zeros((3,3))
        trans = np.zeros(3)

        # compatibility
        src_pcd_input = copy.deepcopy(src_pcd)
        tgt_pcd_input = copy.deepcopy(tgt_pcd)

        return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans, torch.Tensor([0,0]), src_pcd/self.rescale_pcd_factor, tgt_pcd/self.rescale_pcd_factor, torch.ones(1)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts
