import os, time, glob, random, pickle, copy, torch
import numpy as np
import open3d
from scipy.spatial.transform import Rotation
from torch.utils import data
from scripts.cal_overlap import get_overlap_ratio

# Dataset parent class
from torch.utils.data import Dataset
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_matching_indices_colocation_simple, sample_random_trans, follow_presampled_trans, get_matching_indices_colocation


class ColocationNuscenesDataset(Dataset):
    '''
    Training phase dataloader that loads a point cloud and a random neighborhood.
    Only compatible during training phase, and should be used with Finest-Contrastive Loss.
    '''
    
    def __init__(self,config,split,data_augmentation=True):
        super(ColocationNuscenesDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root, split)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min

        self.MIN_DIST = config.min_dist
        self.MAX_DIST = config.max_dist
        self.num_neighborhood = config.num_neighborhood
        assert self.num_neighborhood % 2 == 0, "Parameter 'num_neighborhood' must be even!"

        print(f"Loading the subset {split} from {self.root}")
        self.split = split
        assert self.split == 'train', "Colocation Data Loader loads a point cloud and its neighbourhood, which is only meaningful during training time!"

        self.area_length_per_neighbor = 2*self.MAX_DIST / self.num_neighborhood

        # this assertion ensures the inner area can spawn a neighborhood point cloud
        assert self.MIN_DIST < self.area_length_per_neighbor, "MIN_DIST is too high compared to area_length_per_neighbor! Lower MIN_DIST or lower num_neighborhood instead."
        self.config = config

        # Initiate containers
        self.files = []
        self.nuscenes_icp_cache = {}
        self.nuscenes_cache = {}
        self.split = split
        self.prepare_nuscenes_ply_colocation()
        print(f"Data size for phase {split}: {len(self.files)}")

    def prepare_nuscenes_ply_colocation(self):
        # load all frames that have a full spatial neighbourhood
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[min(int(self.MAX_DIST * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # find the current neighborhood
                skip, nghb = self._get_neighborhood_frames(curr_time)

                if skip:
                    curr_time += 1
                else:
                    self.files.append((dirname, curr_time, nghb))
                    curr_time += 11 # empirical distance parameter between centers


    def _get_neighborhood_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.MAX_DIST))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(left_dist > dist_tmp)[0]
            # print(candidates)
            if len(candidates) == 0:
                # No left-side complement detected
                skip_flag = True
                break
            else:
                list_complement.append(left_frame_bound + candidates[-1])
        
        if skip_flag:
            return (True, [])

        # Find the frames in front of me   
        right_dist = (self.Ts[frame: frame+int(10*self.MAX_DIST)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(right_dist > dist_tmp)[0]
            if len(candidates) == 0:
                # No right-side complement detected
                skip_flag = True
                list_complement = []
                break
            else:
                list_complement.append(frame + candidates[0])
        return (skip_flag, list_complement)
        
    def __len__(self):
        return len(self.files)

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def __getitem__(self, idx):
        dirname, t, t_cmpl = self.files[idx]
        positions = self.get_video_odometry(dirname, [t] + t_cmpl)
        pos = positions[0]
        pos_cmpl = positions[1:]

        # load center point cloud
        xyz = self._get_xyz(dirname, t)

        # load neighbourhood point clouds
        xyz_cmpl = []
        for t_tmp in t_cmpl:
            xyz_cmpl.append(self._get_xyz(dirname, t_tmp))

        # use world-coordinate label to calculate GT transformation
        def GetListM(pos_core, pos_cmpls):
            return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, int(self.num_neighborhood/2))] + \
                    [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(int(self.num_neighborhood/2), len(pos_cmpls))]
        list_M = GetListM(pos, pos_cmpl)

        matching_search_voxel_size = self.matching_search_voxel_size

        # voxelization
        xyz = torch.from_numpy(xyz)
        for i in range(len(xyz_cmpl)):
            xyz_cmpl[i] = torch.from_numpy(xyz_cmpl[i])

        # Make voxelized center points and voxelized center PC
        pcd = to_o3d_pcd(xyz)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        xyz_th = np.array(pcd.points)

        # Make both voxelized-unaligned nghb, and PCs
        # They are voxelized with the same index set (sel_nghb) so that both are synonimous w.r.t. point index
        pcd_cmpl = []
        xyz_cmpl_th = []
        for i in range(len(xyz_cmpl)):
            pcd_tmp = to_o3d_pcd(xyz_cmpl[i])
            pcd_tmp = pcd_tmp.voxel_down_sample(self.voxel_size)
            pcd_cmpl.append(pcd_tmp)
            xyz_cmpl_th.append(np.array(pcd_tmp.points))

        # Get matches
        group, index, finest_flag, central_distance = get_matching_indices_colocation(pcd, pcd_cmpl, xyz_cmpl_th, list_M, matching_search_voxel_size, K=5)
        if(len(index) == 0 and self.split == 'train'):
            return self.__getitem__(np.random.choice(len(self.files),1)[0])

        group = torch.Tensor(group)
        index = torch.Tensor(index)
        finest_flag = torch.Tensor(finest_flag)
        feats=np.ones_like(xyz_th[:,:1]).astype(np.float32)

        central_distance = torch.Tensor([0])

        feats_cmpl = []
        for xyz_tmp_th in xyz_cmpl_th:
            feats_cmpl.append(np.ones_like(xyz_tmp_th[:,:1]).astype(np.float32))

        # CBGF style data augmentation: keep the central PC and nghb PC aligned, at all times
        use_cbgf_augmentation = True
        if (self.data_augmentation and use_cbgf_augmentation):
            # rotate the point cloud
            T0 = sample_random_trans(xyz_th, np.pi / 4)

            xyz_th = self.apply_transform(xyz_th, T0)
            for i, xyz_tmp in enumerate(xyz_cmpl_th):
                Tc = follow_presampled_trans(xyz_tmp, T0)
                xyz_cmpl_th[i] = self.apply_transform(xyz_tmp, Tc)
                list_M[i] = T0 @ list_M[i] @ np.linalg.inv(Tc)
            
            # scale the pcd
            if random.random() < 0.95:
                scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
                matching_search_voxel_size *= scale
                xyz_th = scale * xyz_th
                for i, xyz_tmp in enumerate(xyz_cmpl_th):
                    xyz_cmpl_th[i] = scale * xyz_tmp
                    list_M[i][:3, 3] = scale * list_M[i][:3, 3]

        return [xyz_th] + xyz_cmpl_th, [feats] + feats_cmpl, \
               group.int(), index.long(), finest_flag.bool(), central_distance.float(), \
               torch.Tensor(list_M).float(), torch.ones(1)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, dirname, indices=None, ext='.txt', return_all=False):
        data_path = os.path.join(self.root, 'sequences', dirname, 'poses.npy')
        if data_path not in self.nuscenes_cache:
            self.nuscenes_cache[data_path] = np.load(data_path)
        if return_all:
            return self.nuscenes_cache[data_path]
        else:
            return self.nuscenes_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def make_open3d_point_cloud(self, xyz, color=None):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = open3d.utility.Vector3dVector(color)
        return pcd