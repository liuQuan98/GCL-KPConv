# Basic libs
import os, time, glob, random, pickle, copy, torch
import numpy as np
import open3d
import pathlib
import MinkowskiEngine as ME
from scipy.spatial.transform import Rotation

# Dataset parent class
from torch.utils.data import Dataset
from lib.timer import Timer
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_correspondences
from scripts.cal_overlap import get_overlap_ratio


class NUSCENESDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    icp_voxel_size = 0.05 # 0.05 meters, i.e. 5cm

    def __init__(self,config,split,data_augmentation=True):
        super(NUSCENESDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root, split)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.max_corr = config.max_points
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min
        self.max_correspondence_distance_fine = self.icp_voxel_size * 1.5
        self.load_neighbourhood = True
        if config.mode == 'test':
            self.load_neighbourhood = False

        from lib.utils import Logger
        self.logger = Logger(config.snapshot_dir)

        # rcar data config
        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        # self.min_sample_frame_dist = config.min_sample_frame_dist
        self.complement_pair_dist = config.complement_pair_dist
        self.num_complement_one_side = config.num_complement_one_side
        self.complement_range = self.num_complement_one_side * self.complement_pair_dist

        # pose configuration: use old or new
        assert config.use_old_pose is True, "no slam-based position available!"

        self.icp_path = os.path.join(config.root,'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        # Initiate containers
        self.files = []
        self.nuscenes_icp_cache = {}
        self.nuscenes_cache = {}
        self.split = split
        # load LoNuscenes point cloud pairs, instead of generating them based on distance
        # print(split == 'test', flush=True)
        if split == 'test' and config.LoNUSCENES == True:
            self.files = np.load("configs/nuscenes/file_LoNUSCENES_50.npy", allow_pickle=True)
        else:
            self.prepare_nuscenes_ply(split)
        if split == 'train':
            self.files = self.files[::3]
            self.files = self.files[:1200]
        print(self.__len__())

    def prepare_nuscenes_ply(self, split='train'):
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        print(f"Loading the subset {split} from {self.root}")
        
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))

        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[min(int(self.complement_range * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)
                
                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.complement_range)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/nuscenes/process_nuscenes_data.m#L44
                    next_time = next_time[0] + curr_time - 1
                    skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                    skip_1, cmpl_1 = self._get_complement_frames(next_time)
                    if skip_0 or skip_1:
                        curr_time += 1
                    else:
                        self.files.append((dirname, curr_time, next_time, cmpl_0, cmpl_1))
                        curr_time = next_time + 1

    def _get_complement_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.complement_range))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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
        right_dist = (self.Ts[frame: frame+int(10*self.complement_range)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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


    # simple function for getting the xyz point-cloud w.r.t log and time
    def _get_xyz(self, dirname, time):
        fname = self._get_velodyne_fn(dirname, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]


    # registers source onto target (used by multi-way registration)
    def pairwise_registration(self, source, target, pos_source, pos_target):
        # -----------The following code piece is copied from open3d official documentation
        M = (self.velo2cam @ pos_source.T @ np.linalg.inv(pos_target.T)
             @ np.linalg.inv(self.velo2cam)).T
        icp_fine = open3d.registration.registration_icp(
            source, target, 0.2, M,
            open3d.registration.TransformationEstimationPointToPoint(),
            open3d.registration.ICPConvergenceCriteria(max_iteration=200))
        transformation_icp = icp_fine.transformation
        information_icp = open3d.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    def __getitem__(self, idx):
        prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
        prepare_timer.tic()
        try:
            dirname, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
        except:
            dirname, t_0, t_1 = self.files[idx]
        # print(self.files[idx])
        positions = self.get_video_odometry(dirname, [t_0, t_1])

        pos_0, pos_1 = positions[0:2]

        # load two center point clouds
        xyz_0 = self._get_xyz(dirname, t_0)
        xyz_1 = self._get_xyz(dirname, t_1)
        prepare_timer.toc()

        icp_timer.tic()
        M2 = np.linalg.inv(positions[1]) @ positions[0]
        icp_timer.toc()
            
        rot_crop_timer.tic()
        # refined pose is denoted as trans
        tsfm = M2
        rot = tsfm[:3,:3]
        trans = tsfm[:3,3][:,None]

        # voxelize the point clouds here
        pcd0 = self.make_open3d_point_cloud(xyz_0)
        pcd1 = self.make_open3d_point_cloud(xyz_1)

        pcd0 = pcd0.voxel_down_sample(voxel_size=self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(voxel_size=self.voxel_size)
        src_pcd = np.array(pcd0.points)
        tgt_pcd = np.array(pcd1.points)

        # Get matches
        matching_inds = get_correspondences(pcd0, pcd1, tsfm, self.matching_search_voxel_size)
        if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
            return self.__getitem__(np.random.choice(len(self.files),1)[0])

        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        # add data augmentation
        src_pcd_input = copy.deepcopy(src_pcd)
        tgt_pcd_input = copy.deepcopy(tgt_pcd)
        if(self.data_augmentation):
            # add gaussian noise
            src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
            else:
                tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
            
            # scale the pcd
            scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            src_pcd_input = src_pcd_input * scale
            tgt_pcd_input = tgt_pcd_input * scale

            # shift the pcd
            shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

            src_pcd_input = src_pcd_input + shift_src
            tgt_pcd_input = tgt_pcd_input + shift_tgt
        rot_crop_timer.toc()
        # message = f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, r&c: {rot_crop_timer.avg}, total: {prepare_timer.avg+icp_timer.avg+rot_crop_timer.avg}"
        # # print(message)
        # self.logger.write(message + '\n')

        return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, torch.ones(1)


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