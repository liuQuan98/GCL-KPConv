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


class KITTIDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': './configs/kitti/train_kitti.txt',
        'val': './configs/kitti/val_kitti.txt',
        'test': './configs/kitti/test_kitti.txt'
    }
    discard_pairs =[(5, 1151, 1220), (2, 926, 962), (2, 2022, 2054), \
                    (1, 250, 266), (0, 3576, 3609), (2, 2943, 2979), \
                    (1, 411, 423), (2, 2241, 2271), (0, 1536, 1607), \
                    (0, 1338, 1439), (7, 784, 810), (2, 1471, 1498), \
                    (2, 3829, 3862), (0, 1780, 1840), (2, 3294, 3356), \
                    (2, 2420, 2453), (2, 4146, 4206), (0, 2781, 2829), \
                    (0, 3351, 3451), (1, 428, 444), (0, 3073, 3147)]

    def __init__(self,config,split,data_augmentation=True):
        super(KITTIDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root,'dataset')
        # self.icp_path = os.path.join(config.root,'icp')
        # if not os.path.exists(self.icp_path):
        #     os.makedirs(self.icp_path)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.max_corr = config.max_points
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min

        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        self.complement_pair_dist = config.complement_pair_dist
        self.num_complement_one_side = config.num_complement_one_side
        # self.point_generation_ratio = config.point_generation_ratio
        self.complement_range = self.num_complement_one_side * self.complement_pair_dist
        
        # pose configuration: use old or new
        try:
            self.use_old_pose = config.use_old_pose
        except:
            self.use_old_pose = True

        if self.use_old_pose:
            self.icp_path = os.path.join(config.root,'icp')
        else:
            self.icp_path = os.path.join(config.root,'icp_slam')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)

        # data loading interval: old (no overlap), or new (constant distance between t0s)
        try:
            self.old_dataloader = config.old_dataloader
        except:
            self.old_dataloader = True

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        
        # load LoKITTI point cloud pairs, instead of generating them based on distance
        if split == 'test' and config.LoKITTI == True:
            self.files = np.load("configs/kitti/file_LoKITTI_50.npy")
        else:
            self.prepare_kitti_ply_rcar(split)

        self.split = split


    def prepare_kitti_ply(self, split):
        assert split in ['train','val','test']

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split=='test':
            self.files.remove((8, 15, 58))
        print(f'Num_{split}: {len(self.files)}')


    def prepare_kitti_ply_rcar(self, split='train'):
        import pathlib
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        print(f"Loading the subset {split} from {self.root}")

        # hard-coded parameters for comparison
        # self.min_sample_frame_dist = 10.0
        # self.complement_pair_dist = 10.0
        # self.num_complement_one_side = 4
        all_length = 0

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            if self.use_old_pose:
                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            else:
                all_pos = self.get_slam_odometry(drive_id, return_all=True)

            self.Ts = all_pos[:, :3, 3]
            all_length += len(self.Ts)

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
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                    next_time = next_time[0] + curr_time - 1
                    skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                    skip_1, cmpl_1 = self._get_complement_frames(next_time)
                    skip_2 = (drive_id, curr_time, next_time) in self.discard_pairs
                    if skip_0 or skip_1 or (skip_2 and self.use_old_pose):
                        curr_time += 1
                    else:
                        self.files.append((drive_id, curr_time, next_time))
                        if self.old_dataloader:
                            curr_time = next_time + 1
                        else:
                            curr_time += 11

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


    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        if self.use_old_pose:
            all_odometry = self.get_video_odometry(drive, [t0, t1])
            positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        else:
            positions = self.get_slam_odometry(drive, [t0, t1])
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                if self.use_old_pose:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                    xyz0_t = self.apply_transform(xyz0, M)
                    pcd0 = to_o3d_pcd(xyz0_t)
                    pcd1 = to_o3d_pcd(xyz1)
                    reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                            open3d.registration.TransformationEstimationPointToPoint(),
                                                            open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                    pcd0.transform(reg.transformation)
                    M2 = M @ reg.transformation
                else:
                    M2 = np.linalg.inv(positions[1]) @ positions[0]
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]


        # refined pose is denoted as trans
        tsfm = M2
        rot = tsfm[:3,:3]
        trans = tsfm[:3,3][:,None]

        # voxelize the point clouds here
        pcd0 = to_o3d_pcd(xyz0)
        pcd1 = to_o3d_pcd(xyz1)
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        src_pcd = np.array(pcd0.points)
        tgt_pcd = np.array(pcd1.points)

        # # if idx == 22:
        # print(f"distance: {np.linalg.norm(trans)}")
        # print(f"idx: {idx}, drive:{drive}, t0:{t0}, t1:{t1}")
        # np.save('pcd0.npy', np.asarray(xyz0))
        # np.save('pcd1.npy', np.asarray(xyz1))
        # np.save('trans.npy', np.asarray(M2))
        # print("pcd saved!!!!")
        # raise ValueError

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

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib
    
    def get_slam_odometry(self, drive, indices=None, return_all=False):
        data_path = self.root + '/sequences/%02d' % drive
        calib_filename = data_path + '/calib.txt'
        pose_filename = data_path + '/poses.txt'
        calibration = self.parse_calibration(calib_filename)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        pose_file = open(pose_filename)
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        
        if pose_filename not in self.kitti_cache:
            self.kitti_cache[pose_filename] = np.array(poses)
        if return_all:
            return self.kitti_cache[pose_filename]
        else:
            return self.kitti_cache[pose_filename][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
