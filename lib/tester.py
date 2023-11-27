from lib.trainer import Trainer
import os, torch
from tqdm import tqdm
import numpy as np
from lib.benchmark_utils import ransac_pose_estimation, random_sample, get_angle_deviation, to_o3d_pcd, to_array, Timer
import open3d as o3d

# Modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler
from common.misc import prepare_logger
from collections import defaultdict


class KITTITester(Trainer):
    """
    KITTI tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
        if "rot_threshold" in [k for (k, v) in args.items()] and "trans_threshold" in [k for (k, v) in args.items()]:
            self.rot_threshold = args.rot_threshold
            self.trans_threshold = args.trans_threshold
        else:
            print('No rot & trans upper bound designated. Using default (5 degrees and 2 meters).')
            self.rot_threshold = 5
            self.trans_threshold = 2
        print(f"rot_threshold: {self.rot_threshold}, trans_threshold: {self.trans_threshold}")
    
    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()

        data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()
        
        self.model.eval()
        rot_gt, trans_gt =[],[]
        with torch.no_grad():
            for i in tqdm(range(num_iter)): # loop through this epoch
                data_timer.tic()
                inputs = c_loader_iter.next()
                data_timer.toc()
                ###############################################
                # forward pass
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                # torch.cuda.synchronize()
                feat_timer.tic()
                try:
                    feats = self.model(inputs)  #[N1, C1], [N2, C2]
                    # torch.cuda.synchronize()
                    feat_timer.toc()
                except:
                    feat_timer.toc()
                    continue

                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']

                n_points = 5000
                #######################################
                # run random sampling or probabilistic sampling
                src_pcd, src_feats = random_sample(src_pcd, src_feats, n_points)
                tgt_pcd, tgt_feats = random_sample(tgt_pcd, tgt_feats, n_points)

                ########################################
                # run ransac 
                reg_timer.tic()
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 4)
                tsfm_est.append(ts_est)
                reg_timer.toc()

        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:,:3,:3]
        trans_est = tsfm_est[:,:3,3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:,:,0]

        print(f"rot & trans threshold: {self.rot_threshold}, {self.trans_threshold}")
        print(f"Data time: {data_timer.avg}, Feat time: {feat_timer.avg}, Reg time: {reg_timer.avg}")

        np.savez(f'{self.snapshot_dir}/results',rot_est=rot_est, rot_gt=rot_gt, trans_est = trans_est, trans_gt = trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

        flag_1=r_deviation<self.rot_threshold
        flag_2=translation_errors<self.trans_threshold
        correct=(flag_1 & flag_2).sum()
        precision=correct/rot_gt.shape[0]

        message=f'\n Registration recall: {precision:.3f}\n'

        # used for testing
        success_inds = flag_1 & flag_2
        dists = np.linalg.norm(trans_gt,axis=-1)
        np.save(f"{self.snapshot_dir}/success_dists.npy", dists[np.where(success_inds > 0)])
        np.save(f"{self.snapshot_dir}/fail_dists.npy", dists[np.where(success_inds < 1)])

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors=dict()
        errors['rot_mean']=round(np.mean(r_deviation),3)
        errors['rot_median']=round(np.median(r_deviation),3)
        errors['trans_rmse'] = round(np.mean(translation_errors),3)
        errors['trans_rmedse']=round(np.median(translation_errors),3)
        errors['rot_std'] = round(np.std(r_deviation),3)
        errors['trans_std']= round(np.std(translation_errors),3)

        message+=str(errors)
        print(message)
        self.logger.write(message+'\n')

class NUSCENESTester(Trainer):
    """
    KITTI tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
        if "rot_threshold" in [k for (k, v) in args.items()] and "trans_threshold" in [k for (k, v) in args.items()]:
            self.rot_threshold = args.rot_threshold
            self.trans_threshold = args.trans_threshold
        else:
            print('No rot & trans upper bound designated. Using default (5 degrees and 2 meters).')
            self.rot_threshold = 5
            self.trans_threshold = 2
        print(f"rot_threshold: {self.rot_threshold}, trans_threshold: {self.trans_threshold}")
    
    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        
        self.model.eval()
        rot_gt, trans_gt =[],[]
        with torch.no_grad():
            for i in tqdm(range(num_iter)): # loop through this epoch
                # prepare_timer, feat_timer = Timer(), Timer()
                # prepare_timer.tic()
                inputs = c_loader_iter.next()
                # prepare_timer.toc()
                ###############################################
                # forward pass
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)
                try:
                    feats = self.model(inputs)  #[N1, C1], [N2, C2]
                except:
                    continue

                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']

                ########################################
                # run ransac 
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=True, distance_threshold=distance_threshold, ransac_n = 4)
                tsfm_est.append(ts_est)
        
        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:,:3,:3]
        trans_est = tsfm_est[:,:3,3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:,:,0]

        np.savez(f'{self.snapshot_dir}/results',rot_est=rot_est, rot_gt=rot_gt, trans_est = trans_est, trans_gt = trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

        print(f"rot & trans threshold: {self.rot_threshold}, {self.trans_threshold}")

        flag_1=r_deviation<self.rot_threshold
        flag_2=translation_errors<self.trans_threshold
        correct=(flag_1 & flag_2).sum()
        precision=correct/rot_gt.shape[0]

        message=f'\n Registration recall: {precision:.3f}\n'

        # used for testing
        success_inds = flag_1 & flag_2
        dists = np.linalg.norm(trans_gt,axis=-1)
        np.save(f"{self.snapshot_dir}/success_dists.npy", dists[np.where(success_inds > 0)])
        np.save(f"{self.snapshot_dir}/fail_dists.npy", dists[np.where(success_inds < 1)])

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors=dict()
        errors['rot_mean']=round(np.mean(r_deviation),3)
        errors['rot_median']=round(np.median(r_deviation),3)
        errors['trans_rmse'] = round(np.mean(translation_errors),3)
        errors['trans_rmedse']=round(np.median(translation_errors),3)
        errors['rot_std'] = round(np.std(r_deviation),3)
        errors['trans_std']= round(np.std(translation_errors),3)

        message+=str(errors)
        print(message)
        self.logger.write(message+'\n')

def compute_rigid_transform(a, b, weights):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    tsfm = torch.eye(4)
    tsfm[:3] = transform
    return tsfm.numpy()


def compute_metrics(data , pred_transforms):
    """
    Compute metrics required in the paper
    """
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_array(t_mse),
            't_mae': to_array(t_mae),
            'err_r_deg': to_array(residual_rotdeg),
            'err_t': to_array(residual_transmag),
            'chamfer_dist': to_array(chamfer_dist)
        }

    return metrics

def print_metrics(logger, summary_metrics , losses_by_iteration=None,title='Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))

def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized
       
        

def get_trainer(config):
    if(config.dataset in ['kitti', 'kitti_colocation']):
        return KITTITester(config)
    elif(config.dataset in ['nuscenes', 'nuscenes_colocation']):
        return NUSCENESTester(config)
    else:
        raise NotImplementedError
