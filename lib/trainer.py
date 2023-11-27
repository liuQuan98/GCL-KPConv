import time, os, torch,copy
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient,square_distance,_neg_hash

from tqdm import tqdm
import torch.nn.functional as F
import gc


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')

        self.pos_thresh = args.pos_margin
        self.neg_thresh = args.neg_margin
        self.finest_thresh = args.finest_margin
        self.safe_radius = args.safe_radius

        if args.use_group_circle_loss:
            self.group_loss = self.location_circle_loss
        else:
            self.group_loss = self.location_contrastive_loss

        try:
            self.use_pair_group_positive_loss = args.use_pair_group_positive_loss
        except:
            self.use_pair_group_positive_loss = True

        self.use_mean_distance_err = False
        self.block_finest_gradient = args.block_finest_gradient
        self.pos_weight = args.pos_weight
        self.neg_weight = args.neg_weight
        self.finest_weight = args.finest_weight
        self.log_scale = 16

        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain, config=args)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume, config):
        if os.path.isfile(resume):
            state = torch.load(resume)
            if 'pretrain_restart' in [k for (k, v) in config.items()] and config.pretrain_restart:
                self.logger.write(f'Restart pretrain, only loading model weights.\n')
            else:
                self.start_epoch = state['epoch']
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_loss = state['best_loss']
                self.best_recall = state['best_recall']
            self.model.load_state_dict(state['state_dict'])
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.  # feature match recall, divided by number of ground truth pairs
        # stats['saliency_loss'] = 0.
        # stats['saliency_recall'] = 0.
        # stats['saliency_precision'] = 0.
        # stats['overlap_loss'] = 0.
        # stats['overlap_recall']=0.
        # stats['overlap_precision']=0.
        stats['loss']=0.
        stats['pos_loss']=0.
        stats['neg_loss']=0.
        stats['finest_loss']=0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters


    def location_contrastive_loss(self,
                                  F_out,
                                  central_pcd,
                                  group,
                                  index,
                                  index_hash,
                                  finest_flag,
                                  max_pos_cluster=256,
                                  max_hn_samples=2048):
        """
        Calculates the finest, positive, and negative losses of input co-location groups.
        Works only with proper data input from colocation_data_loader.
        """
        group, index = group.to(self.device), index.to(self.device)
        N_out = len(F_out)
        hash_seed = N_out
        split_timer, pos_timer, dist_timer, hash2_timer, neg_timer = Timer(), Timer(), Timer(), Timer(), Timer()

        # positive loss and finest loss
        split_timer.tic()
        pos_loss, finest_loss = 0, 0
        N_pos_clusters = len(group)
        index_split = torch.split(index, tuple(group.tolist()))
        finest_flag_split = torch.split(finest_flag, tuple(group.tolist()))
        if N_pos_clusters > max_pos_cluster:
            pos_sel = np.random.choice(N_pos_clusters, max_pos_cluster, replace=False)
        else:
            pos_sel = np.arange(N_pos_clusters)
        split_timer.toc()

        pos_timer.tic()
        for i in pos_sel:
            index_set, finest_flag_set = index_split[i], finest_flag_split[i]
            feature_set = F_out[index_set]
            if self.use_pair_group_positive_loss:
                pos_positions = np.random.choice(len(feature_set), 2, replace=False)
                pos_loss += F.relu((feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1) - self.pos_thresh)
            else:
                pos_loss += F.relu(torch.mean((torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1)) - self.pos_thresh)
            # whether we should block the gradient at the finest position during loss calculation
            if self.block_finest_gradient:
                blocked_feature_set = feature_set[torch.bitwise_not(finest_flag_set)]
                finest_loss += F.relu(torch.sqrt((torch.mean(blocked_feature_set, dim=0) - 
                    feature_set[finest_flag_set][0].detach()).pow(2).sum() + 1e-7) - self.finest_thresh)
            else:
                finest_loss += F.relu((torch.mean(feature_set, dim=0) - 
                    feature_set[finest_flag_set][0]).pow(2).sum() - self.finest_thresh)
        pos_loss, finest_loss = pos_loss/len(pos_sel), finest_loss/len(pos_sel)
        pos_timer.toc()

        # negative loss
        dist_timer.tic()
        # calculate between two bunches of separately downsampled features
        sel_hn1 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
        sel_hn2 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
        subF1 = F_out[sel_hn1]
        subF2 = F_out[sel_hn2]
        D_fs = square_distance(subF1[None,:,:], subF2[None,:,:])[0]
        D_fs_min, D_fs_ind = D_fs.min(1)
        D_fs_ind = D_fs_ind.cpu()
        dist_timer.toc()

        # mask the equal-indexed negative to prevent self comparison
        neg_timer.tic()
        mask_self = (sel_hn1 != sel_hn2[D_fs_ind])

        sel_hn2_closest = sel_hn2[D_fs_ind]
        hash2_timer.tic()
        pos_keys = index_hash
        neg_keys = _neg_hash(sel_hn1, sel_hn2_closest, hash_seed)
        hash2_timer.toc()

        mask = np.logical_not(np.isin(neg_keys, pos_keys, assume_unique=False))
        neg_loss = F.relu(self.neg_thresh - D_fs_min[mask & mask_self]).pow(2)
        neg_timer.toc()

        # print(f"split time: {split_timer.avg}, pos time: {pos_timer.avg}, dist time: {dist_timer.avg}, " + 
        #       f"hash_timer: {hash2_timer.avg}, neg time: {neg_timer.avg}")
        return pos_loss, finest_loss, neg_loss.mean()


    def location_circle_loss(self,
                             F_out,
                             central_pcd,
                             group,
                             index,
                             index_hash,
                             finest_flag,
                             max_pos_cluster=256,
                             max_hn_samples=None):
        """
        Calculates the finest and circle losses of input co-location groups.
        Works only with proper data input from colocation_data_loader.
        """
        group, index = group.to(self.device), index.to(self.device)
        split_timer, pos_timer, dist_timer, neg_timer = Timer(), Timer(), Timer(), Timer()

        # positive loss and finest loss
        split_timer.tic()
        pos_loss, finest_loss = 0, 0
        N_pos_clusters = len(group)
        index_split = torch.split(index, tuple(group.tolist()))
        finest_flag_split = torch.split(finest_flag, tuple(group.tolist()))
        if N_pos_clusters > max_pos_cluster:
            pos_sel = np.random.choice(N_pos_clusters, max_pos_cluster, replace=False)
        else:
            pos_sel = np.arange(N_pos_clusters)
        split_timer.toc()

        # create a place for storing positive features and positive coordinates
        coords_sel = torch.zeros((len(pos_sel), 3)).float().to(self.device)
        feats_sel  = torch.zeros((len(pos_sel), F_out.shape[1])).float().to(self.device)

        pos_timer.tic()
        for count, i in enumerate(pos_sel):
            index_set, finest_flag_set = index_split[i], finest_flag_split[i]
            coords_sel[count] = central_pcd[index_set[0]]
            feature_set = F_out[index_set]
            feats_sel[count] = torch.mean(feature_set, dim=0)

            # pos_loss: choose between pair loss or variance loss
            if self.use_pair_group_positive_loss:
                # notice that the circle loss of a single pair simply degrades to softplus(feature_distance)
                pos_positions = np.random.choice(len(feature_set), 2, replace=False)
                pos_loss += F.softplus(torch.sqrt((feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1)+1e-7) - self.pos_thresh)
            else:
                # pos_thresh is divided by 2, so that arbitrary two features will have at most pos_thresh distance
                var_dists = torch.sqrt((torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1) + 1e-7) - self.pos_thresh / 2
                pos_loss += F.softplus(torch.logsumexp(self.log_scale * var_dists * torch.max(torch.zeros_like(var_dists), var_dists).detach(), dim=-1)) / self.log_scale

            # finest_loss: whether we should block the gradient at the finest position during loss calculation
            if self.block_finest_gradient:
                blocked_feature_set = feature_set[~finest_flag_set]
                finest_dists = torch.sqrt((blocked_feature_set - feature_set[finest_flag_set][0].detach()).pow(2).sum(-1) + 1e-7) - self.finest_thresh
            else:
                finest_dists = torch.sqrt((feature_set - feature_set[finest_flag_set][0]).pow(2).sum(-1) + 1e-7) - self.finest_thresh
            finest_loss += F.softplus(torch.logsumexp(self.log_scale * finest_dists * torch.max(torch.zeros_like(finest_dists), finest_dists).detach(), dim=-1)) / self.log_scale

        pos_loss, finest_loss = pos_loss / len(pos_sel), finest_loss / len(pos_sel)
        pos_timer.toc()

        # negative loss
        dist_timer.tic()
        # get L2 coords & feature distance between groups
        coords_dist = torch.sqrt(square_distance(coords_sel[None,:,:], coords_sel[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(feats_sel[None,:,:], feats_sel[None,:,:],normalised=True)).squeeze(0)
        dist_timer.toc()

        # calculate negative circle loss
        neg_timer.tic()
        neg_mask = coords_dist > self.safe_radius 
        sel = (neg_mask.sum(-1)>0).detach() # Find anchors that have negative pairs. All anchors naturally have positive pairs.

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative (self-comparison is also removed here)
        neg_weight = (self.neg_thresh - neg_weight)         # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()
        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_thresh - feats_dist) * neg_weight,dim=-1)
        loss_row = F.softplus(lse_neg_row)/self.log_scale

        neg_loss = loss_row[sel].mean()
        neg_timer.toc()

        # print(f"split time: {split_timer.avg}, pos time: {pos_timer.avg}, dist time: {dist_timer.avg}, " + 
        #       f"hash_timer: {hash2_timer.avg}, neg time: {neg_timer.avg}")
        return pos_loss, finest_loss, neg_loss
    

    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass
            feats = self.model(inputs)

            ###################################################
            # get loss & stats
            # stats = self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, None, None)
            pos_loss, finest_loss, neg_loss = self.group_loss(
                feats,
                inputs['center_pcd'],
                inputs['group'],
                inputs['index'],
                inputs['index_hash'],
                inputs['finest_flag'],
                max_pos_cluster=256,
                max_hn_samples=512)
            
            loss = self.pos_weight * pos_loss + self.finest_weight * finest_loss + self.neg_weight * neg_loss
            loss.backward()
            stats = {}
            stats['pos_loss'] = float(pos_loss.detach())
            stats['neg_loss'] = float(neg_loss.detach())
            stats['finest_loss'] = float(finest_loss.detach())
            stats['loss'] = float(loss.detach())

            return stats

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                feats = self.model(inputs)  #[N1, C1], [N2, C2]
                # pcd =  inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                ###################################################
                # get loss
                stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, None, None)
                stats['circle_loss'] = float(stats['circle_loss'].detach())

        return stats


    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == dict or type(v) == np.ndarray:
                    pass
                else:
                    inputs[k] = v.to(self.device)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats = self.inference_one_batch(inputs, phase)
                
                ###################################################
                # run optimisation
                if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            except Exception as inst:
                print(inst)
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def train(self):
        print(self.model)
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            if stats_meter['recall'].avg > self.best_recall:
                self.best_recall = stats_meter['recall'].avg
                self._snapshot(epoch,'best_recall')
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)
