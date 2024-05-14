import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt           # 46.18% TPAMI
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict
from models.model import AICL
from anet_clustering import cluster, get_cluster_performance, load_json, get_train_label, dump_soft_cluster_2_action
from progressive_reranking import progressive_learning

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

np.set_printoptions(threshold=np.inf)

# def load_weight(net, config):
#     print(config.load_weight)
#     if config.load_weight:
#         model_file = os.path.join(config.model_path, "CAS_Only.pkl")
#         print("loading from file for training: ", model_file)
#         pretrained_params = torch.load(model_file)
#
#         selected_params = OrderedDict()
#         for k, v in pretrained_params.items():
#             if 'base_module' in k:
#                 selected_params[k] = v
#
#         model_dict = net.state_dict()
#         model_dict.update(selected_params)
#         net.load_state_dict(model_dict)
def load_weight(net, config):
    model_file = os.path.join(config.model_path, "CAS_Only.pkl")
    if config.load_weight:
        print("loading from file: ", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)


def get_dataloaders(config):
    if config.inference_only == False:
        data_anno_train_path = os.path.join(os.path.dirname(config.data_anno_path), str(int(config.iter) - 1))
        train_loader = data.DataLoader(
            ThumosFeature(data_path=config.data_path, mode='train',
                          modal=config.modal, feature_fps=config.feature_fps,
                          num_segments=config.num_segments, len_feature=config.len_feature,
                          seed=config.seed, sampling='random', supervision='weak', data_anno_path=data_anno_train_path),
            batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='weak', data_anno_path=config.data_path),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    cluster_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='weak', data_anno_path=config.data_path),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    if config.inference_only == False:
        return data_anno_train_path, train_loader, test_loader, cluster_loader
    else:
        return None, None, test_loader, cluster_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        self.net = AICL(config)
        self.net = self.net.cuda()

        # data
        self.data_anno_train_path, self.train_loader, self.test_loader, self.cluster_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0


    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = inference(self.net, self.config, self.test_loader, self.data_anno_train_path,
                                           model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))

    def extractAndCluster(self):
        anet_label_path = "/root/tanghaoyu/AICL/AICL-anet/ActivityNet12/gt.json"
        anet_label = load_json(anet_label_path)
        anet_label = anet_label['database']
        training_index, action_2_video, video_2_action = get_train_label(anet_label)
        self.net.eval()

        data_list = []  # 存储_data的列表
        actionness1_list = []  # 存储转换后的actionness1的列表
        subset_index = []
        easy = []

        if int(self.config.iter) == 0:
            for _data, _label, temp_anno, vid_name, vid_num_seg in self.cluster_loader:
                _data = _data.cpu().data.numpy()
                _data = np.squeeze(_data, axis=(0,))
                weighted_features = np.mean(_data, axis=0)
                # print(weighted_features.shape)
                weighted_features = weighted_features.reshape(1,2048)
                norm = np.linalg.norm(weighted_features, axis=1, keepdims=True)
                normalized_features = weighted_features / norm
                easy.append(normalized_features)
                subset_index.extend(vid_name)
            normalized_features = np.concatenate(easy, axis=0)
            # print('!')
            label_pred = cluster(100, normalized_features)
            # print('!!')
            cluser_2_action, soft_cluster_2_action = get_cluster_performance(100, label_pred, subset_index,
                                                                             action_2_video,
                                                                             video_2_action, path=self.config.data_anno_path,
                                                                             dump=False)
            # dump_soft_cluster_2_action(soft_cluster_2_action, self.config.data_anno_path)
            progressive_learning(subset_index, normalized_features, label_pred,
                                 top_rate=float(self.config.enlarge_rate), #(int(self.config.iter) + 1) * float(self.config.enlarge_rate),
                                 path=self.config.data_anno_path)

        if int(self.config.iter) != 0:
            with torch.no_grad():
                model_filename = "CAS_Only.pkl"
                self.config.model_file = os.path.join(self.config.model_path, model_filename)

                # load weights
                load_weight(self.net, self.config)

                for _data, _label, temp_anno, vid_name, vid_num_seg in self.cluster_loader:
                    _data = _data.cuda()
                    _label = _label.cuda()
                    # FORWARD PASS
                    cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.net(
                        _data)  # cas:[1,750,100]  action_rgb:[1,1,750]
                    an1 = actionness1.cpu().detach().numpy()
                    an2 = actionness2.cpu().detach().numpy()
                    _data = _data.cpu().data.numpy()
                    # weighted_features = np.sum(_data * vote_consistency[i:i+1, :, np.newaxis], axis=1)
                    weighted_features = np.sum(_data * contrast_pairs['score'][:, :, np.newaxis], axis=1)
                    norm = np.linalg.norm(weighted_features, axis=1, keepdims=True)
                    normalized_features = weighted_features / norm
                    easy.append(normalized_features)
                    subset_index.extend(vid_name)
            normalized_features = np.concatenate(easy, axis=0)
            label_pred = cluster(100, normalized_features)  # print(normalized_features.shape)

            cluser_2_action, soft_cluster_2_action = get_cluster_performance(100, label_pred, subset_index,
                                                                             action_2_video,
                                                                             video_2_action, path=self.config.data_anno_path,
                                                                             dump=False)
            # dump_soft_cluster_2_action(soft_cluster_2_action, self.config.data_anno_path)
            progressive_learning(subset_index, normalized_features, label_pred,
                                 top_rate=float(self.config.enlarge_rate), #(int(self.config.iter) + 1) * float(self.config.enlarge_rate),
                                 path=self.config.data_anno_path)


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments


    def calculate_all_losses1(self, contrast_pairs, contrast_pairs_r,contrast_pairs_f, cas_top, label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2):
        self.contrastive_criterion = ContrastiveLoss()
        loss_contrastive = self.contrastive_criterion(contrast_pairs) + self.contrastive_criterion(contrast_pairs_r) + self.contrastive_criterion(contrast_pairs_f)

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.5 * F.mse_loss(actionness1, actionness2) + 0.5 * F.mse_loss(actionness2, actionness1)

        cost = base_loss + class_agnostic_loss + 5 * modality_consistent_loss + 0.01 * loss_contrastive +\
               0.1 * action_consistent_loss  # + 0.0001 * loss_um

        return cost

    def evaluate(self, epoch=0):
        # if self.step % self.config.detection_inf_step == 0:
        #     self.total_loss_per_epoch /= self.config.detection_inf_step
        if epoch != 0 and epoch % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = inference(self.net, self.config, self.test_loader, self.data_anno_train_path,
                                              model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))

            # print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  now_map={:5.2f}  best_map={:5.2f}".format(
            #         epoch, self.step, self.total_loss_per_epoch, test_acc * 100, mean_ap * 100, self.best_mAP * 100))
            print("epoch={:5d}  epoch={:5d}  Loss={:.4f}  cls_acc={:5.2f}  now_map={:5.2f}  best_map={:5.2f}".format(
                epoch, epoch, self.total_loss_per_epoch, test_acc * 100, mean_ap * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0


    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.permute(0, 2, 1).detach(),
                                                              action_rgb.permute(0, 2, 1))


        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 2, dim=1)
        # _, topk_indices1 = torch.topk(combined_cas, r, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):

            for _data, _label, temp_anno, _, _ in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                # forward pass
                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.forward_pass(_data)

                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # losses
                cost = self.calculate_all_losses1(contrast_pairs, contrast_pairs_r,contrast_pairs_f, cas_top, _label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
            self.evaluate(epoch=epoch)



def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)
    trainer = ThumosTrainer(config)

    if args.inference_only:
        # trainer.test()
        print("!!!!")
        trainer.extractAndCluster()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
