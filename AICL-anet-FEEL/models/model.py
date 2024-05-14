import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math
import numpy as np
import random
from decimal import Decimal

from utils import misc_utils

torch.set_printoptions(profile="full")

class BaseModel(nn.Module):
    def __init__(self, len_feature, num_classes, config=None):
        super(BaseModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_rgb = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)  # 0.5


    def forward(self, x):
        input = x.permute(0, 2, 1)


        emb_flow = self.action_module_flow(input[:, 1024:, :])
        emb_rgb = self.action_module_rgb(input[:, :1024, :])

        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = emb_rgb.permute(0, 2, 1)

        action_flow = torch.sigmoid(self.cls_flow(emb_flow))
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb))

        emb = self.base_module(input)
        embedding = emb.permute(0, 2, 1)
        # emb = self.dropout(emb)
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = cas.sum(dim=2)
        # print(actionness1)
        actionness1 = torch.sigmoid(actionness1)

        actionness2 = (action_flow + action_rgb)/2
        actionness2 = actionness2.squeeze(1)  # 直接动作-背景分类和动作内部分类

        return cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb


class AICL(nn.Module):
    def __init__(self, cfg):
        super(AICL, self).__init__()
        self.len_feature = 2048
        self.num_classes = 100

        self.actionness_module = BaseModel(self.len_feature, self.num_classes, cfg)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_C = 20 # 10
        self.r_I = 20 # 8

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        selected_scores = torch.gather(scores, 1, idx_topk[:, :, 0])
        # print(selected_scores.shape)
        return selected_embeddings, selected_scores

    def select_all_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        selected_scores = torch.gather(scores, 1, idx_topk[:, :, 0])
        selected_scores = torch.zeros_like(scores)

        # 使用 idx_topk 将选中的位置赋值为原始 scores 的值
        # print(idx_topk[:, :, 0].shape)
        # print(scores.gather(1, idx_topk[:, :, 0]).shape)
        selected_scores = selected_scores.scatter_(1, idx_topk[:, :, 0], scores.gather(1, idx_topk[:, :, 0]))
        # print(selected_embeddings.shape)  # 1, 2, 512
        # selected_scores = scores
        # print(selected_scores)
        return selected_embeddings, selected_scores

    def consistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_easy, _data=None ,Cluster=False):

        x = aness_bin1 + aness_bin2
        # print(aness_bin1)  # 激活不出值
        select_idx_act = actionness1.new_tensor(np.where(x == 2, 1, 0))  # actionness1是动作类别相关加出来的
        # print(torch.min(torch.sum(select_idx_act, dim=-1)))

        # print(actionness1)
        actionness_act = actionness1 * select_idx_act  # 预测相同且高得分
        # print(actionness_act)

        select_idx_bg = actionness1.new_tensor(np.where(x == 0, 1, 0))

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        actionness_bg = actionness_rev * select_idx_bg  # 预测相同且低得分

        easy_act, easy_score = self.select_topk_embeddings(actionness_act, embeddings, k_easy)
        if Cluster:
            easy_act_embeddings, easy_score_score = self.select_all_embeddings(actionness_act, embeddings, k_easy)
            # print(easy_act_embeddings.shape)
            # print(easy_score_score.shape)
            easy_act_embeddings = _data.cpu().data.numpy()# easy_act_embeddings.cpu().data.numpy()  # _data.cpu().data.numpy()
            easy_score_score = easy_score_score.cpu().data.numpy()
            # easy_score_score = actionness1.cpu().data.numpy()
            # print(easy_act_embeddings.shape)
            # print(easy_score_score.shape)
            # weighted_features = np.sum(easy_act_embeddings * easy_score_score[:, :, np.newaxis], axis=1)

            # weighted_features = np.sum(easy_act_embeddings * actionness1[:, :, np.newaxis], axis=1)
            # print(weighted_features.shape)
            easy_score = easy_score_score
        else:
            easy_score = 0
        easy_bkg, _ = self.select_topk_embeddings(actionness_bg, embeddings, k_easy)

        return easy_act, easy_bkg, easy_score, select_idx_act.cpu().data.numpy()

    def Inconsistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_hard):

        x = aness_bin1 + aness_bin2
        idx_region_inner = actionness1.new_tensor(np.where(x == 1, 1, 0))
        aness_region_inner = actionness1 * idx_region_inner
        hard_act, _ = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        aness_region_outer = actionness_rev * idx_region_inner
        hard_bkg, _ = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_C = num_segments // self.r_C
        k_I = num_segments // self.r_I

        cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb = self.actionness_module(x)
        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.permute(0, 2, 1).detach(),
                                                              action_rgb.permute(0, 2, 1))
        combined_cas = (actionness1 + actionness2) / 2 #combined_cas.sum(dim=2)
        easy_score = combined_cas.cpu().detach().numpy()
        # print(actionness1.shape)
        # print(actionness2.shape)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        # print(aness_median1[0][0])
        if aness_median1[0][0] == 1.0:
            # print('!!!!!!!!!!')
            aness_bin1 = np.where(aness_np1 >= aness_median1, 1.0, 0.0)
            # aness_median1[0][0] = value
            # print(aness_median1)
        else:
            aness_bin1 = np.where(aness_np1 >= aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 >= aness_median2, 1.0, 0.0)
        # print(actionness2)  # 没问题
        # actionness = actionness1 + actionness2
        # combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
        #                                                       action_flow.permute(0, 2, 1).detach(),
        #                                                       action_rgb.permute(0, 2, 1))
        # combined_cas = combined_cas.sum(dim=2)
        # print(actionness1.shape)
        _, _, easy_score, consistency_idx = self.consistency_snippets_mining1(aness_bin1, aness_bin2, combined_cas, embedding, k_C, x, True)
        CA, CB, _, _ = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        # _, _, easy_score1 = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C, x, True)
        # print(easy_score.shape)
        CAr, CBr, _, _ = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        # _, _, easy_score2 = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C, x, True)
        CAf, CBf, _, _ = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        # easy_score = np.concatenate((easy_score1, easy_score2), axis=1)
        # print(easy_score.shape)
        contrast_pairs = {
            'CA': CA,
            'CB': CB,
            'IA': IA,
            'IB': IB,
            'score': easy_score,
            'idx': consistency_idx
        }

        contrast_pairs_r = {
            'CA': CAr,
            'CB': CBr,
            'IA': IAr,
            'IB': IBr
        }

        contrast_pairs_f = {
            'CA': CAf,
            'CB': CBf,
            'IA': IAf,
            'IB': IBf
        }

        # print(cas.shape)
        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, \
               contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2