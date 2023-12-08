# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import core.utils as utils
from core.model import CoLA
from core.loss import TotalLoss
from core.config import cfg
from core.utils import AverageMeter
from core.dataset import NpyFeature
from torch.utils.tensorboard import SummaryWriter
from eval.eval_detection import ANETdetection
from terminaltables import AsciiTable
from anet_clustering import get_cluster_performance
from anet_clustering import cluster
from anet_clustering import get_train_label
from anet_clustering import load_json
from anet_clustering import dump_soft_cluster_2_action


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    worker_init_fn = None
    if cfg.SEED >= 0:
        utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    utils.set_path(cfg)
    utils.save_config(cfg)
    # print(cfg.MODEL_PATH)
    # print('!!')

    net = CoLA(cfg)
    net = net.cuda()

    if cfg.MODE == 'train':
        cfg.DATA_ANNO_TRAIN_PATH = os.path.join(os.path.dirname(cfg.DATA_ANNO_PATH), str(int(cfg.iter) - 1))
        train_loader = torch.utils.data.DataLoader(
            NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                       modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                       num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                       class_dict=cfg.CLASS_DICT, seed=cfg.SEED,
                       sampling='random', data_anno_path=cfg.DATA_ANNO_TRAIN_PATH),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    if int(cfg.iter) == 0:
        sampl = 'non'
    else:
        sampl = 'uniform'
    easy_act_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED,
                   sampling=sampl, data_anno_path=cfg.DATA_PATH),
        batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='test',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED,
                   sampling='uniform', data_anno_path=cfg.DATA_PATH),
        batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                 "mAP@0.7": []}
    # test_info = {"step": [], "test_acc": [], "average_mAP": [],
    #             "mAP@0.50": [], "mAP@0.55": [], "mAP@0.60": [],
    #             "mAP@0.65": [], "mAP@0.70": [], "mAP@0.75": [],
    #             "mAP@0.80": [], "mAP@0.85": [], "mAP@0.90": [], "mAP@0.95": []}

    best_mAP = -1

    criterion = TotalLoss()

    cfg.LR = eval(cfg.LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR[0],
                                 betas=(0.9, 0.999), weight_decay=0.0005)

    if cfg.MODE == 'test':
        get_easy_act(net, cfg, easy_act_loader, test_info, 0, None)
        # _, _ = test_all(net, cfg, test_loader, test_info, 0, None, cfg.MODEL_FILE)
        # utils.save_best_record_thumos(test_info,
        #     os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))
        # print(utils.table_format(test_info, cfg.TIOU_THRESH, '[CoLA] THUMOS\'14 Performance'))
        return
    else:
        writter = SummaryWriter(cfg.LOG_PATH)

    print('=> test frequency: {} steps'.format(cfg.TEST_FREQ))
    print('=> start training...')
    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR[step - 1] != cfg.LR[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        cost = train_one_step(net, loader_iter, optimizer, criterion, writter, step)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()
        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))

        if step > -1 and step % cfg.TEST_FREQ == 0:
            mAP_50, mAP_AVG = test_all(net, cfg, test_loader, test_info, step, writter)

            if test_info["average_mAP"][-1] > best_mAP:
                best_mAP = test_info["average_mAP"][-1]
                best_test_info = copy.deepcopy(test_info)

                utils.save_best_record_thumos(test_info,
                                              os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))

                torch.save(net.state_dict(), os.path.join(cfg.MODEL_PATH, \
                                                          "model_best.pth.tar"))

            print(('- Test result: \t' \
                   'mAP@0.5 {mAP_50:.2%}\t' \
                   'mAP@AVG {mAP_AVG:.2%} (best: {best_mAP:.2%})'.format(
                mAP_50=mAP_50, mAP_AVG=mAP_AVG, best_mAP=best_mAP)))

    print(utils.table_format(best_test_info, cfg.TIOU_THRESH, '[CoLA] THUMOS\'14 Performance'))


def train_one_step(net, loader_iter, optimizer, criterion, writter, step):
    net.train()

    data, label, _, _, _ = next(loader_iter)
    data = data.cuda()
    label = label.cuda()

    optimizer.zero_grad()
    video_scores, contrast_pairs, _, _ = net(data)
    cost, loss = criterion(video_scores, label, contrast_pairs)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        writter.add_scalar(key, loss[key].cpu().item(), step)
    return cost


@torch.no_grad()
def get_easy_act(net, cfg, test_loader, test_info, step, writter=None):
    # cfg.DATA_ANNO_PATH = os.path.join(cfg.DATA_ANNO_PATH, str(cfg.iter))
    anet_label_path = "data/gt.json"
    anet_label = load_json(anet_label_path)
    anet_label = anet_label['database']
    training_index, action_2_video, video_2_action = get_train_label(anet_label)

    subset_atten_fea = []
    subset_index = []
    # cfg.iter = int(cfg.iter) - 1
    # utils.set_path(cfg)

    # cfg.iter = int(cfg.iter) + 1
    # utils.set_path(cfg)
    if int(cfg.iter) == 0:
        for data, label, _, vid, vid_num_seg in test_loader:
            data = data.cpu().data.numpy()
            easy_act = np.squeeze(data, axis=(0,))
            subset_atten_fea.append(easy_act.mean(axis=0))
            subset_index.append(vid[0])
        subset_atten_fea = np.stack(subset_atten_fea, axis=0)

        num_of_cluster = 20

        label_pred = cluster(num_of_cluster, subset_atten_fea)
        cluser_2_action, soft_cluster_2_action = get_cluster_performance(20, label_pred, subset_index, action_2_video,
                                                                         video_2_action, path=cfg.DATA_ANNO_PATH,
                                                                         dump=True)
        dump_soft_cluster_2_action(soft_cluster_2_action, cfg.DATA_ANNO_PATH)

    if int(cfg.iter) != 0:
        net.eval()
        model_file = os.path.join(cfg.MODEL_PATH, \
                                  "model_best.pth.tar")
        if model_file:
            print('=> loading model: {}'.format(model_file))
            net.load_state_dict(torch.load(model_file))
            print('=> tesing model...')

        for data, label, _, vid, vid_num_seg in test_loader:
            data, label = data.cuda(), label.cuda()
            _, contrast_pairs, _, _ = net(data)
            easy_act = contrast_pairs['EA'].cpu().data.numpy()
            easy_act = np.squeeze(easy_act, axis=(0,))
            if easy_act.shape[0] == 0:
                continue
            else:
                subset_atten_fea.append(easy_act.mean(axis=0))
                subset_index.append(vid[0])
        subset_atten_fea = np.stack(subset_atten_fea, axis=0)
        print(subset_atten_fea.shape)

        num_of_cluster = 20

        label_pred = cluster(num_of_cluster, subset_atten_fea)
        cluser_2_action, soft_cluster_2_action = get_cluster_performance(20, label_pred, subset_index, action_2_video,
                                                                         video_2_action, path=cfg.DATA_ANNO_PATH,
                                                                         dump=True)
        dump_soft_cluster_2_action(soft_cluster_2_action, cfg.DATA_ANNO_PATH)


@torch.no_grad()
def test_all(net, cfg, test_loader, test_info, step, writter=None, model_file=None):
    net.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        net.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': '[CoLA] https://github.com/zhang-can/CoLA', 'results': {}}

    acc = AverageMeter()

    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, _, actionness, cas = net(data)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()

        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))), correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])

        cas_pred = utils.get_pred_activations(cas, pred, cfg)
        aness_pred = utils.get_pred_activations(actionness, pred, cfg)
        proposal_dict = utils.get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)

        final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _, v in proposal_dict.items()]
        # final_proposals = [utils.soft_nms(v) for _,v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))

    anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                                   subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                                   verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if writter:
        writter.add_scalar('Test Performance/Accuracy', acc.avg, step)
        writter.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
        for i in range(cfg.TIOU_THRESH.shape[0]):
            writter.add_scalar('mAP@tIOU/mAP@{:.1f}'.format(cfg.TIOU_THRESH[i]), mAP[i], step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)

    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
    return test_info['mAP@0.5'][-1], average_mAP


if __name__ == "__main__":
    assert len(sys.argv) >= 2 and sys.argv[1] in ['train', 'test'], 'Please set mode (choices: [train] or [test])'
    cfg.MODE = sys.argv[1]
    if cfg.MODE == 'train':
        # assert len(sys.argv) == 4, 'Please set iter num'  # 根据参数更新一些config
        cfg.iter = sys.argv[2]
        # cfg.enlarge_rate = sys.argv[3]
        cfg.LR = '[0.0001]*' + str(int(6000))
        cfg.NUM_ITERS = len(eval(cfg.LR))
        cfg.DATA_ANNO_PATH = os.path.join(cfg.DATA_ANNO_PATH, cfg.iter)
    if cfg.MODE == 'test':
        # assert len(sys.argv) == 4, 'Please set iter num'  # 根据参数更新一些config
        cfg.iter = sys.argv[2]
        # cfg.enlarge_rate = sys.argv[3]
        cfg.DATA_ANNO_PATH = os.path.join(cfg.DATA_ANNO_PATH, str(int(cfg.iter)))
        if not os.path.exists(cfg.DATA_ANNO_PATH):
            os.makedirs(cfg.DATA_ANNO_PATH)
    print(AsciiTable([['CoLA - Compare to Localize Actions']]).table)
    main()