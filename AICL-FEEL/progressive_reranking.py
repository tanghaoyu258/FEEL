import math
import numpy as np
from re_ranking import re_ranking
from anet_clustering import action_2_video
from anet_clustering import video_2_action
from anet_clustering import get_cluster_performance
from anet_clustering import dump_soft_cluster_2_action

def progressive_learning(vid_name, fea, label, top_rate, path):
    # video_2_fea = load_json('video_2_fea.json')
    #label_2_video : dict class_num_label:[tuple(video_name,id)]
    # print(label_2_video)
    all_class = len(set(label))


    all_center = []
    cluster_center = {}
    for i in range(all_class):
        cluster_center[i] = []

    all_fea = []
    all_item = []

    #---getAllCenter---#

    for i in range(fea.shape[0]):
        key = label[i]
        cluster_center[key].append(fea[i])

    for i in range(all_class):
        cluster_center[i] = np.stack(cluster_center[i], axis=0)
        cluster_center[i] = cluster_center[i].mean(axis=0)
        all_center.append(cluster_center[i])
    #     tmp_all_fea = np.stack(tmp_all_fea, axis=0)
    #     cluster_center = tmp_all_fea.mean(axis=0)
    #     #cluster_center = cluster_center.reshape(1,cluster_center.shape[0])
    #     all_center.append(cluster_center)
    # all_fea = np.stack(all_fea, axis=0)

    #---reLocalization---#

    top_rate = top_rate
    top_rate = min(1, top_rate)
    dists = []
    all_num = len(label)
    for i in range(len(all_center)):
        key = i
        cluster_center = all_center[i]
        cluster_center = cluster_center.reshape(1, cluster_center.shape[0])
        final_dist = re_ranking(cluster_center, fea)
        final_dist = np.array(final_dist).flatten()
        dists.append(final_dist)
        # index = np.argsort(final_dist)  # 正序输出索引，从小到大
        # choose_num = math.ceil(len(label_2_video[key]) * top_rate)
        # index = index[:choose_num]
        # label_2_video_re[key] = np.array(all_item)[index]
        # #label_2_video_re[key] = [tuple(i) for i in label_2_video_re[key]]
        # label_2_video[key] = [tuple(i) for i in label_2_video_re[key]]
    dists = np.stack(dists, axis=0)
    index_min = np.argmin(dists, axis=0)
    dists_min = np.min(dists, axis=0)

    pred_score = - dists_min
    labels = index_min
    # nums_to_select = math.ceil(all_num * top_rate)
    # v = np.zeros(len(pred_score))
    index = np.argsort(-pred_score)
    # all_num = 0
    num_to_select = np.zeros(all_class)
    for i in labels:
        num_to_select[i] += 1
    for i in range(len(num_to_select)):
        num_to_select[i] = round(num_to_select[i] * top_rate)
    all_num = np.sum(num_to_select)
    # print(all_item)
    # print(len(all_item))
    # print(labels)
    # print(len(labels))
    label_pred_all = []
    subset_index_all = []
    for i in range(len(index)):  # index[i]样本编号
        label_pred_all.append(labels[index[i]])
        subset_index_all.append(vid_name[index[i]])
    get_cluster_performance(100, label_pred_all, subset_index_all,
                            action_2_video, video_2_action, path=path, dump=False)


    label_pred = []
    subset_index = []
    for i in range(len(index)): # index[i]样本编号
        # label_2_video_re[labels[index[i]]].append(all_item[index[i]])
        if all_num == 0:
            break
        if num_to_select[labels[index[i]]] > 0:
            label_pred.append(labels[index[i]])
            subset_index.append(vid_name[index[i]])
            num_to_select[labels[index[i]]] -= 1
            all_num -= 1
        else:
            continue


    #evaluate

    # for i in range(len(all_class)):
    #     key = i
    #     value = label_2_video[key]
    #     for tv in value:
    #         video_name = tv[0]
    #         res = 1
    #         for j in range(len(video_name)):
    #             if video_name[-(j + 2):-(j + 1)] == '-':
    #                 res = res + j
    #                 break
    #         #video_fea = np.array(video_2_fea[video_name[:-(1 + res)]])
    #         #video_fea = np.array(video_2_fea[video_name[:-(1+len(str(key)))]])
    #         label_pred.append(key)
    #         subset_index.append(video_name[:-(1 + res)])
    cluser_2_action, soft_cluster_2_action = get_cluster_performance(100, label_pred, subset_index, action_2_video, video_2_action, path=path, dump=True)
    dump_soft_cluster_2_action(soft_cluster_2_action, path)