import os
import numpy as np
from sklearn.cluster import KMeans
import json

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


anet_label_path = "data/activity_net.v1-2.min-missing-removed.json"
anet_label = load_json(anet_label_path)
anet_label = anet_label['database']




def get_train_label(anet_label):
    training_index = []
    action_2_video = {}
    video_2_action = {}
    for tv in anet_label:
        if anet_label[tv]["subset"] != "training":
            continue
        training_index.append(tv)
        tc = anet_label[tv]["annotations"][0]["label"]
        #if action_2_video.has_key(tc):
        if tc in action_2_video:
            action_2_video[tc].append(tv)    #动作标签对应哪些视频
        else:
            action_2_video[tc] = []
            action_2_video[tc].append(tv)

        #if video_2_action.has_key(tv):
        if tv in video_2_action:
            video_2_action[tv].append(tc)    #视频对应的动作标签
        else:
            video_2_action[tv] = []
            video_2_action[tv].append(tc)

    for tk in action_2_video:
        action_2_video[tk] = list(set(action_2_video[tk]))
    for tk in video_2_action:
        video_2_action[tk] = list(set(video_2_action[tk]))
    return training_index, action_2_video, video_2_action


def get_subset(num_subset_class, action_2_video, training_index, all_atten_fea):
    # num_subset_cluster = num_subset_class
    subset_class = list(action_2_video.keys())[0:num_subset_class]
    subset_index = []
    subset_atten_fea = []
    dic = {}
    for tc in subset_class:
        tc_sub = action_2_video[tc]
        for tv in tc_sub:
            subset_index.append(tv)
            # print('!')
            # print(tv)
            tv_fea_index = training_index.index(tv)
            # print(all_atten_fea[tv_fea_index].shape)
            # print(all_atten_fea[tv_fea_index])
            dic[tv] = all_atten_fea[tv_fea_index].tolist()
            subset_atten_fea.append(all_atten_fea[tv_fea_index])
    subset_atten_fea = np.stack(subset_atten_fea, axis=0)
    anno_json_file = 'video_2_fea.json'
    temp_json = json.dumps(dic, indent=4, separators=(',', ': '))
    f = open(anno_json_file, 'w')
    f.write(temp_json)
    f.close()
    return subset_class, subset_index, subset_atten_fea

def dump_pseudo_label(num_of_cluster, cluster_res, anet_label, path):
    temp_json = anet_label
    for tmp_cls in range(num_of_cluster):  #聚类的伪标签->视频名
        # print(cluster_res[tmp_cls])
        for tv in cluster_res[tmp_cls]:
            for i in range(len(temp_json[tv]['annotations'])):
                temp_json[tv]['annotations'][i]['label'] = tmp_cls
    #dump
    temp_json = {'database': temp_json}
    #anno_json_file = 'labels/cluster_anno_anet12_100_r_0.json'
    anno_json_file = os.path.join(path, 'gt' + '.json')
    temp_json = json.dumps(temp_json, indent=4, separators=(',', ': '))
    f = open(anno_json_file, 'w')
    f.write(temp_json)
    f.close()

def dump_split_train(subset_index, path):
    anno_txt_file = os.path.join(path, 'split_train' + '.txt')
    file = open(anno_txt_file, "w", encoding='utf-8')
    for vid in subset_index:
        file.write(vid + '\n')
    file.close()

def get_cluster_performance(num_of_cluster, label_pred, subset_index, action_2_video, video_2_action, path, dump=False):
    cluster_res = {}
    for tmp_cls in range(num_of_cluster):
        #print(tmp_cls)
        cluster_res[tmp_cls] = []
    for i in range(len(label_pred)):
        tmp_cls = label_pred[i]
        file_name = subset_index[i]
        cluster_res[tmp_cls].append(file_name)

#    for tmp_cls in range(num_of_cluster):  #聚类的伪标签->视频名
#        print(cluster_res[tmp_cls])
    if dump == True:
        dump_pseudo_label(num_of_cluster, cluster_res, anet_label, path)
        dump_split_train(subset_index, path)

    all_precision = []
    all_recall = []
    all_cluster_label = []
    # add cluster index to action class mapping
    cluser_2_action = {}
    soft_cluster_2_action = {}

    total_true_cnt = 0

    # all class
    for label_index in range(num_of_cluster):
        if len(cluster_res[label_index]) == 0:
            continue
        all_class = list(action_2_video.keys())
        # print(all_class)
        action_cnt = {}
        for tc in all_class:
            action_cnt[tc] = 0
        # print(label_index)
        for tv in cluster_res[label_index]:
            tv_label = video_2_action[tv]
            for sig_label in tv_label:
                action_cnt[sig_label] += 1

            # set the label of cluster as the class which appear most
            max_cnt = 0
            cluster_label = ''
            for tmp_label in action_cnt:
                if action_cnt[tmp_label] > max_cnt:
                    max_cnt = action_cnt[tmp_label]
                    cluster_label = tmp_label
        all_cluster_label.append(cluster_label)

        # add cluster index to action class mapping
        cluser_2_action[label_index] = cluster_label

        soft_cluster_2_action[label_index] = []
        cluster_video_num = len(cluster_res[label_index])

        for tmp_label in action_cnt:
            if action_cnt[tmp_label] == 0:
                continue
            if action_cnt[tmp_label] == (max_cnt):
                # tmp_label_weight = 1.0 * action_cnt[tmp_label] / cluster_video_num
                tmp_label_weight = 1.0
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])
            elif action_cnt[tmp_label] >= 0.5 * max_cnt:
                tmp_label_weight = 0.5
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])

        precision = (1.0 * max_cnt) / (len(cluster_res[label_index]))
        recall = 1.0 * max_cnt / len(action_2_video[cluster_label])
        total_true_cnt += max_cnt

        # print("********")
        # print("cluster label %d" % label_index)
        # print("num of video in cluster %d" % (len(cluster_res[label_index])))
        # print("match class %s" % (cluster_label))
        # print("num of all gt video %d" % (len(action_2_video[cluster_label])))
        # print("num of cluster gt video %d" % (max_cnt))
        # print("precision %.4f" % precision)
        # print("recall %.4f\n" % recall)
        # action_cnt = sorted(action_cnt.items(), key=lambda e: e[1], reverse=True)
        # print(action_cnt[0:10])
        # print("\n")
        # print(soft_cluster_2_action[label_index])

        all_precision.append(precision)
        all_recall.append(recall)

    average_prec = np.mean(np.array(all_precision))
    average_recall = np.mean(np.array(all_recall))
    print("avg prec")
    print(average_prec)
    print("avg recall")
    print(average_recall)
    print("all prec %.4f" % (1.0 * total_true_cnt / len(subset_index)))

    all_cluster_label = list(set(all_cluster_label))
    print("num of cluster label %d" % (len(all_cluster_label)))

    video_index = subset_index
    gt_label = []
    for tv in video_index:
        gt_label.append(video_2_action[tv])
    #     non_over_label = []
    #     for i in range(label_pred.shape[0]):
    #         non_over_label.append(int(label_pred[i]))
    gt_label = np.array(gt_label)
    gt_label = np.squeeze(gt_label)
    # print(gt_label.shape)
    from sklearn import metrics
    print("Adjusted rand score %.4f" % metrics.adjusted_rand_score(gt_label, label_pred))
    print("NMI %.4f" % metrics.normalized_mutual_info_score(gt_label, label_pred))

    return cluser_2_action, soft_cluster_2_action

def dump_soft_cluster_2_action(soft_cluster_2_action, path):
    anno_json_file = os.path.join(path, 'soft_cluster_cls_names_anet12' + '.json')#'#data/soft_cluster_cls_names_anet12_r_' + str(iter) + '.json'
    temp_json = json.dumps(soft_cluster_2_action, indent=4, separators=(',', ': '))
    f = open(anno_json_file, 'w')
    f.write(temp_json)
    f.close()

def cluster(num_of_cluster, fea):
    estimator = KMeans(n_clusters=num_of_cluster, random_state=0)
    estimator.fit_predict(fea)
    label_pred = estimator.labels_
    return label_pred




training_index, action_2_video, video_2_action = get_train_label(anet_label)