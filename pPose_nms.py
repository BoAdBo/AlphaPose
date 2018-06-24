# -*- coding: utf-8 -*-
import torch
import json
import os
import zipfile
import math
import numpy as np


''' Constant Configuration '''
# trained parameters
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
alpha = 0.1


def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores):
    pose_scores[pose_scores == 0] = 1e-5

    final_result = []

    ori_bbox_scores = bbox_scores.clone()
    ori_pose_preds = pose_preds.clone()
    ori_pose_scores = pose_scores.clone()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin

    # 1/10 of the Bounding box's dimension (larger of width or height)
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(dim=1)

    human_ids = np.arange(nsamples)
    # Do pPose-NMS
    pick = []
    merge_ids = []

    # Suspect that this is to arrange poses by similarity and matched keyposes
    # for later merging
    while(human_scores.shape[0] != 0):
        # Pick the one with highest score as reference
        pick_id = torch.argmax(human_scores)

        pick.append(human_ids[pick_id])
        # num_visPart = torch.sum(pose_scores[pick_id] > 0.2)

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= matchThreds)]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id
        #else:
        #    delete_ids = torch.from_numpy(delete_ids)

        #print(delete_ids)
        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)
        bbox_scores = np.delete(bbox_scores, delete_ids, axis=0)
    #print(human_ids)
    #print(pick)
    #print(merge_ids)

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    num_pick = 0

    # added bbox picking
    bbox_pick = bboxes[pick]

    for j in range(len(pick)):
        ids = np.arange(17)
        #print(scores_pick)
        max_score = torch.max(scores_pick[j, ids, 0])
        #print(scores_pick[j, ids, 0])

        # max score of each keypose under threshold cast it out
        if max_score < scoreThreds:
            continue

        # Merge poses
        # could save some computation when two poses are the same,
        # which is very likely
        merge_id = merge_ids[j]

        merge_pose, merge_score = None, None

        merge_pose, merge_score = p_merge(preds_pick[j],
                                          ori_pose_preds[merge_id],
                                          ori_pose_scores[merge_id],
                                          ref_dists[pick[j]])

        # don't merge box... just add to result
        merge_box = bbox_pick[j]

        ## give up...
        # # if merge id has only the already picked one, no need to merge
        # if len(merge_id) > 1 and merge_id[0] != pick[j]:
        #     pass
        #     # merge_pose, merge_score = p_merge(preds_pick[j],
        #     #                                   ori_pose_preds[merge_id],
        #     #                                   ori_pose_scores[merge_id],
        #     #                                   ref_dists[pick[j]])
        # else:
        #     merge_pose, merge_score = preds_pick[j], scores_pick[j]
        #     print((preds_pick[j] == merge_pose).all())
        #     print((merge_score[np.newaxis].t() == scores_pick[j]).all())

        max_score = torch.max(merge_score[ids])
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])

        # if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < 40 * 40.5):
        #     continue
        if (xmax - xmin) * (ymax - ymin) < 720.0:
            continue

        # bbox added
        # bbox in the fashion of upper left, bottom right
        num_pick += 1
        final_result.append({
            'keypoints': merge_pose - 0.3,
            'bbox': merge_box,
            'kp_score': merge_score,
            'proposal_score': torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score)
        })

    return final_result


def p_merge(ref_pose, cluster_preds, cluster_scores, ref_dist):
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    for i in range(kp_num):
        cluster_joint_scores = cluster_scores[:, i][mask[:, i]]
        cluster_joint_location = cluster_preds[:, i, :][mask[:, i].unsqueeze(
            -1).repeat(1, 2)].view((torch.sum(mask[:, i]), -1))

        # Get an normalized score
        normed_scores = cluster_joint_scores / torch.sum(cluster_joint_scores)

        # Merge poses by a weighted sum
        final_pose[i, 0] = torch.dot(cluster_joint_location[:, 0], normed_scores.squeeze(-1))
        final_pose[i, 1] = torch.dot(cluster_joint_location[:, 1], normed_scores.squeeze(-1))

        final_score[i] = torch.dot(cluster_joint_scores.transpose(0, 1).squeeze(0), normed_scores.squeeze(-1))

    return final_pose, final_score


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    # the reference keypoint
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_preds[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    mask = (dist <= 1)

    # Define a keypoints distance
    score_dists = torch.zeros(all_preds.shape[0], 17)
    keypoint_scores.squeeze_()
    if keypoint_scores.dim() == 1:
        keypoint_scores.unsqueeze_(0)
    if pred_scores.dim() == 1:
        pred_scores.unsqueeze_(1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.repeat(1, all_preds.shape[0]).transpose(0, 1)

    score_dists[mask] = torch.tanh(pred_scores[mask] / delta1) * torch.tanh(keypoint_scores[mask] / delta1)

    point_dist = torch.exp((-1) * dist / delta2)
    final_dist = torch.sum(score_dists, dim=1) + mu * torch.sum(point_dist, dim=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_pred[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = torch.sum(
        dist / ref_dist <= 1,
        dim=1
    )

    return num_match_keypoints


def write_json(all_results, outputpath):
    json_results = []
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            result['image_id'] = int(im_name.split('/')[-1].split('.')[0])
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)

            # added
            result['bbox'] = [float(x) for x in human['bbox']]

            json_results.append(result)

            # coco_part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
            # colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
            # pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[11,13],[13,15],[12,14],[14,16],[6,12],[5,11]]
            # colors_skeleton = ['y', 'y', 'y', 'y', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r','m','m']
            # for idx_c, color in enumerate(colors):
            #     plt.plot(np.clip(pose[idx_c,0],0,width),
            #              np.clip(pose[idx_c,1],0,height),
            #              marker='o', color=color, ms=4*np.mean(pose[idx_c,2]))
            # for idx in range(len(colors_skeleton)):
            #     plt.plot(np.clip(pose[pairs[idx],0],0,width),np.clip(pose[pairs[idx],1],0,height),'r-',
            #              color=colors_skeleton[idx],linewidth=4*np.mean(pose[pairs[idx],2]), alpha=0.12*np.mean(pose[pairs[idx],2]))

    with open(os.path.join(outputpath, 'keypoint_result.json'), 'w') as json_file:
        json_file.write(json.dumps(json_results))
    result_zip = zipfile.ZipFile(os.path.join(outputpath, 'keypoint_result.zip'), 'w')
    result_zip.write(os.path.join(outputpath, 'keypoint_result.json'))
    result_zip.close()
