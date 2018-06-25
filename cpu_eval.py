import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
from dataloader import Image_loader, crop_from_dets, Mscoco
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction
# implementation on drawCOCO
from visualize.visual import drawCOCO
import os
from tqdm import tqdm
import time

import numpy as np
from ssd.torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
from ssd.torchcv.models.ssd import SSD512, SSDBoxCoder
# import visualization utilities
from ssd.torchcv.visualizations import vis_image
from pPose_nms import pose_nms, write_json

# svm classification
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split, \
    ShuffleSplit, LeaveOneOut, KFold, LeavePOut, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import json
import argparse
import random
import pickle
import itertools

from opt import opt
args = opt
args.dataset = 'coco'

def preprocess(image):
    # image of numpy in RGB mode

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    ow = oh = 512
    img = image.resize((ow, oh))
    inp = torchvision.transforms.ToTensor()(image)
    img = transform(img)
    img = img[np.newaxis]
    inp = inp[np.newaxis]

    return img, inp

def inference_from_image(image):
    # Load SSD model
    print('Loading SSD model..')
    det_model = FPNSSD512(num_classes=21).cpu()
    det_model.load_state_dict(
        torch.load('./models/ssd/fpnssd512_20_trained.pth'))

    det_model.eval()
    box_coder = SSDBoxCoder(det_model)
    pose_dataset = Mscoco()
    #pose_model = InferenNet_faster(4 * 1 + 1, pose_dataset)
    pose_model = InferenNet(4 * 1 + 1, pose_dataset, cuda=False)

    #pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.cpu()
    #pose_model.cpu()
    pose_model.eval()
    # cannot run
    #pose_model.half()

    final_result = []


    img, inp = preprocess(image)
    start_time = time.time()
    with torch.no_grad():

        # for resizing the bounding box
        ht = inp.size(2)
        wd = inp.size(3)
        # Human Detection
        img = Variable(img).cpu()
        print(img.size())
        loc_preds, cls_preds = det_model(img)

        boxes, labels, scores = box_coder.decode(ht, wd,
            loc_preds.data.squeeze().cpu(), F.softmax(cls_preds.squeeze(), dim=1).data.cpu())

        if len(boxes) == 0:
            pass

        assert boxes.shape[0] == scores.shape[0]

        inps, pt1, pt2 = crop_from_dets(inp[0], boxes, scores)

        # thnn_conv2d_forward is not implemented for type torch.HalfTensor
        # inps = Variable(inps.cpu().half(), volatile=True)
        inps = Variable(inps.cpu())

        # time for search the bbox
        det_time1 = time.time() - start_time
        # Expected object of type torch.FloatTensor but found type torch.HalfTensor for argument #2 'weight'
        hm = pose_model(inps).float()

        # n*kp*2 | n*kp*1
        preds_hm, preds_img, preds_scores = getPrediction(
            hm.cpu().data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

        # time for pose estimation
        det_time2 = time.time() - start_time
        # print('\n\n\n')
        # print("for pose prediction: ", det_time2)

        #print(boxes)
        result = pose_nms(boxes, scores, preds_img, preds_scores)

        # time for pose nms
        det_time3 = time.time() - start_time
        #print(result)

        vis_image(np.transpose(inp[0].data.numpy(), (1, 2, 0)),
                  [res['bbox'] for res in result])
        drawCOCO(np.transpose(inp[0].data.numpy(), (1, 2, 0)), result)

        #print(result)

        result = {
            'result': result,
            'imgsize': (ht, wd)
        }
        #print(result)
        final_result.append(result)

        # convert to numpy
        # some redundant dimension
        keyposes = [np.squeeze(x['keypoints'].numpy()) for x in result['result']]
        scores = [x['kp_score'].numpy() for x in result['result']]

        # normalize to within bbox
        bboxes = np.array([x['bbox'].numpy() for x in result['result']])
        assert(len(bboxes) == len(keyposes))
        xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # list of width and height
        widths = xmax - xmin
        heights = ymax - ymin

        feature_list = list()

        for i in range(len(keyposes)):
            width = widths[i]
            height = heights[i]

            feature = np.array([])

            # normalize it
            for j in range(scores[i].shape[0]):
                feature = np.append(feature, (keyposes[i][j][0] - xmin[i]) / width)
                feature = np.append(feature, (keyposes[i][j][1] - ymin[i]) / height)
                feature = np.append(feature, scores[i][j])

            feature_list.append(feature)

        feature_list = np.array(feature_list)

        from sklearn.externals import joblib
        clf = joblib.load('./models/svm/keypoint_recall.pkl')

        rep = clf.predict_proba(feature_list)
        print(rep)

    # TQDM
    print('Speed: {total:.2f} FPS | Num Poses: {pose} | Det time1, 2, 3: {det:.3f}, '.format(
        total=1 / (time.time() - start_time),
        pose=len(result['result']),
        det=det_time1)
          +
          '{det2:.3f}, {det3:.3f}'.format(
              det2=det_time2 - det_time1,
              det3=det_time3 - det_time2)
    )


    #write_json(final_result, args.outputpath)

# main function for testing
if __name__ == '__main__':
    from PIL import Image
    #image = Image.open('examples/demo/2.jpg')
    image_location = '/home/king-kong/Downloads/storage/datasets/ai_challenger_dataset/ai_challenger_keypoint_train_20170902/ai_challenger_not_fight/08841929973516d175e1ce93962aa1aed5a50f9e.jpg'
    image_location = 'examples/demo/1.jpg'

    image = Image.open(image_location)

    if image.mode == 'L':
        image = imagel.convert('RGB')

    inference_from_image(image)
