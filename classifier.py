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

class Classifier():
    def __init__(self, cuda=False):
        ssd_path = './models/ssd/fpnssd512_20_trained.pth'
        self.cuda = cuda

        # load for cpu
        if not cuda:
            # Load SSD model
            print('Loading SSD model..')
            self.det_model = FPNSSD512(num_classes=21).cpu()
            self.det_model.load_state_dict(
                torch.load('./models/ssd/fpnssd512_20_trained.pth',
                           map_location='cpu'))

            self.det_model.eval()
            self.box_coder = SSDBoxCoder(self.det_model)
            pose_dataset = Mscoco()
            #pose_model = InferenNet_faster(4 * 1 + 1, pose_dataset)
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset, cuda=False)

            self.pose_model.cpu()
            self.pose_model.eval()
        else:
            print('Loading SSD model..')
            self.det_model = FPNSSD512(num_classes=21).cuda()
            self.det_model.load_state_dict(
                torch.load('./models/ssd/fpnssd512_20_trained.pth'))

            self.det_model.eval()
            self.box_coder = SSDBoxCoder(self.det_model)
            pose_dataset = Mscoco()
            #pose_model = InferenNet_faster(4 * 1 + 1, pose_dataset)
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset, cuda=True)

            self.pose_model.cuda()
            self.pose_model.eval()

    def preprocess(self, image):
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

    def svm_predict(self, keyposes, scores, bboxes):
        # normalize to within bbox
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

        return rep
        #print(rep)


    def predict(self, image):
        img, inp = self.preprocess(image)
        start_time = time.time()

        with torch.no_grad():

            # for resizing the bounding box
            ht = inp.size(2)
            wd = inp.size(3)
            if self.cuda:
                # Human Detection
                img = Variable(img).cuda()
            else:
                img = Variable(img).cpu()

            print(img.size())
            loc_preds, cls_preds = self.det_model(img)

            boxes, labels, scores = self.box_coder.decode(ht, wd,
                                                          loc_preds.data.squeeze().cpu(),
                                                          F.softmax(cls_preds.squeeze(),
                                                                    dim=1).data.cpu())

            if len(boxes) == 0:
                pass

            assert boxes.shape[0] == scores.shape[0]

            inps, pt1, pt2 = crop_from_dets(inp[0], boxes, scores)

            # thnn_conv2d_forward is not implemented for type torch.HalfTensor
            # inps = Variable(inps.cpu().half(), volatile=True)
            if self.cuda:
                inps = Variable(inps.cuda())
            else:
                inps = Variable(inps.cpu())

            # time for search the bbox
            det_time1 = time.time() - start_time
            # Expected object of type torch.FloatTensor but found type torch.HalfTensor for argument #2 'weight'
            hm = self.pose_model(inps).float()

            # n*kp*2 | n*kp*1
            preds_hm, preds_img, preds_scores = getPrediction(
                hm.cpu().data, pt1, pt2,
                opt.inputResH,
                opt.inputResW,
                opt.outputResH,
                opt.outputResW)

            # time for pose estimation
            det_time2 = time.time() - start_time

            result = pose_nms(boxes, scores, preds_img, preds_scores)

            # time for pose nms
            det_time3 = time.time() - start_time

            # visualization
            # vis_image(np.transpose(inp[0].data.numpy(), (1, 2, 0)),
            #           [res['bbox'] for res in result])
            # drawCOCO(np.transpose(inp[0].data.numpy(), (1, 2, 0)), result)

            result = {
                'result': result,
                'imgsize': (ht, wd)
            }

            # convert result to numpy
            # some redundant dimension
            keyposes = [np.squeeze(x['keypoints'].numpy()) for x in result['result']]
            scores = [x['kp_score'].numpy() for x in result['result']]
            bboxes = np.array([x['bbox'].numpy() for x in result['result']])

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

            rep = self.svm_predict(keyposes, scores, bboxes)
            return rep, bboxes

if __name__ == '__main__':
    from PIL import Image
    image_location = 'examples/demo/1.jpg'
    image = Image.open(image_location)

    if image.mode == 'L':
        image = imagel.convert('RGB')

    clf = Classifier(cuda=False)
    rep = clf.predict(image)
    print(rep)
