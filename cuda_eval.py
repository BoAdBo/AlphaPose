import torch.nn.functional as F
import torchvision.transforms as transforms

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

from ssd.torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
from ssd.torchcv.models.ssd import SSD512, SSDBoxCoder
# import visualization utilities
from ssd.torchcv.visualizations import vis_image
from pPose_nms import pose_nms, write_json

# import preprocess transform
import torchvision.transforms as transforms

from opt import opt
#args = opt
#args.dataset = 'coco'

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms,Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    ow = oh = 512
    im = image.resize((ow, oh))
    inp = image

    return im, inp

#
def inference_from_image(image):
    # Load SSD model
    print('Loading SSD model..')
    det_model = FPNSSD512(num_classes=21).cuda()
    det_model.load_state_dict(
        torch.load('./models/ssd/fpnssd512_20_trained.pth'))

    det_model.eval()
    box_coder = SSDBoxCoder(det_model)

    pose_dataset = Mscoco()

    #pose_model = InferenNet_faster(4 * 1 + 1, pose_dataset)
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)

    pose_model.cuda()
    pose_model.eval()

    final_result = []

    # do some preprocessing

    #for i, (img, inp, im_name) in enumerate(im_names_desc):
    img, inp = preprocess(image)
    start_time = time.time()
    with torch.no_grad():

        # for resizing the bounding box
        ht = inp.size(2)
        wd = inp.size(3)
        # Human Detection
        img = Variable(img).cuda()
        loc_preds, cls_preds = det_model(img)

        # the ht, wd of the function is removed, yet passing in these two,
        # inducing troubles
        boxes, labels, scores = box_coder.decode(ht, wd,
            loc_preds.data.squeeze().cpu(), F.softmax(cls_preds.squeeze(), dim=1).data.cpu())

        if len(boxes) == 0:
            pass#continue
        # if no boxes is detected, return None
        # if boxes.shape[0] == 0:
        #     continue

        assert boxes.shape[0] == scores.shape[0]

        inps, pt1, pt2 = crop_from_dets(inp[0], boxes, scores)

        # thnn_conv2d_forward is not implemented for type torch.HalfTensor
        # inps = Variable(inps.cpu().half(), volatile=True)
        inps = Variable(inps.cuda())

        # time for search the bbox
        det_time1 = time.time() - start_time
        # Expected object of type torch.FloatTensor but found type torch.HalfTensor for argument #2 'weight'
        hm = pose_model(inps).float()

        # n*kp*2 | n*kp*1
        preds_hm, preds_img, preds_scores = getPrediction(
            hm.cpu().data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

        # time for pose estimation
        det_time2 = time.time() - start_time

        result = pose_nms(boxes, scores, preds_img, preds_scores)

        # time for pose nms
        det_time3 = time.time() - start_time

        # vis_image(np.transpose(inp[0].data.numpy(), (1, 2, 0)),
        #           [res['bbox'] for res in result])
        # drawCOCO(np.transpose(inp[0].data.numpy(), (1, 2, 0)), result)

        result = {
            'imgname': im_name[0],
            'result': result,
            'imgsize': (ht, wd)
        }
        final_result.append(result)

        print('Speed: {total:.2f} FPS | Num Poses: {pose} | Det time1, 2, 3: {det:.3f}, '.format(
                total=1 / (time.time() - start_time),
            pose=len(result['result']),
            det=det_time1)
        +
        '{det2:.3f}, {det3:.3f}'.format(
            det2=det_time2 - det_time1,
            det3=det_time3 - det_time2))

    write_json(final_result, args.outputpath)
