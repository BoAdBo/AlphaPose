import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
from dataloader import Image_loader, crop_from_dets, Mscoco
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction
#from SPPE.src.utils.img import drawCOCO
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

from opt import opt
args = opt
args.dataset = 'coco'
#torch.set_num_threads(12)

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load SSD model
    print('Loading SSD model..')
    det_model = FPNSSD512(num_classes=21).cpu()
    det_model.load_state_dict(
        torch.load('./models/ssd/fpnssd512_20_trained.pth',
                   map_location='cpu'))

    det_model.eval()
    box_coder = SSDBoxCoder(det_model)

    # don't understand...
    #box_coder = FPNSSDBoxCoder()

    print(inputpath)
    print(inputlist)
    if len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    elif len(inputlist):
        with open(inputlist, 'r') as f:
            im_names = []
            for line in f.readlines():
                im_names.append(line.split('\n')[0])
    else:
        raise IOError('Error: ./run.sh must contain either --indir/--list')

    dataset = Image_loader(inputlist)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True
    )
    im_names_desc = tqdm(test_loader)

    pose_dataset = Mscoco()
    if opt.fast_inference:
        pose_model = InferenNet_faster(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)

    pose_model = torch.nn.DataParallel(pose_model).cpu()
    pose_model.cpu()
    pose_model.eval()
    # cannot run
    #pose_model.half()

    final_result = []

    for i, (img, inp, im_name) in enumerate(im_names_desc):
        start_time = time.time()
        with torch.no_grad():

            # for resizing the bounding box
            ht = inp.size(2)
            wd = inp.size(3)
            # Human Detection
            img = Variable(img).cpu()
            loc_preds, cls_preds = det_model(img)
            #print(img.size())

            # some exception handling for not finding a person in image
            # pytorch cat() would compian: RuntimeError: expected a non-empty list of Tensors

            # the ht, wd of the function is removed, yet passing in these two,
            # inducing troubles
            boxes, labels, scores = box_coder.decode(ht, wd,
                loc_preds.data.squeeze().cpu(), F.softmax(cls_preds.squeeze(), dim=1).data.cpu())

            # label in tensor, so no way to visualize it
            #print(labels)
            # visualize it
            #vis_image(np.transpose(img[0].data.numpy(), (1, 2, 0)), boxes, label_names=None, scores=scores)

            if len(boxes) == 0:
                continue
            # if no boxes is detected, return None
            # if boxes.shape[0] == 0:
            #     continue

            assert boxes.shape[0] == scores.shape[0]
            # opt.inputResH = inp[0].size(1)
            # opt.inputResW = inp[0].size(2)
            # print(inp[0].size(1))
            # print(inp[0].size(2))
            # print(opt.inputResH)
            # print(opt.inputResW)
            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp[0], boxes, scores)

            # thnn_conv2d_forward is not implemented for type torch.HalfTensor
            # inps = Variable(inps.cpu().half(), volatile=True)
            inps = Variable(inps.cpu())
            det_time = time.time() - start_time
            # Expected object of type torch.FloatTensor but found type torch.HalfTensor for argument #2 'weight'
            hm = pose_model(inps).float()

            # n*kp*2 | n*kp*1
            preds_hm, preds_img, preds_scores = getPrediction(
                hm.cpu().data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

            #print(boxes)
            result = pose_nms(boxes, scores, preds_img, preds_scores)
            #print(result)

            # vis_image(np.transpose(inp[0].data.numpy(), (1, 2, 0)),
            #           [res['bbox'] for res in result])
            #drawCOCO(np.transpose(inp[0].data.numpy(), (1, 2, 0)), result)

            result = {
                'imgname': im_name[0],
                'result': result
            }
            final_result.append(result)

        # TQDM
        im_names_desc.set_description(
            'Speed: {total:.2f} FPS | Num Poses: {pose} | Det time: {det:.3f}'.format(
                total=1 / (time.time() - start_time),
                pose=len(result['result']),
                det=det_time)
        )

    write_json(final_result, args.outputpath)
