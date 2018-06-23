import numpy as np
import torch
import scipy.misc
from torchsample.transforms import SpecialCrop, Pad
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt

# the original image and the result json, accept only one image
# in numpy
# draw the image with keypoints
def drawCOCO(img, results):
    assert img.ndim == 3
    p_color = ['g', 'b', 'purple', 'b', 'purple',
               'y', 'orange', 'y', 'orange', 'y', 'orange',
               'pink', 'r', 'pink', 'r', 'pink', 'r']

    #img = img.data.numpy()
    #img = np.transpose(img, (1, 2, 0))

    # nImg = img.size(0)
    # imgs = []
    # for n in range(nImg):
    #     img = to_numpy(inps[n])
    #     img = np.transpose(img, (1, 2, 0))
    #     imgs.append(img)

    fig = plt.figure()
    plt.imshow(img)
    ax = fig.add_subplot(1, 1, 1)

    for result in results:
        poses = result['keypoints']
        scores = result['kp_score']

        for p in range(17):
            if scores[p] < 0.01:
                continue

            #x, y = poses.data.tolist()[p]
            x, y = poses.data[p]
            #print(poses.data[p])
            x, y = float(x), float(y)
            #print(x, y)
            cor = (round(x), round(y)), 3
            ax.add_patch(plt.Circle(*cor, color=p_color[p]))

    plt.axis('off')

    plt.show()

    return img
