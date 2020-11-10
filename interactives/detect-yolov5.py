# %% markdown
# # Scratchpad temporary code

# %% codecell
import sys
import os
import time
from os.path import join
sys.path.append(join(os.getcwd(), 'vframe_cli'))
from vframe.settings import app_cfg
sys.path.append('/work/vframe/3rdparty/yolov5')

# %% codecell
import cv2
import numpy as np
from vframe.models.geometry import BBox
from vframe.utils import draw_utils, file_utils, im_utils

# %% codecell
#from utils.torch_utils import select_device
#from models.experimental import attempt_load
# -----------------------------------------------------------------------------
# Start YoloV5
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        #c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        # x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        # s = 'Using CUDA '
        # for i in range(0, ng):
        #     if i == 1:
        #         s = ' ' * len(s)
            # print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__# Images()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


# -----------------------------------------------------------------------------
# END YoloV5
# -----------------------------------------------------------------------------

# %% codecell
# Model
fp_weights = '/data_store_vframe/vframe/training/sa_05a/runs/exp1/weights/best.pt'
fp_labels = '/work/vframe/modelzoo/models/pytorch/detection/sa_05a/labels.txt'
device = select_device('0')
model = attempt_load(fp_weights, map_location=device).autoshape()
model.conf = 0.5

# %% codecell
print(model.iou)
# %% codecell
# load labels
labels = file_utils.load_txt(fp_labels)

# %% codecell
# Images
fp_im = '/data_store_nas/datasets/vframe/photography/20201109_dresden_park/test/_MG_1359.jpg'
im = cv2.imread(fp_im)[:, :, ::-1]  # opencv (BGR to RGB)
dim = im.shape[:2][::-1]
imgs = [im]

# %% codecell
# Inference
st = time.time()
n_iters = 20
for i in range(n_iters):
    prediction = model(imgs, size=640)  # includes NMS
te = time.time() - st
print(f'{n_iters/te:.2f}')

# %% codecell
im_resized = im_utils.resize(im_utils.rgb2bgr(im), width=640)
dim_new = im_resized.shape[:2][::-1]

# %% codecell
from vframe.utils import im_utils
for i, img in enumerate(imgs):
    for pred in prediction[i]:
        pred = pred.tolist()
        xyxy = list(map(int, pred[:4]))
        bbox = BBox(*xyxy, *dim).redim(dim_new)
        conf = pred[4]
        class_idx = int(pred[5])
        label = f'{labels[class_idx]} {(100*conf):.2f}%'
        im_resized = draw_utils.draw_bbox(im_resized, bbox, label=label, size_label=10)
        im_pil = im_utils.np2pil(im_resized)
    im_pil.show()
