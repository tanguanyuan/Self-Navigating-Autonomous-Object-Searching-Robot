import os
import sys
from pathlib import Path
import numpy as np

FILE = Path(os.path.abspath('')).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.augmentations import letterbox

# testing loading img
model = DetectMultiBackend('best.pt')
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)
imgsz = check_img_size(imgsz, s=stride)
bs = 1
source = 'image_6.png'

im0s = cv2.imread(source)
im = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]  # padded resize
im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
im = np.ascontiguousarray(im)  # contiguous

im = torch.from_numpy(im)
im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

# Inference
pred = model(im, augment=False, visualize=False)

# NMS
pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

for i, det in enumerate(pred):  # per image
    im0 = np.zeros(im0s.copy().shape)
    im0 = np.uint8(im0)

    if len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :] = 255 #np.uint8(cls);

print(det)