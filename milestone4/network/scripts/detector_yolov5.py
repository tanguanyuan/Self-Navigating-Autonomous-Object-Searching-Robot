import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.augmentations import letterbox
from torchvision import transforms
import cv2

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        self.model = DetectMultiBackend('network/scripts/best5.pt')
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        cmd_printer.divider(text="warning")
        print('This detector uses "RGB" input convention by default')
        print('If you are using Opencv, the image is likely to be in "BRG"!!!')
        cmd_printer.divider()
        self.colour_code = np.array([(220, 220, 220), (128, 0, 0), (155, 255, 70), (255, 85, 0), (255, 180, 0), (0, 128, 0)])
        # color of background, redapple, greenapple, orange, mango, capsicum

    def detect_single_image(self, np_img):
        # torch_img = self.np_img2torch(np_img)
        # tick = time.time()

        # Preprocessing
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz, s=stride)

        im = self.preprocess_image(np_img, imgsz, stride, pt)

        with torch.no_grad():
            pred = self.model(im)
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            # if self.use_gpu:
            #     pred = torch.argmax(pred.squeeze(),
            #                         dim=0).detach().cpu().numpy()
            # else:
            #     pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()

        pred = self.convert_pred_to_labels(pred, np_img, im)
        # dt = time.time() - tick
        # print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        colour_map = self.visualise_output(pred)
        return pred, colour_map

    def preprocess_image(self, np_img, imgsz, stride, pt=True):
        im = letterbox(np_img, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))  # HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        return im

    def convert_pred_to_labels(self, pred, np_img, im):
        for i, det in enumerate(pred):  # per image
            im0 = np.zeros(np_img.copy().shape)
            im0 = np.uint8(im0)
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if self.validate_pred(xyxy, conf, cls):
                        im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :] = cls
        
        pred = im0[:, :, 0].squeeze()
        return pred

    def validate_pred(self, xyxy, conf, cls):
        width = 1.0*int(xyxy[2]) - 1.0*int(xyxy[0])
        height = 1.0*int(xyxy[3]) - 1.0*int(xyxy[1])
        
        # invalidate pred if :
        if conf < 0.7: # not confident
            return False
        
        # too long
        if cls in [1, 2, 3, 5]: # for redapples, greenapples, orange, capsicum
            if width/height > 1.1: # not valid if width/height ratio is more than 1.1
                return False
        
        # too short
        if cls in [1]: # for redapples
            if height/width < 1.1: 
                return False
            
        # too tall
        if cls == 4: # for mango
            if height/width > 1.1:
                return False
        
        return True
        
    
    def visualise_output(self, nn_output):
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
        colour_map = np.stack([r, g, b], axis=2)
        colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 160)
        pad = 5
        labels = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map  = cv2.putText(colour_map, labels[i-1],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (0, 0, 0))
            pt = (pt[0], pt[1]+h+pad)
        return colour_map

    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path,
                              map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                        #                         saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        return img
