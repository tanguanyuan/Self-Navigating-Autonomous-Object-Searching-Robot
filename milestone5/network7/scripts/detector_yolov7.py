import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from models.experimental import attempt_load
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from torchvision import transforms
import cv2

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        self.model = attempt_load('network7/scripts/best7_2.pt')
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

    def detect_single_image(self, np_img, np_aruco_img):
        # torch_img = self.np_img2torch(np_img)
        # tick = time.time()

        # Preprocessing

        im = self.preprocess_image(np_img)

        with torch.no_grad():
            pred = self.model(im)
            pred = non_max_suppression(pred[0], 0.25, 0.45, None, False)

            # if self.use_gpu:
            #     pred = torch.argmax(pred.squeeze(),
            #                         dim=0).detach().cpu().numpy()
            # else:
            #     pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()

        colour_map = self.visualise_output_detailed(pred, np_aruco_img)
        pred = self.convert_pred_to_labels(pred, np_img, im)
        # dt = time.time() - tick
        # print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        return pred, colour_map

    def preprocess_image(self, np_img):
        im = torch.from_numpy(np_img)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        im = im.permute((0, 3, 1, 2))
        
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
        
        height_width_ratio = height/width
        # invalidate pred if :
        if conf < 0.7: # not confident
            return False
        
        if cls == 1: # redapple
            if height_width_ratio < 0.940 or height_width_ratio > 1.300: #1.284 0.958
                return False
        elif cls == 2: # greenapple
            if height_width_ratio < 0.770 or height_width_ratio > 0.980: #0.792, 0.971
                return False
        elif cls == 3: # orange
            if height_width_ratio < 0.909 or height_width_ratio > 1.080: # 1.067 0.981
                return False
        elif cls == 4: # mango
            if height_width_ratio < 0.450 or height_width_ratio > 0.950: #0.476, 0.943
                return False
        elif cls == 5: # capsicum
            if height_width_ratio < 0.999 or height_width_ratio > 1.450: #1.407 1.140
                return False
        else: 
            print('Unknown class')
        
        return True
        
    
    def visualise_output(self, nn_output, np_img):
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

    def visualise_output_detailed(self, pred, np_img):
        colour_map = np_img.copy().astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(np_img.shape[:2], det[:, :4], np_img.shape[:2]).round()

                for *xyxy, conf, cls in reversed(det):
                    if self.validate_pred(xyxy, conf, cls):
                        c = self.colour_code[int(cls)]
                        width = 1.0*int(xyxy[2]) - 1.0*int(xyxy[0])
                        height = 1.0*int(xyxy[3]) - 1.0*int(xyxy[1])
                        # bounding box
                        colour_map = cv2.rectangle(colour_map, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (int(c[0]), int(c[1]), int(c[2])), thickness = 3)
                        # text outline
                        colour_map  = cv2.putText(colour_map, f'{conf:.2f},{height/width:.3f}', (int(xyxy[0]), int(xyxy[1])), font, 0.8, (255,255,255), thickness = 8)
                        # text
                        colour_map  = cv2.putText(colour_map, f'{conf:.2f},{height/width:.3f}', (int(xyxy[0]), int(xyxy[1])), font, 0.8, (int(c[0]), int(c[1]), int(c[2])), thickness = 3)
        colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 160)
        pad = 5
        labels = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map  = cv2.putText(colour_map, labels[i-1],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (255, 255, 255))
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
