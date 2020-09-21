import argparse
import os
import pickle
import platform
import shutil
import socket
import struct
import time
from pathlib import Path

import numpy as np
from numpy import random

import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, letterbox
from utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)
from utils.torch_utils import load_classifier, select_device, time_synchronized


label_id_mapping = {
    'up': '1',
    'down': '2',
    'right': '3',
    'left': '4',
    'go': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '0': '10',
    'v': '11',
    'w': '12',
    'x': '13',
    'y': '14',
    'z': '15'
}


# need to confirm on communication protocol later
def convert_to_message(label):
    prefix = ''
    label = prefix + label
    return label.encode()


def detect(weights='mdp/weights/weights.pt',
           img_size=416,
           conf_thres=0.7,
           iou_thres=0.5,
           device='',
           classes=None,
           agnostic_nms=False,
           augment=False,
           update=False):
    # Initialize
    set_logging()
    device = select_device(device)

    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    s = np.stack([letterbox(x, new_shape=img_size)[0].shape for x in img], 0)  # inference shapes
    rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

    #open stream connection via http
    cap = cv2.VideoCapture('http:/192.168.15.1:8000/stream.mjpg')

    while True:
        ret, frame = cap.read()

        img = [frame]
        img0 = img.copy()

        # Letterbox
        img = [letterbox(x, new_shape=img_size, auto=rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', img0[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    predicted_label = names[int(cls)]
                    label_id = label_id_mapping.get(predicted_label)

                    rpi_message = convert_to_message(label_id)

                    #conn.sendall(rpi_message)  # send result to rpi
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    print(('%s ' * 5 + '\n') % (label_id, *xywh))  # label format

                    label = '%s %.2f' % (label_id, conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    break
            cv2.imshow('ImageWindow', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        # time.sleep(0.1)


if __name__ == '__main__':
    detect()
