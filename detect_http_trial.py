import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import requests

from conf_thresh import confidence_threshold
from bounding_box import check_bounding_box
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


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

image_seen = {
    'up': False,
    'down': False,
    'right': False,
    'left': False,
    'go': False,
    '6': False,
    '7': False,
    '8': False,
    '9': False,
    '0': False,
    'v': False,
    'w': False,
    'x': False,
    'y': False,
    'z': False
}

detected_images = [[]]
def append_image(img, row_num):
    if len(detected_images[row_num]) < 3:
        detected_images[row_num].append(img)
    else:
        row_num += 1
        detected_images.append([])
        detected_images[row_num].append(img)
    return row_num


def detect(weights='mdp/weights/weights.pt',
           source_address='http://localhost:8008', # 192.168.15.1
           img_size=416,
           conf_thres=0.01,
           iou_thres=0.5,
           device='',
           classes=None,
           agnostic_nms=False,
           augment=False,
           update=False,
           scale_percent=50):
    source = source_address + '/stream.mjpg'
    label_server = source_address + '/labels'

    predicted_label = None
    imgsz = img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    row_num = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

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
                    if predicted_label:
                        label_id = label_id_mapping.get(predicted_label)
                        if conf < confidence_threshold(label_id):  # fine tune for up arrow (white)
                            break
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        print(('%s ' * 5 + '\n') % (label_id, *xywh))  # label format

                        if not image_seen[predicted_label]:
                            # determine image position
                            x, y, w, h = xywh
                            r = requests.post(label_server, json={'label': label_id, 'x': x, 'y': y})  # send result to rpi
                            print(r.text)
                            image_seen[predicted_label] = True

                        label = '%s %.2f' % (label_id, conf)
                        good, text = check_bounding_box(xywh, im0.shape[0], im0.shape[1])
                        if not good:
                            label = text
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        break
            cv2.imshow(p, im0)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


if __name__ == '__main__':
    detect()
