import argparse
import os
import pickle
import platform
import shutil
import socket
import struct
import time
import json
from pathlib import Path

import numpy as np
from numpy import random

import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from conf_thresh import confidence_threshold
from bounding_box import check_bounding_box
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


def receive_frame(data, payload_size, conn):
    # image
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # coordinates
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    coor_data = data[:msg_size]
    data = data[msg_size:]


    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    coor = pickle.loads(coor_data, fix_imports=True, encoding="bytes")
    coor = json.loads(coor)
    return frame, coor, data


# need to confirm on communication protocol later
def convert_to_message(label):
    prefix = ''
    label = prefix + label
    return label.encode()


def detect(weights='mdp/weights/weights.pt',
           img_size=416,
           conf_thres=0.01,
           iou_thres=0.5,
           device='',
           classes=None,
           agnostic_nms=False,
           augment=False,
           update=False,
           scale_percent=100):
    # Initialize
    set_logging()
    device = select_device(device)
    csvfile = open('predictions.csv', 'w+')
    csvfile.write('robot_x,robot_y,robot_dir,img_label,img_x,img_y\n')
    csvfile.close()

    imgstr = open('image_string.txt', 'w+')
    imgstr.write(' ')
    imgstr.close()

    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # tcp connection
    # HOST = ''
    # PORT = 8080
    # s_algo = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # print('Socket created')
    # s_algo.bind((HOST, PORT))
    # print('Socket bind complete')
    # s_algo.listen(10)
    # print('Socket now listening')
    # conn_algo, addr_algo = s_algo.accept()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.15.1', 8008)) # 192.168.15.1
    connection = client_socket.makefile('wb')
    conn_rpi = client_socket

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    # frame, data = receive_frame(data, payload_size, conn_rpi)
    # img = [frame]

    # s = np.stack([letterbox(x, new_shape=img_size)[0].shape for x in img], 0)  # inference shapes
    # rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

    row_num = 0
    while True:
        frame, coor, data = receive_frame(data, payload_size, conn_rpi)
        # print("Received picture at {}, {} facing {}".format(coor['X'], coor['Y'], coor['O']))

        img = [frame]
        img0 = img.copy()

        s = np.stack([letterbox(x, new_shape=img_size)[0].shape for x in img], 0)  # inference shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

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
                    if predicted_label:
                        if not image_seen[predicted_label]:
                            label_id = label_id_mapping.get(predicted_label)
                            if conf < confidence_threshold(label_id):  # fine tune for up arrow (white)
                                break
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            good, text = check_bounding_box(xywh, im0.shape[0], im0.shape[1])
                            if not good:
                                break

                            print(('%s ' * 5 + '\n') % (label_id, *xywh))  # label format
                            image_seen[predicted_label] = True

                            # determine image position
                            x, y, w, h = xywh
                            # conn_algo.sendall(bytes(json.dumps({'label': label_id, 'x': x, 'y': y}), 'utf-8'))  # send result to algo
                            print(json.dumps({'label': label_id, 'x': x, 'y': y}))
                            csvfile = open('predictions.csv', 'a+')
                            csvfile.write('{},{},{},{},{},{}\n'.format(coor['X'],coor['Y'],coor['O'],label_id,x,y))
                            csvfile.close()

                            imgstr = open('image_string.txt', 'a+')
                            str_x, str_y = coor['X'], coor['Y']
                            if coor['O'] == 'Right':
                                str_y -= 2
                            elif coor['O'] == 'Up':
                                str_x += 2
                            elif coor['O'] == 'Left':
                                str_y += 2
                            elif coor['O'] == 'Down':
                                str_x -= 2
                            imgstr.write('({}, {}, {}), '.format(label_id, str_x, str_y))
                            imgstr.close()

                            label = '%s %.2f' % (label_id, conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            #percent by which the image is resized
                            # scale_percent = 50
                            #calculate the 50 percent of original dimensions
                            width = int(im0.shape[1] * scale_percent / 100)
                            height = int(im0.shape[0] * scale_percent / 100)
                            # dsize
                            dsize = (width, height)
                            # resize image
                            im0 = cv2.resize(im0, dsize)

                            # detected_images.append(im0)
                            row_num = append_image(im0, row_num)

                            def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                                w_min = min(im.shape[1] for im in im_list)
                                im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                                                  for im in im_list]
                                return cv2.vconcat(im_list_resize)

                            def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                                h_min = min(im.shape[0] for im in im_list)
                                im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                                              for im in im_list]
                                return cv2.hconcat(im_list_resize)

                            def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
                                im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
                                return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

                            im_tile = concat_tile_resize(detected_images)

                            cv2.imshow('ImageWindow', im_tile)
                            cv2.imwrite('result.png', im_tile)
                            break
                    else:
                        print("no prediction")
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


if __name__ == '__main__':
    detect()
