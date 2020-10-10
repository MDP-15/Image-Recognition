import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import json

# algo connection
HOST = ''
PORT = 9000
s1 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
s1.bind((HOST, PORT))
print('Socket bind complete')
s1.listen(10)
print('Socket now listening')
conn_algo, addr_algo = s1.accept()

# image connection
HOST = ''
PORT = 8008
s2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
s2.bind((HOST, PORT))
print('Socket bind complete')
s2.listen(10)
print('Socket now listening')
conn_image, addr_rpi = s2.accept()

cam = cv2.VideoCapture(0)

cam.set(3, 320);
cam.set(4, 240);

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    recv_data = conn_algo.recv(1024)
    if recv_data:
        recv_data = recv_data.decode()
        try:
            data_dict = json.loads(recv_data)
            if isinstance(data_dict, dict) and 'x' in data_dict and 'y' in data_dict and 'dir' in data_dict:
                # image
                ret, frame = cam.read()  # take a picture using laptop camera
                result, frame = cv2.imencode('.jpg', frame, encode_param)
                image_data = pickle.dumps(frame, 0)
                size = len(image_data)
                image_data = struct.pack(">L", size) + image_data

                # coordinates
                data_dict = json.dumps(data_dict)
                coor_data = pickle.dumps(data_dict, 0)
                size = len(coor_data)
                coor_data = struct.pack(">L", size) + coor_data

                conn_image.sendall(image_data + coor_data)
        except Exception:
            pass




cam.release()