import cv2
import time
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, make_divisible
from numpy import random
import numpy as np
import os
import pickle 
from settings import GADGET_DETECTION_YOLOV7_WEIGHTS

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess(img0):
    # Padded resize
    img_size = 640
    stride = 32
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img

def detect_from_img(model, img0):
    stride = int(model.stride.max())
    imgsz = 640
    names = model.names
    imgsz = make_divisible(imgsz, stride)
    
    img = preprocess(img0)
    img = torch.from_numpy(img)#.to(device)
    img = img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    conf_thres = 0.5
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    pred = non_max_suppression(pred, conf_thres = conf_thres, iou_thres = iou_thres, classes = classes, agnostic = agnostic_nms)
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        output_list = [(*map(float, xyxy), names[int(cls)], f'{conf:.2f}') for *xyxy, conf, cls in reversed(det)]
    return output_list

#os.chdir("/home/spanidea-168/Documents/SpanIdea_Office_work/Exam_proctoring_project/ai_proctor/yolov7_raw")
#print(os.getcwd())
device = "cpu"
#weights = '/home/spanidea-168/Documents/SpanIdea_Office_work/Exam_proctoring_project/yolov7/weights/gadget_prediction.pt'

model_gd = attempt_load(GADGET_DETECTION_YOLOV7_WEIGHTS, device)
print("Model loaded successfully!!")

# model_obj_filepath = "/home/spanidea-168/Documents/SpanIdea_Office_work/Exam_proctoring_project/yolov7/model.obj"
# filehandler = open(model_obj_filepath, 'rb') 
# object = pickle.load(filehandler)