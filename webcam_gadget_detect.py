import cv2
import time
import torch
# from gadgetmodule.models.experimental import attempt_load
# from gadgetmodule.utils.general import non_max_suppression, scale_coords, make_divisible
from numpy import random
#import numpy as np
#from gadget_detect import detect_from_img, model
from gadget_detect import detect_from_img, model_gd

# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)

# def preprocess(img0):
#     # Padded resize
#     img_size = 640
#     stride = 32
#     img = letterbox(img0, img_size, stride=stride)[0]
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#     return img

# def detect_from_img(model, img0):
#     stride = int(model.stride.max())
#     imgsz = 640
#     names = model.names
#     imgsz = make_divisible(imgsz, stride)
    
#     img = preprocess(img0)
#     img = torch.from_numpy(img).to(device)
#     img = img.float()
#     img /= 255.0

#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     with torch.no_grad():
#         pred = model(img)[0]

#     conf_thres = 0.5
#     iou_thres = 0.45
#     classes = None
#     agnostic_nms = False
#     pred = non_max_suppression(pred, conf_thres = conf_thres, iou_thres = iou_thres, classes = classes, agnostic = agnostic_nms)
    
#     for det in pred:
#         if len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

#         output_list = [(*map(float, xyxy), names[int(cls)], f'{conf:.2f}') for *xyxy, conf, cls in reversed(det)]
#     return output_list

# device = "cpu"
# weights = '/home/spanidea-168/Documents/SpanIdea_Office_work/Exam_proctoring_project/yolov7/weights/gadget_prediction.pt'
# model = attempt_load(weights, device)
names = model_gd.names
color_codes = {element:tuple([random.randint(0, 255) for i in range(3)]) for element in names}

# img_path = "/home/spanidea-168/Documents/SpanIdea_Office_work/dataset_proctoring/dataset_manali/gadget_person/2022-07-14-171557.jpg"
# img = cv2.imread(img_path)



# initial_timer = time.time()
#output_list = detect_from_img(model, img)

# current_timer = time.time()
# print(current_timer - initial_timer)
    
video_capture = cv2.VideoCapture(0)
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    start = time.time()

    # Only process every other frame of video to save time
    if process_this_frame:
        output_list = detect_from_img(model_gd, frame)

    process_this_frame = not process_this_frame

    end = time.time()
    print(f"{end-start} seconds")

    for (left, top, right, bottom, gadget_name, confidence) in output_list:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(frame, (left, top), (right, bottom), color_codes[gadget_name], 3)
        conf = gadget_name + " " + confidence
        
        if top-30 <=0:
            print_loc = (left, bottom + 10)
        else:
            print_loc = (left, top - 10)

        cv2.putText(frame, conf, print_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam 
video_capture.release() 
cv2.destroyAllWindows()   