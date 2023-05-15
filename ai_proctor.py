#Imports
import cvlib as cv
import face_recognition
import cv2
import numpy as np
import sys
import os
import dlib
import pickle5 as pickle
from skimage.feature import hog
import time 
import torch
from numpy import random
from facemodule.faceutils import (
                                get_student_encodings, convert_bbox, detect_eyes,
                                get_face_names, predict
                                )
from settings import REGISTER_DIR_PATH, LANDMARK_PREDICTOR_PATH, WEIGHT_FILE_FRONT_SIDE
#from utils.general import non_max_suppression, scale_coords, make_divisible
#from models.experimental import attempt_load
from gadget_detect import (
                          detect_from_img, model_gd
                          )
import threading
from queue import Queue

student_name=sys.argv[1]
landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
known_face_encodings = get_student_encodings(REGISTER_DIR_PATH)

if len(known_face_encodings)==0:
    print("You need to register first... Please run student_registration.py first!!")
    exit(1)

#---------------------------
known_face_names = [student_name]*len(known_face_encodings)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

model_fs = pickle.load(open(WEIGHT_FILE_FRONT_SIDE, 'rb')) # load the model from disk
#define categories for front/side classification
categories = ['fine' , 'looking away']
predicted_class = None
prob = None
left_eye_idxs = [36, 37, 38, 39, 40, 41]
right_eye_idxs = [42, 43, 44, 45, 46, 47]
font = cv2.FONT_HERSHEY_DUPLEX

#for gadget detection
device = "cpu"
weights = 'all_models/gadget_detection_YOLOV7.pt'
#model_gd = attempt_load(weights, device)
names = model_gd.names
color_codes = {element:tuple([random.randint(0, 255) for i in range(3)]) for element in names}

video_capture = cv2.VideoCapture(1)
process_this_frame = True

def detect_activity(model_fs, frame,queue):
    categories=['fine' , 'looking away']
    face_locations, _ = cv.detect_face(frame,threshold=0.6) 
    face_locations_fr = convert_bbox(face_locations)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
    all_eyes=[]
    all_eyes_wrt_face=[]
    for (x1, y1, x2, y2) in face_locations:
        #rectangle object to be passed for dlib eye detection
            
        face_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2) 
        eyes = detect_eyes(gray, face_rect)
        all_eyes.append(eyes)
            
        #Following are eyes coordinates wrt cropped face
        face_gray_array = gray[y1:y2, x1:x2]
        h,w = face_gray_array.shape
        face_rect_wrt_face = dlib.rectangle(left=0, top=0, right=w, bottom=h)

        eyes_wrt_face = detect_eyes(face_gray_array, face_rect_wrt_face)
        all_eyes_wrt_face.append(eyes_wrt_face)

    face_encodings = face_recognition.face_encodings(frame, face_locations_fr)
    face_names = get_face_names(face_encodings, known_face_encodings, known_face_names)

    predictions = []
    for (left, top, right, bottom),eyes_wrt_face in zip(face_locations,all_eyes_wrt_face):
        face_img = frame[top:bottom,left:right]
        try:
            predicted_class, prob = predict(model_fs,face_img, eyes_wrt_face, categories)
            predictions.append((predicted_class,prob))
        except:
            predictions.append((None,None))

        # try:
        #     predicted_class, prob = predict(model_fs,face_img, eyes_wrt_face, categories)
        # except:
        #     pass
    #return face_locations, face_names, all_eyes, predictions
    queue.put((face_locations, face_names, all_eyes, predictions))
    # print("detect_activity completed execution!!")
    # print((face_locations, face_names, all_eyes, predictions))

def detect_gadget(model_gd,frame, queue):
    try:
        output_list = detect_from_img(model_gd, frame)
    except:
        output_list = []

    #return output_list
    queue.put(output_list)
    # print("detect_gadget completed execution!!")
    # print(output_list)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    start = time.time()
    # Only process every other frame of video to save time
    if process_this_frame: 
        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_small_frame = small_frame[:, :, ::-1]
        #rgb_small_frame = frame[:, :, ::-1]
        #rgb_small_frame = frame

        # Find all the faces and face encodings in the current frame of video
        # face_locations, _ process_this_frame= cv.detect_face(frame,threshold=0.6) 
        # face_locations_fr = convert_bbox(face_locations)
      
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # all_eyes=[]
        # all_eyes_wrt_face=[]
        # for (x1, y1, x2, y2) in face_locations:
        #     #rectangle object to be passed for dlib eye detection
            
        #     face_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2) 
        #     eyes = detect_eyes(gray, face_rect)
        #     all_eyes.append(eyes)
            
        #     #Following are eyes coordinates wrt cropped face
        #     face_gray_array = gray[y1:y2, x1:x2]
        #     h,w = face_gray_array.shape
        #     face_rect_wrt_face = dlib.rectangle(left=0, top=0, right=w, bottom=h)

        #     eyes_wrt_face = detect_eyes(face_gray_array, face_rect_wrt_face)
        #     all_eyes_wrt_face.append(eyes_wrt_face)

        # face_encodings = face_recognition.face_encodings(frame, face_locations_fr)
        # face_names = get_face_names(face_encodings, known_face_encodings, known_face_names)

        #yolov7 prediction
        # try:
        #     output_list = detect_from_img(model_gd, frame)
        # except:
        #     output_list = []
        queue1 = Queue(maxsize=1)
        queue2 = Queue(maxsize=1)
        th1 = threading.Thread(target=detect_activity, args=(model_fs, frame, queue1))
        th2 = threading.Thread(target=detect_gadget, args=(model_gd, frame, queue2))

        # face_locations, face_names, all_eyes, predictions = detect_activity(model_fs, frame, categories=['fine' , 'looking away'])
        # gadget_output_list = detect_gadget(model_gd,frame)
        # print(f"output list from gadget detect: {gadget_output_list}")
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        (face_locations, face_names, all_eyes, predictions) = queue1.get()
        gadget_output_list = queue2.get()
        #wait till we get both the outputs
        # start = time.time()
        # while True:
        #     flag1, flag2 = queue1.empty(), queue2.empty()
        #     if flag1==False and flag2==False:
        #         (face_locations, face_names, all_eyes, predictions) = queue1.get()
        #         gadget_output_list = queue2.get()
        #         end = time.time()
        #         break
        #     else:
        #         pass
        # print(f"{end-start} seconds")
    process_this_frame = not process_this_frame

    end = time.time()
    #print(f"{end-start} seconds")
    # Display the results
    #for (top, right, bottom, left), name in zip(face_locations, face_names):
    #print(f"length:{len(face_locations)},{len(face_names)},{len(eyes)}")
    if student_name not in face_names:
        cv2.putText(frame, 'Warning! Student not in Frame!', (6, 200), font, 0.85, (0, 0, 255), 2)

    # print(f"face_locations:{face_locations}")
    # print(f"face_names:{face_names}")
    # print(f"all_eyes:{all_eyes}")
    # print(f"predictions:{predictions}")
    #for (left, top, right, bottom), name, eyes, eyes_wrt_face, (left_g, top_g, right_g, bottom_g, gadget_name, confidence) in zip(face_locations, face_names, all_eyes, all_eyes_wrt_face, output_list):
    #for (left, top, right, bottom), name, eyes, (predicted_activity_class, prob) in zip(face_locations, face_names, all_eyes, predictions):
    for (left, top, right, bottom), name, eyes, (predicted_activity_class, prob) in zip(face_locations, face_names, all_eyes, predictions):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4
    
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a box around the eyes
        for (x1,y1,x2,y2) in eyes:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

        # face_img = frame[top:bottom,left:right]
        # try:
        #     predicted_class, prob = predict(model_fs,face_img, eyes_wrt_face, categories)
        # except:
        #     pass

        #----------------------------------------------------------------------
        # # Draw a label with a name below the face also write if suspicious activity found
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        
        # if name==student_name and predicted_activity_class=='looking away' and prob>=0.5:
        #     text = f"{predicted_activity_class} {round(prob,2)}%"
        #     #print(predicted_class)
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #     cv2.putText(frame, text, (left , bottom - 240), font, 1.0, (100, 255, 255), 2)

        # elif name== "Unknown":
        #     cv2.putText(frame, 'Warning! Unknown Person found in Frame!', ( 6, 230), font, 0.85, (0, 0, 255), 2)
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # else:
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # for (x1,y1,x2,y2) in eyes:
        #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        #------------------------------------------------------------------------
        if name== "Unknown":
            cv2.putText(frame, 'Warning! Unknown Person found in Frame!', ( 6, 230), font, 0.85, (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if name==student_name and predicted_activity_class=='looking away' and prob>=0.5:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&77")
            text = f"{predicted_activity_class} {round(prob,2)}"
            #print(predicted_class)
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, text, (left , bottom - 240), font, 1.0, (0, 255, 255), 2)



        #------------------------------------------------------------------------
        if len(gadget_output_list)>0:
            for (left_g, top_g, right_g, bottom_g, gadget_name, confidence) in gadget_output_list:
                #print(f"output list from gadget detect:{(left_g, top_g, right_g, bottom_g, gadget_name, confidence)}")
                left_g, top_g, right_g, bottom_g = int(left_g), int(top_g), int(right_g), int(bottom_g)
                cv2.rectangle(frame, (left_g, top_g), (right_g, bottom_g), color_codes[gadget_name], 3)
                conf = gadget_name + " " + confidence
                
                if top-30 <=0:
                    print_loc = (left_g, bottom_g + 10)
                else:
                    print_loc = (left_g, top_g - 10)

                cv2.putText(frame, conf, print_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)#cv2.resize(frame, (500, 500)))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam 
video_capture.release() 
cv2.destroyAllWindows()       