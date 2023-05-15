import cv2
import time
import requests
video_capture = cv2.VideoCapture(0)
process_this_frame = True
url='http://127.0.0.1:8000/pridict'
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    start = time.time()
    if process_this_frame: 
        face_locations, face_names, all_eyes, predictions,gadget_output_list=requests.post()

