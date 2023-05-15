from fastapi import FastAPI,Form
from pydantic import BaseModel, Field
import cvlib as cv
import face_recognition
from fastapi import FastAPI, File, UploadFile, Request
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import os
import dlib
import pickle5 as pickle
from skimage.feature import hog
import time 
import io
from starlette.responses import StreamingResponse
import torch
from typing import List
from numpy import random
import os
import cv2
from skimage.feature import hog
import numpy as np
import pandas as pd
import pickle
import time
import dlib
import cvlib as cv
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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
landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
global known_face_encodings
known_face_encodings = get_student_encodings(REGISTER_DIR_PATH)
#known_face_names = [student_name]*len(known_face_encodings)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
WEIGHT_FILE_FRONT_SIDE='/home/spanidea/Documents/aicv_proctor1/aicv_proctor/saved/a.sav'
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

video_capture = cv2.VideoCapture(0)
process_this_frame = True


def train(dir,saved):
    categories = ["down", "front", "left", "right", "up"]
    flat_data_arr = []
    target_arr = []
    datadir =dir# "C:\\Users\\nites\\OneDrive\\Desktop\\proctoring_software\\Dataset_for_student_registration"
    for i in categories:
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            h, w, _ = img_array.shape
            resized_img = cv2.resize(img_array, (200, 200))
            fd, _ = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            flat_data_arr.append(fd)
            target_arr.append(categories.index(i))
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    x = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, stratify = y)
    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        
        filepath = os.path.join(saved,'a.sav')#"C:\\Users\\nites\\OneDrive\\Desktop\\proctoring_software\\model_new_no_normalization.sav"
        pickle.dump(model, open(filepath, "wb"))
        return {"Accuracy score": accuracy_score(y_pred, y_test) * 100}
    except Exception as e:
        print(e)
        return False


def detect_activity(model_fs, frame,queue,known_face_names):
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
    #queue.put((face_locations, face_names, all_eyes, predictions))
    return (face_locations, face_names, all_eyes, predictions)
    # print("detect_activity completed execution!!")
    # print((face_locations, face_names, all_eyes, predictions))

def detect_gadget(model_gd,frame, queue):
    try:
        output_list = detect_from_img(model_gd, frame)
    except:
        output_list = []

    #return output_list
    queue.put(output_list)
    return output_list
    # print("detect_gadget completed execution!!")
    # print(output_list)


#from app.model.model import predict_pipeline
app = FastAPI()
model_version="start here !!"

class Frame(BaseModel):
   fame:list=[]
   student_name:str="you"
   



@app.get("/ ") 
def check_registered():
    #print(known_face_encodings)
    known_face_encodings = get_student_encodings(REGISTER_DIR_PATH)
    if len(known_face_encodings)==0:
        return False
        #exit(1)
    else:
        return True


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/pridict")
async def predict(student_name:str=Form(...),file: UploadFile = File(...)):
    #aa={}
    # for i in f:
    #     aa[i[0]]=i[1]
    # student_name='you'
    res={}
    f_name=file.filename
    start=time.time()
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    
    known_face_names = [student_name]*len(known_face_encodings)
    # print("************************************")
    # print(known_face_names)
    # #frame=np.array(aa['fame'])
    #frame=cv2.imread('/home/spanidea/Documents/aicv_proctor1/aicv_proctor/student/student_img10.jpg')
    # print(frame)
    # #if process_this_frame: 
    queue1 = Queue(maxsize=1)
    queue2 = Queue(maxsize=1)
    # # th1 = threading.Thread(target=detect_activity, args=(model_fs, frame, queue1,known_face_names))
    # # th2 = threading.Thread(target=detect_gadget, args=(model_gd, frame, queue2))
    # # th1.start()
    # # th2.start()
    # # th1.join()
    # # th2.join()
    # # (face_locations, face_names, all_eyes, predictions) = queue1.get()
    # # gadget_output_list = queue2.get()
    (face_locations, face_names, all_eyes, predictions) = detect_activity(model_fs, frame, queue1,known_face_names)
    gadget_output_list = detect_gadget(model_gd, frame, queue2)
    # #process_this_frame = not process_this_frame
    # print(face_locations)
    # print(all_eyes)
    # print(predictions)
    # print(face_locations)

    di={
        'face_location':face_locations,
        'face_names':face_names,
        'all_eyes':all_eyes,
        'predictions':predictions
    }
    #print(di)
    # end = time.time()

    if student_name not in face_names:
        cv2.putText(frame, 'Warning! Student not in Frame!', (6, 200), font, 0.85, (0, 0, 255), 2)
        res['Not_in_Frame']=True
    else:
        res['Not_in_Frame']=False
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
        res['Name']=name
        if name==student_name and predicted_activity_class=='looking away' and prob>=0.5:
            print("LOL&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            text = f"{predicted_activity_class} {round(prob,2)}"
            #print(predicted_class)
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, text, (left , bottom - 240), font, 1.0, (0, 255, 255), 2)
            res['activity']='looking away'
        else:
            res['activity']='Okay'


        #------------------------------------------------------------------------
        g=[]
        if len(gadget_output_list)>0:
            
            for (left_g, top_g, right_g, bottom_g, gadget_name, confidence) in gadget_output_list:
                #print(f"output list from gadget detect:{(left_g, top_g, right_g, bottom_g, gadget_name, confidence)}")
                left_g, top_g, right_g, bottom_g = int(left_g), int(top_g), int(right_g), int(bottom_g)
                cv2.rectangle(frame, (left_g, top_g), (right_g, bottom_g), color_codes[gadget_name], 3)
                conf = gadget_name + " " + confidence
                g.append(gadget_name)
                
                if top-30 <=0:
                    print_loc = (left_g, bottom_g + 10)
                else:
                    print_loc = (left_g, top_g - 10)

                cv2.putText(frame, conf, print_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        res['gadgets']=g

    # return (face_locations, face_names, all_eyes, predictions,gadget_output_list)
    cv2.imwrite(os.path.join('output',f_name),frame)
    res, im_png = cv2.imencode(".png", frame)
    end=time.time()-start
    print(end)
    print(res)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    #return 0
    
    #return {"health_check": "OK", "model_version": model_version}

class Student(BaseModel):
   id: int
   name :str = Field(None, title="name of student", max_length=10)
   subjects: list=[]

@app.post("/check_arguments")
def ss(student:Student):

    t=""

    # for i in student['subjects']:
    #     t+=str(type(i))
    # for i in student:
    #     print(i)
    a={}
    for i in student:
        #print(i)
        a[i[0]]=i[1]
        #print(type(i))
    print(a['subjects'])

    return a
    #return{'frame':t,'student_name':str(type(student['name']))}

@app.post("/check_frame")
def dd(f:Frame):
    a={}
    for i in f:
        #print(i)
        a[i[0]]=i[1]
        #print(type(i))

@app.post("/upload_img")
def dd( file: UploadFile = File(...)):
    print(file)
    cv2img=cv2.imread('/home/spanidea/Documents/aicv_proctor1/aicv_proctor/student/student_img10.jpg')
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


class Analyzer(BaseModel):
    filename: str
    img_dimensions: str
    encoded_img: str

@app.post("/analyze")
async def analyze_route(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_dimensions = str(img.shape)
    print(type(img))

    #return_img = processImage(img)

    # line that fixed it
    #_, encoded_img = cv2.imencode('.PNG', return_img)

    #encoded_img = base64.b64encode(encoded_img)

    return 0


# from fastapi import File, UploadFile
import shutil
import os
@app.get("/registration_start")
def registration_start():
    dir='student/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    files=os.listdir(dir)
    try:
        for i in files:
            os.remove(dir+i)
        return 1
    except Exception as e:
        print(e)
        return 0
    
    
        
@app.post("/uploadFile")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
    return {"message": f"Successfully uploaded {file.filename}"}


from fastapi import File, UploadFile
from typing import List

@app.post("/upload_list")
def upload_list(files: List[UploadFile] = File(...)):
    for file in files:
        try:
            contents = file.file.read()
            with open(os.path.join(REGISTER_DIR_PATH,file.filename), 'wb') as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    return {"message": f"Successfuly uploaded {[file.filename for file in files]}"} 




@app.post("/train_svm")
def upload_list(dir_path:str,saved_path:str):
    t=train(dir_path,saved_path)
    
    return t



    #return {"message": f"Successfuly uploaded {[file.filename for file in files]}"} 