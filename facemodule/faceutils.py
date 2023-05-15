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
from settings import LANDMARK_PREDICTOR_PATH

#landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
left_eye_idxs = [36, 37, 38, 39, 40, 41]
right_eye_idxs = [42, 43, 44, 45, 46, 47]

def get_eyes_array(face_img, eyes, target_size=(70,70)):
    """
    Input:
    face_img(numpy array of size (h,w,3)): cropped face image
    eyes(list of tuples): eyes[i]=(ex1,ey1,ex2,ey2) where ex1=x-coordinate of top-left corner of bbox
                                                          ey1=y-coordinate of top-left corner of bbox
                                                          ex2=x-coordinate of bottom-right corner of bbox
                                                          ey2=y-coordinate of bottom-right corner of bbox
                          w.r.t cropped face image
    target_size(tuple): default:- (70,70) | for training SVM classifier for front/side this is kept for (40,40)

    Output:
    eye_arr(list of numpy arrays): 
    hog features of eyes bbox given in the input
    """
    eyes_arrays=[]
    for (ex1,ey1,ex2,ey2) in eyes:
        #cv2.rectangle(img_resized,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #print(ex1,ey1,ex2,ey2)
        eye=face_img[ey1:ey2,ex1:ex2]
        #print(eye.shape)
        h, w, _ = eye.shape
        
        if h>0 and w>0:
            eye_resized=cv2.resize(eye,target_size)

            #eye_array=eye_resized.flatten()
            eye_fd, _ = hog(eye_resized, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)    #(32,)

            #print(eye_fd.shape)
            eyes_arrays.append(eye_fd)
    
    if len(eyes_arrays)==0:
        eye_arr=np.concatenate((np.zeros(32),np.zeros(32)))
    elif len(eyes_arrays)==1:
        eye_arr=np.concatenate((eyes_arrays[0],np.zeros(32)))
    elif len(eyes_arrays)>=2:
        eye_arr=np.concatenate((eyes_arrays[0],eyes_arrays[1]))
        #print(eye_arr.shape)
    return eye_arr                   #(64,)

def shape_to_np(shape, dtype="int"):
    """
    Input:
    shape(object): this is the object returned by dlib.shape_predictor
                    this object contains 68 facial landmarks.
    Output:
    coords(numpy array): this is numpy array of shape (68,2) which contains
                         (x,y) coordinate of 68 landmarks
    """
	#initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
    return coords

def get_eye_bbox(shape,side):
    """
    Input:
    shape(numpy array): this array is of shape (68,2) i.e., (x,y) coordinates 
                         of 68 facial landmarks.
    side(list): this is list with the indices corresponding to left/right eye
    """
    eye_coord = shape[side]
    xs = eye_coord[:,0]
    ys = eye_coord[:,1]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return left-15, top-25, right+15, bottom+25

def detect_eyes(gray_img,face_rect_obj):
    """
    Input:
    gray_img(2D numpy array): gray scale image 
    face_rect_obj(instance of object dlib.rectangle): bounding box cordinates 
    of face in gray_img as dlib.rectangle object.

    Output:
    eyes(list of len=2): eyes[0], eye[1] are tuples with top, left, bottom, right 
                         coordinates of left and right eye respectively.

    """
    shape = landmark_predictor(gray_img, face_rect_obj)
    shape_np = shape_to_np(shape)

    try:
        left_eye = get_eye_bbox(shape_np, left_eye_idxs)
    except:
        left_eye = ()

    try:
        right_eye = get_eye_bbox(shape_np, right_eye_idxs)
    except:
        right_eye = ()

    eyes = [left_eye, right_eye]
    return eyes

def convert_bbox(face_loc_list):
    """
    Input:
    face_loc_list(list): list of tuples with bounding box coordinates in 
                         (left, top, right, bottom) format

    Output(list): list of tuples with bounding box coordinates in
                  (top, right, bottom, left) format

    This function converts the format of bounding box coordinates in the format
    required by face_recognition library
    """
    face_locations_fr=[]
    for (x1, y1, x2, y2) in face_loc_list:
        face_locations_fr.append((y1, x2, y2, x1 ))

    return face_locations_fr

def get_student_encodings(img_dir):
    """
    Input:
    img_dir(string): path of directory where registered images of a student is stored

    Output:
    known_face_encodings(list): list of embeddings for face in all images in the given
                                directory.

    This function gets the embeddings of face in all images in a given directory.
    """
    student_img_filenames=os.listdir(img_dir)
    known_face_encodings=[]
    for filename in student_img_filenames:
        try:
            student_img_filepath=os.path.join(img_dir,filename)
            student_image = cv2.imread(student_img_filepath)
            face_locations, _ = cv.detect_face(student_image,threshold=0.6) 
            face_locations_fr = convert_bbox(face_locations)
            
            student_face_encoding = face_recognition.face_encodings(student_image,face_locations_fr)[0]
            known_face_encodings.append(student_face_encoding)
        except:
            pass

    return known_face_encodings

def get_face_names(face_encodings, known_face_encodings, known_face_names):
        """
        Input:
        face_encodings(list): embeddings of all faces detected in a test image
        known_face_encodings(list): embeddings of known face captured during student
                                    registration
        known_face_names(list): names of known face captured during student
                                registration

        Output:
        face_names(list): predicted names of a detected faces

        This function compares the face embeddings from test image with all the known
        face embeddings and predicts the name corresponding to each face embedding 
        """
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            min_distance=np.min(face_distances)

            if min_distance<=0.47:
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
            #print(f"distance ={min_distance} , name={name}")

            face_names.append(name)
        return face_names

def predict(model,face_array,eyes,categories):
    """
    Input:
    model(instance of object): this is instance of object for svm classifier
                               which is trained for classification task between 
                               categories: looking front, looking away
    face_array(3D numpy array): rgb image of cropped face 
    eyes(list): list of tuples containing bounding box coordinates of left and
                right eye w.r.t cropped face image
    categories(list): list of names of categories i.e, string here, ['fine','looking away']
    """
    img_resized=cv2.resize(face_array,(200,200))
        
    eye_arr=get_eyes_array(face_array, eyes, target_size=(40,40))

    fd, _ = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    data_arr=np.concatenate((fd,eye_arr))
    data_arr=data_arr.reshape(1,-1)
    
    probability=model.predict_proba(data_arr)
    predicted_class=model.predict(data_arr)
    predicted_class=predicted_class[0]
    return categories[predicted_class] , probability[0][predicted_class]