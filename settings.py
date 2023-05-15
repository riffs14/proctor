import os
#from datetime import *
from os.path import join, dirname
from dotenv import load_dotenv, find_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(find_dotenv(), override=True)
ENV = os.environ

REGISTER_DIR_PATH = "student" #directory path where images during registration will be saved
LANDMARK_PREDICTOR_PATH = "all_models/shape_predictor_68_face_landmarks.dat"#file pat of .dat file i.e,shape_predictor_68_face_landmarks for landmark prediction
WEIGHT_FILE_FRONT_SIDE = "all_models/SVM_model_during_exam.sav" #weight file for front-side classification during examination
GADGET_DETECTION_YOLOV7_WEIGHTS = "all_models/gadget_detection_YOLOV7.pt" #weight file for yolov7 gadget detection model
WEIGHT_FILE_REGISTRATION = "all_models/SVM_model_during_registration.sav" #weight file for SVM model for classification betweeen front/left/right/up/down during registrationf                 