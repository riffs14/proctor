#import needed packages

import os
import cv2
from skimage.feature import hog
import numpy as np
import pandas as pd
import pickle
import time
import dlib
import cvlib as cv
from playsound import playsound
from threading import Thread
from settings import WEIGHT_FILE_REGISTRATION, REGISTER_DIR_PATH

def predict_label(model, img_array):
    """
    Input: 
    model: model object
    img_array: input image

    Output:
    category(string): predicted category
    probability(float): confidence probability

    custom built python function to classify the image
    """

    img_resized = cv2.resize(img_array, (200, 200))

    fd, _ = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    
    probability = model.predict_proba([fd])
    predicted_class = model.predict([fd])
    predicted_class = predicted_class[0]
    return categories[predicted_class], probability[0][predicted_class]

#clean the folder which saves the images before using the code
if not os.path.exists(REGISTER_DIR_PATH):
    os.mkdir(REGISTER_DIR_PATH)
files=os.listdir(REGISTER_DIR_PATH)
for file in files:
    filepath=os.path.join(REGISTER_DIR_PATH,file)
    os.remove(filepath)

#load the pre trained model from files
loaded_model = pickle.load(open(WEIGHT_FILE_REGISTRATION, "rb"))

#this is the sequence in which the files are stored in the file directory(the y is also trained in the same sequence)
categories = ["down", "front", "left", "right", "up"]

#this is the sequence in which the code will check the face directions
instruction_text = ["front", "right", "left", "up", "down"]

#initialization of some timer and boolean flag variables
initial_timer, initial_2_timer, current_capture, idx = time.time(), time.time(), 0, 0  
phase_start, s_pressed, restart, first_message_said, last_message_said, press_s_message = (False, ) * 6

#code to use the webcam camera
cap = cv2.VideoCapture(2)

#while loop which will run through the whole duration of the code
while time.time() - initial_2_timer < 120:
    
    #code cecks that the very first message is said already or not, if it already said then avoid saying it and if not then say it once
    if not press_s_message:
        t1 = Thread(target=playsound, args=("voice_messages/start_registration.mp3",))
        t1.start()
        press_s_message = True
    
    #using try except to check if there are any people on screen or not as no people on the screen will result in an error
    try:

        #reading the frame from the image capture and copying it as the frame will be manipulated many times
        _, actual_frame = cap.read()
        frame = actual_frame.copy()
         
        #variable to read the current time and then start a timer
        current_timer = time.time()
        time_difference = current_timer - initial_timer

        #need to press s key to proceed
        k = cv2.waitKey(1)
        if k == ord("s"):
        
            #code to avoid the message being said multiple times
            if first_message_said == False:
        
                #code to say first message to look front
                t1 = Thread(target=playsound, args = ("voice_messages/front.mp3",))
                t1.start()
                first_message_said = True
            s_pressed = True
        
        #state before pressing s key is noting but text on the screen
        if not s_pressed:
            cv2.putText(frame, "Press s to start registration", (30, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
       
        #after s is pressed then this code will start
        if s_pressed:
        
            #the program will close if the code detects more than 1 people in frame
            #for that the value of restart variable will be changed to boolean True
            if restart:

                #when the code will close, it will first give some warnings and tell that the program will close
                cv2.putText(frame, "More than 1 person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please restart the registration process", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                #the reset timer is given the value of 10 seconds, so the code will be closed after 10 seconds
                if restart_timer - time.time() <= 0:
                    break 
            
            #else if there is no need to close the code then the code will work as intended
            else:
                
                #check if all the face directions are captured or not 
                #and avoid running it again after current_capture = 5 as it will cause index error
                if current_capture < 5:

                        #save only every 10th picture
                        #note that the actual frame is being saved and not the copy
                        if idx %10 == 0:
                            cv2.imwrite(f'{REGISTER_DIR_PATH}/student_img{idx}.jpg',actual_frame) 
                        idx += 1

                        #detect all the faces from the image
                        face_locations, _ = cv.detect_face(frame, threshold = 0.5)
                        
                        #code should only work if there is 1 person in the frame
                        if len(face_locations) == 1:       

                                #crop the face from the picture
                                left, top, right, bottom = face_locations[0]
                                cropped_image = frame[top:bottom, left:right]

                                #predict which direction the person is facing using the model we loaded earlier
                                label, probability = predict_label(loaded_model, cropped_image)

                                #check if there is only 1 person on the frame and check if the phase has started or not
                                #phase means the time period which starts once when the user starts facing the needed direction
                                #its duration is of 3 seconds if the person is facing continuously in the intended direction
                                #if the person looks in different direction for once during the phase the phase restarts
                                if not phase_start and len(face_locations) == 1:

                                    #the first 3 seconds for every direction is idle time
                                    #then after that if the person looks in the needed direction then the phase starts
                                    if time_difference > 3:

                                        if label == instruction_text[current_capture]:
                                            initial_timer = time.time()
                                            phase_start = True
                                #this is the code when the detection phase starts
                                #here the user has to look in the needed direction continuously for 3 seconds
                                #if he does not then the phase timer will restart 
                                else:
                                    if time_difference < 2 and label != instruction_text[current_capture]:
                                        initial_timer = time.time()
                                    
                                    #now that the timer has reached 3 seconds means that the person followed the instructions properly
                                    elif time_difference >= 3:

                                        #check again if the current_capture is less than 4 to avoid any errors
                                        if current_capture < 4:

                                            #now play the instruction for the next direction
                                            t1 = Thread(target=playsound, args = ("voice_messages/" + instruction_text[current_capture + 1] + ".mp3", ))
                                            t1.start()

                                        #after that the value of current_capture will increase by 1 
                                        #now the code will detect the next direction from the next iteration
                                        current_capture += 1

                                        #the detection phase is now over and will activate again for the next direction from next iteration
                                        phase_start = False

                                #this text will be printed on the screen throughout the duration of the normal run of the code
                                #normal run = no second person on screen and person is not missing from the screen
                                cv2.putText(frame, "Please look " + instruction_text[min(current_capture, 4)], (30, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                        
                        #this is an extra code to make sure that face not found message is printed on the screen even if there is no exception trigger
                        elif len(face_locations) == 0:
                            cv2.putText(frame, "Face not found", (30, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

                        #if there are more than 1 person on the screen then the value of restart boolean will become True
                        #and then in the next iteration the will detect that restart is True and will shut the code after 10 seconds and giving warning
                        elif len(face_locations) > 1:
                            restart_timer = 10 + time.time()
                            restart = True
                                   
                #now this is te part where the person has followed all the instructions
                else:

                    #this is the code to avoid the final message being played infinitely
                    if not last_message_said:
                        t1 = Thread(target = playsound, args = ("voice_messages/done.mp3",))
                        t1.start()
                        last_message_said = True

                    #print on the screen to close the window
                    #cv2.putText(frame, "DONE, Long press q to close", (30, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                    
        #this is the part where the altered frame will be shown on the screen
        cv2.imshow("Frame", cv2.resize(frame, (1000, 800))) 

        #press q to end the program
        if cv2.waitKey(1) & 0xFF == ord("q"):
           break

        if last_message_said:
            break

    #here if there is no person on the screen then the code will throw an error
    #as the code in the try block assumes that the face list is not empty
    #then if the length of the face list is 0 then the code will throw an error and the except block will be triggered which says no face found
    except:
        cv2.putText(frame, "Face not found", (30, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

#after ending the loop, close all the opencv windows and close the video capture
cv2.destroyAllWindows()
cap.release()

# #print(restart)
# if not restart:
#     os.system("python3 ai_proctor.py YOU")