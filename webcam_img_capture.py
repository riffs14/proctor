from settings import REGISTER_DIR_PATH
import cv2
import time
import os

start_text="Press q to start registration..."
print(start_text)
# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
#folder_path="student"

folder_path=REGISTER_DIR_PATH
files=os.listdir(folder_path)
for file in files:
    filepath=os.path.join(folder_path,file)
    os.remove(filepath)
    
TIMER = int(15)
  
# Open the camera
cap = cv2.VideoCapture(0)
instruction_text=['Look front','Look left','Look right','Look up','Look down','Done!!']
idx=0
idx1=0
text = f"Registering student... \n {instruction_text[idx1]}"
y0, dy = 200, 50

while TIMER>0:#True: idx1<len(instruction_text): 
    # Read and display each frame
    ret, img = cap.read()
    img1=img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1, start_text, (40, img1.shape[0]-200), font, 1, (0,255,255),4,cv2.LINE_AA)
    cv2.imshow('Student Registration', img1)

    # check for the key pressed
    k = cv2.waitKey(125)
 
    # set the key for the countdown
    # to begin. Here we set q
    # if key pressed is q
    

    if k == ord('q'):
        print("Registering Student images...")
        prev = time.time()
        prev_time=time.time()
        prev_time1=time.time()
        while TIMER >= 0:
            ret, img = cap.read()
            current_time=time.time()
            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            
            img1=img.copy()
        
            text = f"Registering student... \n {instruction_text[idx1]} \n TIMER:{TIMER}"
            for i, line in enumerate(text.split('\n')):
                if i==len(text.split('\n'))-1:
                    cv2.putText(img1, line, (img1.shape[1]-200, 40), font, 1, (0,0,255),4,cv2.LINE_AA)
                else:
                    y = y0 + i*dy
                    cv2.putText(img1, line, (20, y), font, 1, (255,0,255),4,cv2.LINE_AA)

            cv2.imshow('Student Registration', img1)
            cv2.waitKey(125)
            if current_time-prev_time > 1:
                cv2.imwrite(f'{folder_path}/student_img{idx}.jpg',img) 
                idx+=1
                prev_time=current_time

            if current_time-prev_time1 > 3:
                idx1+=1
                prev_time1=current_time
 
            cur = time.time()
 
            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1

            
        
        break
 
# close the camera
cap.release()
  
# close all the opened windows
cv2.destroyAllWindows()
print("Registration done succesfully!!")