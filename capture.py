import cv2


# make a video capture object
cap = cv2.VideoCapture(0)     # 0 is for webcam

while True:

    #reading the camera feed per frame
    _, frame = cap.read()

    # do anything with the frame
    
    #then show the screen which has the image in the feed
    cv2.imshow("Screen", frame)

    # set the waitkey and check if the button q is pressed, if it is then the window will be closed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# remove all the windows
cv2.destroyAllWindows()

# close the video capture object
cap.release()