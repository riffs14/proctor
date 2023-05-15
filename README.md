# aicv_proctor_project

To setup project:

### Clone the repo:
git clone https://github.com/spanideaAI/aicv_proctor.git

### Go inside repo director
cd aicv_proctor

### create python environment
python3 -m venv proctor_env

### activate environment
source proctor_env/bin/activate

### Install requirements
pip install -r requirements.txt

## If you are having problems with installing the dlib library then follow the below guide:
Input these commands in the environment terminal:

1. pip install wheel

2. sudo apt-get install cmake

3. wget https://files.pythonhosted.org/packages/05/57/e8a8caa3c89a27f80bc78da39c423e2553f482a3705adc619176a3a24b36/dlib-19.17.0.tar.gz

4. tar -xvzf dlib-19.17.0.tar.gz

5. cd dlib-19.17.0/

6. sudo python3 setup.py install
   
   This command will install dlib if it was not happening earlier

7. cd ..

After this procedure you can check the installation by importing dlib in a python file and running the file, it should work without giving the error for dlib.


# NOTE : RUN STUDENT REGISTRATION FILE IN THE BEGINNING OTHERWISE THE OTHER CODE WILL NOT WORK

### To register student images
python3 student_registration.py

### To monitor student during examination
python3 ai_proctor.py <student name>

example: If name of a student is abc execute command :- python3 ai_proctor.py abc

### To train the models for use, the files are in training folder

SVM_for_front_side_during_exam.ipynb :- for training the model for checking if the student is looking at the screen or not.
One pretrained model is saved in all_models/SVM_model_during_exam.sav

face_which_side_SVM copy.py :- for training the model which will detect which direction the student is facing during the time of registration.
One pretrained model is saved in all_models/SVM_model_during_registration.sav

tiny_yolo_object_detection.py :- for training the yolo model which will detect any phones, earphones or gadgets through the screen.
One pretrained model is saved in all_models/gadget_detection_YOLOV&.pt
