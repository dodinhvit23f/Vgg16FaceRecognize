import argparse
import imutils
import mtcnn
import cv2
import pdb
import os
from tensorflow.keras.models import load_model
import json
import numpy as np
import tensorflow as tf
from face_detection import detect_face_draw, detect_face_on_frames
#tf.config.threading.set_inter_op_parallelism_threads(3)

# python face_rec_cam.py --model "face_detector_sinhvien.h5" --classes "class_sinhvien.json"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of file .h5', default=0)
    parser.add_argument('--classes', type=str, help='Path of file .json')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    model = None
    classes = None
    
    if os.path.exists(current_dir + "/Classes/{}".format(args.classes)):
        with open(current_dir + "/Classes/{}".format(args.classes), 'r') as file:
            classes = json.load(file)
    else:
        print("There are no classes. Please insert classes.json file")
        exit()
        
    if os.path.exists(current_dir + "/model/{}".format(args.model)):
        model = load_model(current_dir + "/model/{}".format(args.model))      
    else:
        print("There are no {} file in folder model.".format(args.mode))
        exit()        
           
    # khoi tạo mạng MTCNN
    face_detector = mtcnn.MTCNN()

    #cap = cv2.VideoCapture(0) # cho win10
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # cho win7
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while (True):
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        # frame = cv2.resize(frame,(400,400))

        detect_face_draw(face_detector, frame, model, classes)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()